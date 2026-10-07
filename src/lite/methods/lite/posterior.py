from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from lite.methods.common.batching import resolve_batch_size
from lite.methods.common.kernels import StationaryKernel
from lite.methods.common.linalg import cholesky_psd, solve_psd
from lite.methods.common.profiling import step_scope


@dataclass(frozen=True)
class PosteriorResult:
    mean: torch.Tensor
    var: torch.Tensor
    q_dim: int
    block_size: int
    s_interactions: int
    conditioning_dim: int


def _radial(r, outputscale, kind):
    if kind == "rbf":
        k = outputscale * torch.exp(-0.5 * r)
        return k, k, -k
    if kind == "matern52":
        a = math.sqrt(5.0) * r.clamp_min(1e-12).sqrt()
        exp = outputscale * torch.exp(-a)
        return exp * (1 + a + a.square() / 3), exp * (5 / 3) * (1 + a), exp * (-25 / 3)
    raise ValueError(f"Unknown kernel: {kind}")


def coefficient_rows(H, alpha_cc, beta_star, alpha_i):

    m = H.shape[-1]
    target = -alpha_i.unsqueeze(-1) * torch.eye(m, dtype=H.dtype, device=H.device)
    gamma = alpha_cc * beta_star.unsqueeze(-2)
    correction = gamma - torch.diag_embed(gamma.sum(-1))
    C = torch.stack((target, correction), dim=-2).reshape(*H.shape[:-2], 2 * m, m)
    sites = torch.arange(m, device=H.device).repeat_interleave(2)
    return C.contiguous(), sites


def normalize_coefficient_rows(C, scaled_delta):
    """Unit scaled-coordinate directions, preserving every nonzero row's span.

    First remove coefficient amplitude so tiny kernel weights are not squared.
    Compute lengths from explicit vectors rather than the cancellation-prone
    Gram quadratic form. Zero directions remain padded zeros.
    """
    amplitude = C.abs().amax(dim=-1, keepdim=True)
    scaled_C = C / torch.where(amplitude > 0, amplitude, torch.ones_like(amplitude))
    norm = torch.linalg.vector_norm(scaled_C @ scaled_delta, dim=-1, keepdim=True)
    normalized = scaled_C / torch.where(norm > 0, norm, torch.ones_like(norm))
    return torch.where(norm > 0, normalized, torch.zeros_like(normalized))


def conditional_moments(
    targets,
    neighbors,
    values,
    gradients,
    *,
    lengthscale,
    outputscale,
    noise_y_var,
    noise_g_var,
    kernel="rbf",
    gradient_noise_model="scaled",
    freeze_directions=False,
    coefficients=None,
    profile_regions=False,
    jitter=1e-8,
    normalize_directions=False,
    return_expected_mse=False,
):

    if gradient_noise_model not in {"iid", "scaled"}:
        raise ValueError("gradient_noise_model must be 'iid' or 'scaled'")
    ell = torch.as_tensor(lengthscale, dtype=targets.dtype, device=targets.device).reshape(-1)
    if ell.numel() not in {1, targets.shape[-1]}:
        raise ValueError("lengthscale must be scalar or have one entry per input dimension")
    batch, m = neighbors.shape[:2]
    os = torch.as_tensor(outputscale, dtype=targets.dtype, device=targets.device).reshape(())
    if m == 0:
        result = (targets.new_zeros(batch), os.expand(batch))
        return (*result, os.expand(batch)) if return_expected_mse else result

    with step_scope("gather_gram", profile_regions):
        delta = neighbors - targets[:, None]
        scaled = delta / ell
        H = scaled @ scaled.transpose(-1, -2)
        h = H.diagonal(dim1=-2, dim2=-1)
    with step_scope("assembly", profile_regions):
        r_cc = (h[:, :, None] + h[:, None, :] - 2 * H).clamp_min(0)
        Kff, alpha_cc, beta_cc = _radial(r_cc, os, kernel)
        k_fc, alpha_i, _ = _radial(h, os, kernel)
        Kff = Kff + noise_y_var * torch.eye(m, dtype=targets.dtype, device=targets.device)
    with step_scope("cholesky", profile_regions):
        Lff = cholesky_psd(Kff, jitter=jitter)
        beta_star = torch.cholesky_solve(k_fc.unsqueeze(-1), Lff).squeeze(-1)
    with step_scope("assembly", profile_regions):
        if coefficients is None:
            if freeze_directions:
                with torch.no_grad():
                    C, sites = coefficient_rows(H, alpha_cc, beta_star, alpha_i)
                    if normalize_directions:
                        C = normalize_coefficient_rows(C, scaled)
            else:
                C, sites = coefficient_rows(H, alpha_cc, beta_star, alpha_i)
                if normalize_directions:
                    C = normalize_coefficient_rows(C, scaled)
        else:
            C = coefficients
            sites = torch.arange(m, device=targets.device).repeat_interleave(2)
            if normalize_directions:
                C = normalize_coefficient_rows(C, scaled)

    with step_scope("assembly", profile_regions):
        rows = torch.arange(2 * m, device=targets.device)
        CH = C @ H
        diag_ch = CH[:, rows, sites]
        bar_k = -alpha_i[:, sites] * diag_ch
        Q = -alpha_cc[:, sites, :] * (diag_ch[:, :, None] - CH)
        c_h = diag_ch[:, :, None] - CH.index_select(-1, sites)
        G = alpha_cc[:, sites[:, None], sites[None, :]] * (CH @ C.transpose(-1, -2)) - beta_cc[
            :, sites[:, None], sites[None, :]
        ] * c_h * c_h.transpose(-1, -2)
        with step_scope("gather_gram", profile_regions):
            R = H if gradient_noise_model == "scaled" else delta @ delta.transpose(-1, -2)
        same_site = sites[:, None] == sites[None, :]
        G = G + noise_g_var * (C @ R @ C.transpose(-1, -2)) * same_site
        with step_scope("gather_gram", profile_regions):
            g_delta = gradients @ delta.transpose(-1, -2)
        observed = (C * g_delta.index_select(-2, sites)).sum(-1)

    with step_scope("cholesky", profile_regions):
        V = torch.linalg.solve_triangular(Lff, Q.transpose(-1, -2), upper=False)
    with step_scope("assembly", profile_regions):
        residual_cov = G - V.transpose(-1, -2) @ V
        residual_cov = 0.5 * (residual_cov + residual_cov.transpose(-1, -2))
        residual_cross = bar_k - (Q @ beta_star.unsqueeze(-1)).squeeze(-1)
    with step_scope("cholesky", profile_regions):
        weights = solve_psd(residual_cov, residual_cross.unsqueeze(-1), jitter=jitter).squeeze(-1)
        correction = torch.linalg.solve_triangular(
            Lff.transpose(-1, -2),
            V @ weights.unsqueeze(-1),
            upper=True,
        ).squeeze(-1)
    with step_scope("other", profile_regions):
        mean = ((beta_star - correction) * values).sum(-1) + (weights * observed).sum(-1)
        var = os - (k_fc * beta_star).sum(-1) - (weights * residual_cross).sum(-1)
        result = (mean, var.clamp_min(torch.finfo(targets.dtype).eps))
        if not return_expected_mse:
            return result
        value_weights = beta_star - correction
        cross_term = (k_fc * value_weights).sum(-1) + (bar_k * weights).sum(-1)
        quadratic = (value_weights * (Kff @ value_weights.unsqueeze(-1)).squeeze(-1)).sum(-1)
        quadratic = quadratic + 2 * (weights * (Q @ value_weights.unsqueeze(-1)).squeeze(-1)).sum(
            -1
        )
        quadratic = quadratic + (weights * (G @ weights.unsqueeze(-1)).squeeze(-1)).sum(-1)
        # Risk under the original covariance, including the actual jittered weights.
        return (*result, os - 2 * cross_term + quadratic)


@torch.no_grad()
def nearest_neighbors(train_scaled, targets_scaled, m, chunk_size=4096):

    k = min(max(int(m), 0), len(train_scaled))
    batch = len(targets_scaled)
    best = targets_scaled.new_empty(batch, 0)
    indices = torch.empty(batch, 0, dtype=torch.long, device=targets_scaled.device)
    if k == 0:
        return indices
    for start in range(0, len(train_scaled), chunk_size):
        distances = torch.cdist(targets_scaled, train_scaled[start : start + chunk_size])
        candidate_ids = torch.arange(
            start, start + distances.shape[-1], device=indices.device
        ).expand(batch, -1)
        candidates = torch.cat((best, distances), dim=-1)
        ids = torch.cat((indices, candidate_ids), dim=-1)
        best, chosen = candidates.topk(min(k, candidates.shape[-1]), largest=False, dim=-1)
        indices = ids.gather(-1, chosen)
    return indices


def predict_marginals(
    targets,
    train_X,
    train_y,
    train_g,
    *,
    lengthscale,
    outputscale,
    noise_y_var,
    noise_g_var,
    m,
    kernel="rbf",
    gradient_noise_model="scaled",
    prediction_batch_size=None,
    neighborhoods=None,
    train_scaled=None,
    jitter=1e-8,
    batch_size=None,
    normalize_directions=False,
    return_expected_mse=False,
):

    prediction_batch_size = resolve_batch_size(
        prediction_batch_size, batch_size, default=256, name="prediction_batch_size"
    )
    ell = torch.as_tensor(lengthscale, dtype=train_X.dtype, device=train_X.device)
    if train_scaled is None:
        train_scaled = train_X / ell
    if neighborhoods is not None and len(neighborhoods) != len(targets):
        raise ValueError("neighborhood count must match target count")
    means, variances, risks = [], [], []
    for start in range(0, len(targets), prediction_batch_size):
        xb = targets[start : start + prediction_batch_size]
        if neighborhoods is None:
            ids = nearest_neighbors(train_scaled, xb.detach() / ell.detach(), m)
        else:
            ids = torch.stack(neighborhoods[start : start + prediction_batch_size]).to(
                device=train_X.device
            )
        result = conditional_moments(
            xb,
            train_X[ids],
            train_y[ids],
            train_g[ids],
            lengthscale=ell,
            outputscale=outputscale,
            noise_y_var=noise_y_var,
            noise_g_var=noise_g_var,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            jitter=jitter,
            normalize_directions=normalize_directions,
            return_expected_mse=return_expected_mse,
        )
        mean, var = result[:2]
        if return_expected_mse:
            risks.append(result[2])
        means.append(mean)
        variances.append(var)
    if not means:
        result = (targets.new_empty(0), targets.new_empty(0))
        return (*result, targets.new_empty(0)) if return_expected_mse else result
    result = (torch.cat(means), torch.cat(variances))
    return (*result, torch.cat(risks)) if return_expected_mse else result


def predict(
    x_test,
    x_train,
    y_obs,
    g_obs,
    kernel: StationaryKernel,
    noise_y,
    noise_g,
    m,
    *,
    neighborhoods=None,
    x_train_scaled=None,
    prediction_batch_size=None,
    batch_size=None,
):

    prediction_batch_size = resolve_batch_size(
        prediction_batch_size, batch_size, default=256, name="prediction_batch_size"
    )
    if neighborhoods is not None and len({len(ids) for ids in neighborhoods}) > 1:
        raise ValueError("batched LITE requires equal-size neighborhoods")
    mean, var = predict_marginals(
        x_test,
        x_train,
        y_obs,
        g_obs,
        lengthscale=kernel.lengthscale,
        outputscale=kernel.outputscale,
        noise_y_var=noise_y**2,
        noise_g_var=noise_g**2,
        kernel=kernel.kind,
        gradient_noise_model="iid",
        m=m,
        prediction_batch_size=prediction_batch_size,
        neighborhoods=neighborhoods,
        train_scaled=x_train_scaled,
        jitter=kernel.jitter,
    )
    m_eff = len(neighborhoods[0]) if neighborhoods else min(max(m, 0), len(x_train))
    if len(x_test) == 0:
        m_eff = 0
    return PosteriorResult(mean, var, 2 * m_eff, 2 * m_eff, m_eff, 3 * m_eff)
