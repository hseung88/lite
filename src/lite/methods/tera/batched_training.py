"""Vectorized TERA conditional likelihood, preserving the scalar objective."""

from __future__ import annotations

import math

import torch

from .batched_prediction import solve_joint_batch
from .model import _alpha_from_r, _beta_from_r


def _function_covariance(X1, X2, ell, outputscale, kernel):
    # Use the training path's direct distances and Matern clamp, rather than
    # the prediction path's centered Gram distances.
    r = ((X1.unsqueeze(-2) - X2.unsqueeze(-3)) / ell).square().sum(-1)
    if kernel == "rbf":
        return outputscale * torch.exp(-0.5 * r)
    if kernel == "matern52":
        a = math.sqrt(5.0) * r.clamp_min(1e-12).sqrt()
        return outputscale * (1.0 + a + a.square() / 3.0) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel}")


def _target_solve(matrix, rhs):
    """The target-only ablation raises if the scalar jitter schedule fails."""
    eye = torch.eye(matrix.shape[-1], dtype=matrix.dtype, device=matrix.device)
    pending = torch.arange(len(matrix), device=matrix.device)
    result = torch.zeros_like(rhs)
    jitter = 1e-8
    while pending.numel():
        with torch.no_grad():
            probe, info = torch.linalg.cholesky_ex(matrix[pending] + jitter * eye)
            good = info == 0
        if good.any():
            ids = pending[good]
            factors = (
                torch.linalg.cholesky(matrix[ids] + jitter * eye)
                if torch.is_grad_enabled() and matrix.requires_grad
                else probe[good]
            )
            result = result.index_copy(0, ids, torch.cholesky_solve(rhs[ids], factors))
        pending = pending[~good]
        jitter *= 10.0
        if jitter > 1e-1:
            break
    if pending.numel():
        raise torch.linalg.LinAlgError("TERA target-only Cholesky failed after jitter retries")
    return result


def _local_observed_y_factors(
    *,
    state,
    positions,
    m,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g,
    kernel,
    gradient_noise_model,
    training_mode,
    gram_distances,
):
    targets = state.X[positions]
    batch = len(positions)
    prior_var = outputscale + sigma_f
    if m == 0:
        return targets.new_zeros(batch), prior_var.clamp_min(torch.finfo(targets.dtype).eps).expand(
            batch
        )
    ids = state.neighbors[positions, :m]
    Xc, yc, gc = state.X[ids], state.y[ids], state.g[ids]
    delta = (Xc - targets[:, None]).transpose(-1, -2)
    Xcs, xs = Xc / lengthscale, targets / lengthscale
    delta_scaled = (Xcs - xs[:, None]).transpose(-1, -2)
    H = delta_scaled.transpose(-1, -2) @ delta_scaled
    r_i = H.diagonal(dim1=-2, dim2=-1)
    r_cc = (
        (r_i.unsqueeze(-1) + r_i.unsqueeze(-2) - 2.0 * H).clamp_min(0)
        if gram_distances
        else (Xcs[:, :, None] - Xcs[:, None, :]).square().sum(-1)
    )
    alpha_i = _alpha_from_r(r_i, kernel, outputscale)
    alpha_cc = _alpha_from_r(r_cc, kernel, outputscale)
    beta_cc = _beta_from_r(r_cc, kernel, outputscale)
    eye = torch.eye(m, device=targets.device, dtype=targets.dtype)
    Kff = _function_covariance(Xc, Xc, lengthscale, outputscale, kernel)
    Kff = 0.5 * (Kff + Kff.transpose(-1, -2)) + sigma_f * eye
    k_fc = _function_covariance(Xc, targets[:, None], lengthscale, outputscale, kernel).squeeze(-1)

    if training_mode == "full":
        cols = H.transpose(-1, -2)
        q = cols[:, :, None, :] - cols[:, None, :, :]
        bar_k = (-alpha_i[:, :, None] * cols).reshape(batch, m * m)
        Q = (-alpha_cc[:, :, :, None] * q).permute(0, 1, 3, 2).reshape(batch, m * m, m)
        G = alpha_cc[:, :, :, None, None] * H[:, None, None]
        G = G + beta_cc[:, :, :, None, None] * (q.unsqueeze(-1) * q.unsqueeze(-2))
        if bool((sigma_g > 0).detach().cpu().item()):
            if gradient_noise_model == "iid":
                R = delta.transpose(-1, -2) @ delta
            else:
                delta_noise = delta * lengthscale.square().reciprocal().reshape(1, -1, 1)
                R = delta.transpose(-1, -2) @ delta_noise
            R = 0.5 * (R + R.transpose(-1, -2))
            site = torch.arange(m, device=targets.device)
            G[:, site, site] = G[:, site, site] + sigma_g * R[:, None]
        G = G.permute(0, 1, 3, 2, 4).reshape(batch, m * m, m * m)
        observed = (gc @ delta).reshape(batch, m * m)
        solve = solve_joint_batch
    else:
        bar_k = alpha_i * r_i
        Q = alpha_cc * (r_i.unsqueeze(-1) - H)
        G = alpha_cc * H + beta_cc * (H - r_i.unsqueeze(-1)) * (r_i.unsqueeze(-2) - H)
        if bool((sigma_g > 0).detach().cpu().item()):
            noise_diag = (
                delta.square().sum(-2)
                if gradient_noise_model == "iid"
                else (delta.square() * lengthscale.square().reciprocal().reshape(1, -1, 1)).sum(-2)
            )
            G = G + torch.diag_embed(sigma_g * noise_diag)
        observed = -(gc * delta.transpose(-1, -2)).sum(-1)
        solve = _target_solve

    top = torch.cat((Kff, Q.transpose(-1, -2)), dim=-1)
    bottom = torch.cat((Q, G), dim=-1)
    joint = torch.cat((top, bottom), dim=-2)
    joint = 0.5 * (joint + joint.transpose(-1, -2))
    obs = torch.cat((yc, observed), dim=-1)
    cross = torch.cat((k_fc, bar_k), dim=-1)
    solved = solve(joint, torch.stack((obs, cross), dim=-1))
    mean = (cross * solved[..., 0]).sum(-1)
    var = (prior_var - (cross * solved[..., 1]).sum(-1)).clamp_min(torch.finfo(targets.dtype).eps)
    return mean, var


def batch_nll(
    *,
    state,
    target_positions,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g,
    kernel,
    gradient_noise_model,
    training_mode="full",
    gram_distances=False,
):
    """Average conditional NLL over targets grouped by neighborhood size."""
    if training_mode not in {"full", "target"}:
        raise ValueError(f"Unknown TERA training_mode: {training_mode}")
    if gradient_noise_model not in {"iid", "scaled"}:
        raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
    positions = target_positions.to(device="cpu", dtype=torch.long)
    if positions.numel() == 0:
        raise ValueError("TERA training batch must contain at least one target")
    counts = positions.clamp_max(state.neighbors.shape[1])
    total = state.X.new_zeros(())
    for m in counts.unique().tolist():
        pos = positions[counts == m].to(state.X.device)
        mean, var = _local_observed_y_factors(
            state=state,
            positions=pos,
            m=m,
            lengthscale=lengthscale,
            outputscale=outputscale,
            sigma_f=sigma_f,
            sigma_g=sigma_g,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            training_mode=training_mode,
            gram_distances=gram_distances,
        )
        total = (
            total
            + 0.5
            * (var.log() + (state.y[pos] - mean).square() / var + math.log(2.0 * math.pi)).sum()
        )
    return total / positions.numel()
