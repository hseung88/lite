"""Batched full TERA prediction with the original joint-covariance formulation."""

from __future__ import annotations

import torch

from lite.methods.common.data import PredictiveMarginals
from lite.methods.common.deroos import k_kp_kpp_from_r
from lite.methods.common.ordering import knn_to_eval
from lite.methods.common.utils import scale_inputs
from lite.methods.tera.model import TERAPredictor, _alpha_from_r, _beta_from_r


def solve_joint_batch(matrix, rhs):
    """Use the scalar TERA jitter schedule independently for each target."""
    eye = torch.eye(matrix.shape[-1], device=matrix.device, dtype=matrix.dtype)
    # Reuse the differentiable factor when the whole batch succeeds. Selecting
    # only successful factors from a mixed batch is unsafe: Cholesky backward
    # can produce NaNs for failed factors even with zero upstream gradients.
    factors, info = torch.linalg.cholesky_ex(matrix + 1e-8 * eye)
    if not info.any():
        return torch.cholesky_solve(rhs, factors)
    del factors, info

    pending = torch.arange(len(matrix), device=matrix.device)
    result = torch.zeros_like(rhs)
    jitter = 1e-8
    while pending.numel():
        # Failed Cholesky factors must not enter autograd: even zero upstream
        # gradients can produce NaNs in their backward solves.
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
            solved = torch.cholesky_solve(rhs[ids], factors)
            result = result.index_copy(0, ids, solved)
        pending = pending[~good]
        jitter *= 10.0
        if jitter > 1e-1:
            break
    if pending.numel():
        solved = torch.linalg.pinv(matrix[pending]) @ rhs[pending]
        result = result.index_copy(0, pending, solved)
    return result


def function_covariance_batch(X1, X2, ell, outputscale, kernel, *, centered=True):
    if not centered:
        r = ((X1.unsqueeze(-2) - X2.unsqueeze(-3)) / ell).square().sum(-1)
        return k_kp_kpp_from_r(r, kernel, outputscale)[0]
    center = 0.5 * (X1.mean(dim=-2, keepdim=True) + X2.mean(dim=-2, keepdim=True))
    x1, x2 = (X1 - center) / ell, (X2 - center) / ell
    r = x1.square().sum(-1, keepdim=True) + x2.square().sum(-1).unsqueeze(-2)
    r = (r - 2.0 * (x1 @ x2.transpose(-1, -2))).clamp_min(0)
    return k_kp_kpp_from_r(r, kernel, outputscale)[0]


class BatchedTERAPredictor(TERAPredictor):
    def __init__(self, m, gradient_noise_model, prediction_batch_size):
        super().__init__(m, gradient_noise_model)
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.prediction_batch_size = prediction_batch_size

    @torch.no_grad()
    def predict_f_marginals(self, X_eval, *, neighborhoods=None, return_expected_mse=False):
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")
        if not len(X_eval):
            return PredictiveMarginals(
                mean=X_eval.new_empty(0),
                var=X_eval.new_empty(0),
                expected_mse=X_eval.new_empty(0) if return_expected_mse else None,
            )
        data = self.data
        scaled = scale_inputs(X_eval, data.lengthscale)

        if neighborhoods is None:
            neighborhoods = knn_to_eval(data.X_train_scaled, scaled, self.m)
        elif len(neighborhoods) != len(X_eval):
            raise ValueError("neighborhood count must match target count")
        means, variances, risks = [], [], []
        for start in range(0, len(X_eval), self.prediction_batch_size):
            stop = start + self.prediction_batch_size
            ids = torch.stack(neighborhoods[start:stop])
            result = self._predict_batch(
                X_eval[start:stop], scaled[start:stop], ids, return_expected_mse=return_expected_mse
            )
            mean, var = result[:2]
            if return_expected_mse:
                risks.append(result[2])
            means.append(mean)
            variances.append(var)
        return PredictiveMarginals(
            mean=torch.cat(means),
            var=torch.cat(variances),
            expected_mse=torch.cat(risks) if return_expected_mse else None,
        )

    def _predict_batch(
        self, targets, targets_scaled, ids, *, centered=True, return_expected_mse=False
    ):
        data = self.data
        batch, m = ids.shape
        os = targets.new_tensor(data.outputscale)
        if m == 0:
            result = (targets.sum(-1) * 0.0, os.expand(batch) + targets.sum(-1) * 0.0)
            return (*result, os.expand(batch)) if return_expected_mse else result
        Xc = data.X_train[ids]
        Xcs = data.X_train_scaled[ids]
        delta = (Xc - targets[:, None]).transpose(-1, -2)
        delta_scaled = (Xcs - targets_scaled[:, None]).transpose(-1, -2)
        H = delta_scaled.transpose(-1, -2) @ delta_scaled
        cols = H.transpose(-1, -2)
        q = cols[:, :, None, :] - cols[:, None, :, :]
        r_i = H.diagonal(dim1=-2, dim2=-1)
        r_cc = (Xcs[:, :, None] - Xcs[:, None, :]).square().sum(-1)
        alpha_i = _alpha_from_r(r_i, data.kernel_name, os)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, os)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, os)

        ell = data.lengthscale
        Kff = function_covariance_batch(Xc, Xc, ell, os, data.kernel_name, centered=centered)
        k_fc = function_covariance_batch(
            Xc, targets[:, None], ell, os, data.kernel_name, centered=centered
        ).squeeze(-1)
        Kff = Kff + data.sigma_f**2 * torch.eye(m, dtype=targets.dtype, device=targets.device)
        bar_k = (-alpha_i[:, :, None] * cols).reshape(batch, m * m)
        Q = (-alpha_cc[:, :, :, None] * q).permute(0, 1, 3, 2).reshape(batch, m * m, m)
        G = alpha_cc[:, :, :, None, None] * H[:, None, None]
        G = G + beta_cc[:, :, :, None, None] * q.unsqueeze(-1) * q.unsqueeze(-2)
        if self.gradient_noise_model == "iid":
            R = delta.transpose(-1, -2) @ delta
        else:
            delta_noise = delta * ell.square().reciprocal().reshape(1, -1, 1)
            R = delta.transpose(-1, -2) @ delta_noise
        R = 0.5 * (R + R.transpose(-1, -2))
        site = torch.arange(m, device=targets.device)
        G[:, site, site] = G[:, site, site] + data.sigma_g * R[:, None]
        G = G.permute(0, 1, 3, 2, 4).reshape(batch, m * m, m * m)
        observed = (data.g_train_obs[ids] @ delta).reshape(batch, m * m)
        top = torch.cat((Kff, Q.transpose(-1, -2)), dim=-1)
        bottom = torch.cat((Q, G), dim=-1)
        joint = torch.cat((top, bottom), dim=-2)
        joint = 0.5 * (joint + joint.transpose(-1, -2))
        obs = torch.cat((data.f_train_obs[ids], observed), dim=-1)
        cross = torch.cat((k_fc, bar_k), dim=-1)
        solved = solve_joint_batch(joint, torch.stack((obs, cross), dim=-1))
        mean = (cross * solved[..., 0]).sum(-1)
        var = os - (cross * solved[..., 1]).sum(-1)
        result = (mean, var.clamp_min(torch.finfo(targets.dtype).eps))
        if not return_expected_mse:
            return result
        weights = solved[..., 1]
        quadratic = (weights * (joint @ weights.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return (*result, os - 2 * (cross * weights).sum(-1) + quadratic)
