from __future__ import annotations

import math

import torch

from lite.methods.common.base import NeighborhoodPredictor
from lite.methods.common.covariance import condition_scalar
from lite.methods.common.data import SimulatedDataset
from lite.methods.common.deroos import function_covariance
from lite.methods.common.utils import cholesky_with_jitter, scale_inputs


class TERAPredictor(NeighborhoodPredictor):
    def __init__(self, m: int, *, rank_rtol: float | None = None) -> None:
        self.m = m
        self.rank_rtol = rank_rtol
        self.data: SimulatedDataset | None = None

    def build(self, data: SimulatedDataset) -> None:
        self.data = data

    def predict_local(
        self, target: torch.Tensor, indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")
        target = target.reshape(1, -1)
        return self._predict_one(
            x_eval=target,
            x_eval_scaled=scale_inputs(target, self.data.lengthscale),
            idx=indices,
        )

    def _predict_one(
        self, *, x_eval: torch.Tensor, x_eval_scaled: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.data is not None
        data = self.data
        device = x_eval.device
        dtype = x_eval.dtype

        m_local = int(idx.numel())
        k_xx = (
            function_covariance(
                x_eval, x_eval, data.lengthscale, data.outputscale, data.kernel_name
            )
            .reshape(())
            .to(dtype=dtype)
        )
        if m_local == 0:
            return x_eval.new_zeros(()), k_xx

        Xc = data.X_train[idx]
        Xc_scaled = data.X_train_scaled[idx]
        Kff = function_covariance(Xc, Xc, data.lengthscale, data.outputscale, data.kernel_name)
        Kff = 0.5 * (Kff + Kff.T)
        if data.sigma_f > 0.0:
            Kff = Kff + (data.sigma_f**2) * torch.eye(m_local, device=device, dtype=dtype)

        k_fc = function_covariance(
            Xc, x_eval, data.lengthscale, data.outputscale, data.kernel_name
        ).squeeze(-1)

        delta = (Xc - x_eval).T.contiguous()
        delta_scaled = (Xc_scaled - x_eval_scaled).T.contiguous()
        H = delta_scaled.T @ delta_scaled
        if self.rank_rtol is None:
            cols = H.T.contiguous()
            basis_gram = H
            directions = delta
        else:
            basis, _ = torch.linalg.qr(delta_scaled, mode="reduced")
            cols = delta_scaled.T @ basis
            basis_gram = torch.eye(basis.shape[1], device=device, dtype=dtype)
            directions = data.lengthscale.reshape(-1, 1) * basis
        r = cols.shape[1]
        q = cols[:, None, :] - cols[None, :, :]

        r_i = torch.diagonal(H, 0)
        alpha_i = _alpha_from_r(r_i, data.kernel_name, data.outputscale)

        delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
        r_cc = (delta_cc * delta_cc).sum(dim=-1)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, data.outputscale)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, data.outputscale)

        bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * r).contiguous()
        Q = (
            ((-alpha_cc[:, :, None]) * q)
            .permute(0, 2, 1)
            .reshape(m_local * r, m_local)
            .contiguous()
        )

        G0_blocks = alpha_cc[:, :, None, None] * basis_gram.view(1, 1, r, r)
        G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
        if data.sigma_g > 0.0:
            R = directions.T @ directions
            diag_idx = torch.arange(m_local, device=device)
            G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + (data.sigma_g**2) * R
        G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m_local * r, m_local * r).contiguous()

        top = torch.cat([Kff, Q.T], dim=1)
        bottom = torch.cat([Q, G0], dim=1)
        K_joint = torch.cat([top, bottom], dim=0)
        K_joint = 0.5 * (K_joint + K_joint.T)

        q_obs = (data.g_train_obs[idx] @ directions).reshape(-1).contiguous()
        obs = torch.cat([data.f_train_obs[idx], q_obs], dim=0)
        cross = torch.cat([k_fc, bar_k], dim=0)
        if self.rank_rtol is not None:
            mean, var, _ = condition_scalar(K_joint, cross, obs, k_xx, rtol=self.rank_rtol)
            return mean, var
        rhs = torch.stack([obs, cross], dim=1)
        try:
            L_joint = cholesky_with_jitter(K_joint)
            sol = torch.cholesky_solve(rhs, L_joint)
        except torch.linalg.LinAlgError:
            sol = torch.linalg.pinv(K_joint) @ rhs

        mean = torch.dot(cross, sol[:, 0])
        var = torch.clamp(k_xx - torch.dot(cross, sol[:, 1]), min=torch.finfo(dtype).eps)
        return mean, var


def _alpha_from_r(r: torch.Tensor, kernel_name: str, outputscale: float) -> torch.Tensor:
    os = outputscale
    if kernel_name == "rbf":
        return os * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return (5.0 / 3.0) * os * (1.0 + a) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def _beta_from_r(r: torch.Tensor, kernel_name: str, outputscale: float) -> torch.Tensor:
    os = outputscale
    if kernel_name == "rbf":
        return -os * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return -(25.0 / 3.0) * os * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")
