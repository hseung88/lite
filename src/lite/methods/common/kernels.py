from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

KernelKind = Literal["rbf", "matern52"]


@dataclass(frozen=True)
class StationaryKernel:
    kind: KernelKind = "rbf"
    lengthscale: float = 1.0
    outputscale: float = 1.0
    jitter: float = 1e-8

    def _lambda(self, x: torch.Tensor) -> torch.Tensor:
        return torch.as_tensor(1.0 / (self.lengthscale**2), dtype=x.dtype, device=x.device)

    def scaled_sqdist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x2 = x.square().sum(dim=-1, keepdim=True)
        y2 = y.square().sum(dim=-1, keepdim=True).T
        return (x2 + y2 - 2.0 * x @ y.T).clamp_min(0.0) / (self.lengthscale**2)

    def _phi(self, s: torch.Tensor) -> torch.Tensor:
        if self.kind == "rbf":
            return self.outputscale * torch.exp(-0.5 * s)
        if self.kind == "matern52":
            r = torch.sqrt(s.clamp_min(0.0) + self.jitter)
            a = torch.sqrt(torch.as_tensor(5.0, dtype=s.dtype, device=s.device))
            return self.outputscale * (1.0 + a * r + (5.0 / 3.0) * s) * torch.exp(-a * r)
        raise ValueError(f"Unknown kernel: {self.kind}")

    def _phi_prime(self, s: torch.Tensor) -> torch.Tensor:
        if self.kind == "rbf":
            return -0.5 * self.outputscale * torch.exp(-0.5 * s)
        if self.kind == "matern52":
            r = torch.sqrt(s.clamp_min(0.0) + self.jitter)
            a = torch.sqrt(torch.as_tensor(5.0, dtype=s.dtype, device=s.device))
            return self.outputscale * (-(5.0 / 6.0) * (1.0 + a * r) * torch.exp(-a * r))
        raise ValueError(f"Unknown kernel: {self.kind}")

    def _phi_second(self, s: torch.Tensor) -> torch.Tensor:
        if self.kind == "rbf":
            return 0.25 * self.outputscale * torch.exp(-0.5 * s)
        if self.kind == "matern52":
            r = torch.sqrt(s.clamp_min(0.0) + self.jitter)
            a = torch.sqrt(torch.as_tensor(5.0, dtype=s.dtype, device=s.device))
            return self.outputscale * ((25.0 / 12.0) * torch.exp(-a * r))
        raise ValueError(f"Unknown kernel: {self.kind}")

    def cov_ff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self._phi(self.scaled_sqdist(x, y))

    def cov_f_dir(self, x: torch.Tensor, sites: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:

        if sites.numel() == 0:
            return torch.empty(x.shape[0], 0, dtype=x.dtype, device=x.device)
        lam = self._lambda(x)
        r = x[:, None, :] - sites[None, :, :]
        s = lam * r.square().sum(dim=-1)
        dot = (dirs[None, :, :] * (lam * r)).sum(dim=-1)
        return (-2.0 * self._phi_prime(s)) * dot

    def cov_dir_dir(
        self,
        sites_a: torch.Tensor,
        dirs_a: torch.Tensor,
        sites_b: torch.Tensor,
        dirs_b: torch.Tensor,
    ) -> torch.Tensor:

        if sites_a.numel() == 0 or sites_b.numel() == 0:
            return torch.empty(
                sites_a.shape[0], sites_b.shape[0], dtype=sites_a.dtype, device=sites_a.device
            )
        lam = self._lambda(sites_a)
        r = sites_a[:, None, :] - sites_b[None, :, :]
        s = lam * r.square().sum(dim=-1)
        dir_dot = lam * (dirs_a[:, None, :] * dirs_b[None, :, :]).sum(dim=-1)
        u_lam_r = (dirs_a[:, None, :] * (lam * r)).sum(dim=-1)
        v_lam_r = (dirs_b[None, :, :] * (lam * r)).sum(dim=-1)
        return (-2.0 * self._phi_prime(s)) * dir_dot - 4.0 * self._phi_second(s) * u_lam_r * v_lam_r


def _matern52_corr(tau: float) -> float:
    import math

    a = math.sqrt(5.0) * tau
    return (1.0 + a + a * a / 3.0) * math.exp(-a)


def tau_from_target_median_correlation(kernel_kind: KernelKind, target_corr: float) -> float:
    import math

    if not (0.0 < target_corr < 1.0):
        raise ValueError(f"target_corr must satisfy 0<target_corr<1, got {target_corr}")
    if kernel_kind == "rbf":
        return math.sqrt(-2.0 * math.log(target_corr))
    if kernel_kind == "matern52":
        lo, hi = 0.0, 1.0
        while _matern52_corr(hi) > target_corr:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _matern52_corr(mid) > target_corr:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    raise ValueError(f"Unknown kernel: {kernel_kind}")


def calibrate_lengthscale(
    x: torch.Tensor, kernel_kind: KernelKind, target_corr: float = 0.3
) -> float:

    if x.shape[0] < 2:
        raise ValueError("Need at least two inputs to calibrate lengthscale.")
    with torch.no_grad():
        d2 = torch.pdist(x, p=2.0).pow(2)
        med = torch.median(d2)
        tau = tau_from_target_median_correlation(kernel_kind, target_corr)
        ell = torch.sqrt(med / x.new_tensor(tau * tau))
        return float(ell.item())
