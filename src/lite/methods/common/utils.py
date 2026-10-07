from __future__ import annotations

import math
from typing import Iterable

import torch


def dtype_from_name(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype name: {name}")


def _matern52_corr(tau: float) -> float:
    a = math.sqrt(5.0) * tau
    return (1.0 + a + a * a / 3.0) * math.exp(-a)


def tau_from_target_median_correlation(kernel: str, target_median_correlation: float) -> float:
    if not (0.0 < target_median_correlation < 1.0):
        raise ValueError(
            f"target_median_correlation must satisfy 0<target_median_correlation<1, got {target_median_correlation}"
        )
    if kernel == "rbf":
        return math.sqrt(-2.0 * math.log(target_median_correlation))
    if kernel == "matern52":
        lo, hi = 0.0, 1.0
        while _matern52_corr(hi) > target_median_correlation:
            hi *= 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if _matern52_corr(mid) > target_median_correlation:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)
    raise ValueError(f"Unknown kernel name: {kernel}")


def calibrate_isotropic_lengthscale_from_inputs(
    X: torch.Tensor, kernel: str, target_median_correlation: float
) -> torch.Tensor:

    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least two inputs to calibrate target_median_correlation.")
    with torch.no_grad():
        d2 = torch.pdist(X, p=2.0).pow(2)
        med = torch.median(d2)
        tau = tau_from_target_median_correlation(kernel, target_median_correlation)
        ell = torch.sqrt(med / X.new_tensor(tau * tau))
        return ell.reshape(1)


def resolve_lengthscale(
    lengthscale: float | Iterable[float] | None,
    d: int,
    use_ard: bool,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if lengthscale is None:
        return torch.tensor([0.2], device=device, dtype=dtype)
    if isinstance(lengthscale, (float, int)):
        return torch.tensor([float(lengthscale)], device=device, dtype=dtype)
    vals = torch.tensor(list(lengthscale), device=device, dtype=dtype)
    if not use_ard:
        return vals[:1].clone()
    if vals.numel() == 1:
        return vals.reshape(1)
    if vals.numel() < d:
        vals = torch.cat([vals, vals[-1].repeat(d - vals.numel())])
    elif vals.numel() > d:
        vals = vals[:d]
    return vals


def scale_inputs(X: torch.Tensor, lengthscale: torch.Tensor) -> torch.Tensor:
    if lengthscale.numel() == 1:
        return X / lengthscale.reshape(1)
    return X / lengthscale.view(1, -1)


def cholesky_with_jitter(
    K: torch.Tensor, jitter0: float = 1e-8, jitter_max: float = 1e-1
) -> torch.Tensor:
    jitter = jitter0
    eye = torch.eye(K.shape[-1], dtype=K.dtype, device=K.device)
    while True:
        try:
            return torch.linalg.cholesky(K + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter *= 10.0
            if jitter > jitter_max:
                raise


def function_output_indices(n: int, d: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, n * (d + 1), step=d + 1, device=device, dtype=torch.long)


def split_value_grad(z: torch.Tensor, d: int) -> tuple[torch.Tensor, torch.Tensor]:
    z2 = z.reshape(-1, d + 1)
    return z2[:, 0].contiguous(), z2[:, 1:].contiguous()
