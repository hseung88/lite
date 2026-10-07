from __future__ import annotations

import torch

from lite.methods.common.deroos import function_covariance
from lite.methods.common.utils import cholesky_with_jitter, resolve_lengthscale


def median_pairwise_lengthscale(X: torch.Tensor, *, max_points: int = 2048) -> torch.Tensor:

    n = int(X.shape[0])
    if n < 2:
        return torch.ones(1, device=X.device, dtype=X.dtype)
    max_points = int(max_points)
    if max_points > 0 and n > max_points:
        idx = torch.linspace(0, n - 1, steps=max_points, device=X.device).round().long()
        X_ref = X[idx].contiguous()
    else:
        X_ref = X
    with torch.no_grad():
        dist = torch.pdist(X_ref, p=2.0)
        positive = dist[dist > 0]
        if positive.numel() == 0:
            return torch.ones(1, device=X.device, dtype=X.dtype)
        ell = torch.median(positive).clamp_min(torch.finfo(X.dtype).eps)
    return ell.reshape(1).to(device=X.device, dtype=X.dtype)


def resolve_kernel_lengthscale(
    X: torch.Tensor,
    *,
    lengthscale: float | list[float] | None,
    lengthscale_init: str = "median",
    lengthscale_init_max_points: int = 2048,
    use_ard: bool = False,
) -> torch.Tensor:
    if lengthscale is not None:
        return resolve_lengthscale(lengthscale, X.shape[1], use_ard, device=X.device, dtype=X.dtype)
    if lengthscale_init == "median":
        ell = median_pairwise_lengthscale(X, max_points=lengthscale_init_max_points)
    elif lengthscale_init == "one":
        ell = torch.ones(1, device=X.device, dtype=X.dtype)
    else:
        raise ValueError(
            f"Unknown lengthscale_init={lengthscale_init!r}. Supported values are 'median' and 'one'."
        )
    if use_ard:
        return ell.repeat(X.shape[1]).contiguous()
    return ell


def kff_cholesky(
    X: torch.Tensor, lengthscale: torch.Tensor, outputscale, sigma_f, kernel: str
) -> torch.Tensor:
    K = function_covariance(X, X, lengthscale, outputscale, kernel)
    K = 0.5 * (K + K.T)
    sf = torch.as_tensor(sigma_f, device=X.device, dtype=X.dtype)
    if bool((sf > 0.0).detach().cpu().item()):
        K = K + sf * torch.eye(X.shape[0], device=X.device, dtype=X.dtype)
    return cholesky_with_jitter(K)


__all__ = [
    "function_covariance",
    "median_pairwise_lengthscale",
    "resolve_kernel_lengthscale",
    "kff_cholesky",
]
