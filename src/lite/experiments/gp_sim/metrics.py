from __future__ import annotations

import torch


def gaussian_kl_1d(
    mu_ref: torch.Tensor, var_ref: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    eps = torch.finfo(var.dtype).eps
    var_ref = torch.clamp(var_ref, min=eps)
    var = torch.clamp(var, min=eps)
    return 0.5 * (torch.log(var / var_ref) + (var_ref + (mu_ref - mu).pow(2)) / var - 1.0)


def sum_marginal_kl(
    mu_ref: torch.Tensor, var_ref: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    return gaussian_kl_1d(mu_ref, var_ref, mu, var).sum()


def average_marginal_kl(
    mu_ref: torch.Tensor, var_ref: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    return gaussian_kl_1d(mu_ref, var_ref, mu, var).mean()
