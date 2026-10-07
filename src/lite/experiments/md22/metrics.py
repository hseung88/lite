from __future__ import annotations

import torch


def normalized_energy_rmse_per_atom(
    y_pred: torch.Tensor, y_true: torch.Tensor, n_atoms: int
) -> float:

    rmse = torch.sqrt(torch.mean((y_pred.reshape(-1) - y_true.reshape(-1)).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())


def energy_rmse_per_atom(E_pred: torch.Tensor, E_true: torch.Tensor, n_atoms: int) -> float:

    E_pred = E_pred.reshape(-1)
    E_true = E_true.to(device=E_pred.device, dtype=E_pred.dtype).reshape(-1)
    rmse = torch.sqrt(torch.mean((E_pred - E_true).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())


def raw_energy_rmse_per_atom(
    y_pred: torch.Tensor,
    E_true: torch.Tensor,
    *,
    energy_mean: torch.Tensor | float,
    energy_std: torch.Tensor | float,
    n_atoms: int,
) -> float:

    mean = torch.as_tensor(energy_mean, device=y_pred.device, dtype=torch.float64)
    std = torch.as_tensor(energy_std, device=y_pred.device, dtype=torch.float64)
    E_pred = mean - std * y_pred.double().reshape(-1)
    E_true = E_true.to(device=y_pred.device, dtype=torch.float64).reshape(-1)
    rmse = torch.sqrt(torch.mean((E_pred - E_true).pow(2)))
    return float((rmse / float(n_atoms)).detach().cpu())
