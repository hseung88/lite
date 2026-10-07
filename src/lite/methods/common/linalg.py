from __future__ import annotations

import torch


def cholesky_psd(a: torch.Tensor, jitter: float = 1e-8, max_tries: int = 8) -> torch.Tensor:
    eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
    shift = a.new_full(a.shape[:-2], jitter)
    factor, info = torch.linalg.cholesky_ex(a + shift[..., None, None] * eye)
    if not info.any():
        return factor
    with torch.no_grad():
        for _ in range(max_tries):
            shift = torch.where(info > 0, shift * 10, shift)
            _, info = torch.linalg.cholesky_ex(a + shift[..., None, None] * eye)
            if not info.any():
                break
    return torch.linalg.cholesky(a + shift[..., None, None] * eye)


def solve_psd(a: torch.Tensor, rhs: torch.Tensor, jitter: float = 1e-8) -> torch.Tensor:
    try:
        return torch.cholesky_solve(rhs, cholesky_psd(a, jitter=jitter))
    except torch.linalg.LinAlgError:
        if a.ndim == 2:
            return torch.linalg.pinv(a, hermitian=True) @ rhs
        matrices = a.reshape(-1, *a.shape[-2:])
        vectors = rhs.reshape(-1, *rhs.shape[-2:])
        return torch.stack(
            [solve_psd(matrix, vector, jitter) for matrix, vector in zip(matrices, vectors)]
        ).reshape(rhs.shape)


def conditional_gaussian(
    prior_mean: torch.Tensor,
    prior_var: torch.Tensor,
    obs: torch.Tensor,
    obs_mean: torch.Tensor,
    obs_cov: torch.Tensor,
    cross: torch.Tensor,
    jitter: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    L = cholesky_psd(obs_cov, jitter=jitter)
    alpha = torch.cholesky_solve((obs - obs_mean)[:, None], L).squeeze(-1)
    mean = prior_mean + cross @ alpha
    solved = torch.cholesky_solve(cross.T, L)
    var = prior_var - (cross * solved.T).sum(dim=-1)
    return mean, var.clamp_min(jitter)
