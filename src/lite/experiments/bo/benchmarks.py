from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SyntheticProblem:
    name: str
    dim: int
    active_dim: int
    lower: torch.Tensor
    upper: torch.Tensor
    optimum_value: float
    kind: str

    def to(self, *, device: torch.device, dtype: torch.dtype) -> "SyntheticProblem":
        return SyntheticProblem(
            name=self.name,
            dim=self.dim,
            active_dim=self.active_dim,
            lower=self.lower.to(device=device, dtype=dtype),
            upper=self.upper.to(device=device, dtype=dtype),
            optimum_value=self.optimum_value,
            kind=self.kind,
        )

    def objective(self, X_unit: torch.Tensor) -> torch.Tensor:
        X = self._to_native(X_unit)
        if self.kind == "ackley":
            value = _ackley(X)
            return -value
        if self.kind == "levy":
            value = _levy(X)
            return -value
        if self.kind == "rastrigin":
            value = _rastrigin(X)
            return -value
        if self.kind == "rosenbrock":
            value = _rosenbrock(X)
            return -value
        if self.kind == "griewank":
            value = _griewank(X)
            return -value
        if self.kind == "hartmann6":
            return _hartmann6_reward(X)
        raise ValueError(f"Unknown synthetic problem kind: {self.kind}")

    def observe(
        self,
        X_unit: torch.Tensor,
        *,
        noise_std: float = 0.0,
        with_grad: bool = True,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        X_req = X_unit.detach().clone().requires_grad_(True)
        y = self.objective(X_req)
        grad = (
            torch.autograd.grad(y.sum(), X_req, create_graph=False)[0]
            if with_grad
            else torch.zeros_like(X_req)
        )
        if noise_std > 0.0:
            y = y + noise_std * torch.randn(
                y.shape, generator=generator, device=y.device, dtype=y.dtype
            )
        return y.detach(), grad.detach()

    def _to_native(self, X_unit: torch.Tensor) -> torch.Tensor:
        X_active = X_unit[..., : self.active_dim]
        lower = self.lower[: self.active_dim]
        upper = self.upper[: self.active_dim]
        return lower + (upper - lower) * X_active


def get_problem(
    name: str, *, device: torch.device, dtype: torch.dtype, config=None
) -> SyntheticProblem:
    key = name.lower()
    if key in {"lassodna", "lasso_dna", "lasso-dna"}:
        from lite.experiments.bo.lassodna import LassoDNAProblem

        kw = (
            {}
            if config is None
            else dict(
                data_path=config.lassodna_data_path,
                data_home=config.lassodna_data_home,
                split_seed=config.lassodna_split_seed,
                n_splits=config.lassodna_n_splits,
                test_size=config.lassodna_test_size,
                tol=config.lassodna_tol,
                max_iter=config.lassodna_max_iter,
                backend=config.lassodna_backend,
                torch_max_iter=config.lassodna_torch_max_iter,
            )
        )
        return LassoDNAProblem(device=device, dtype=dtype, **kw)
    parts = key.split("_")
    if len(parts) < 2:
        raise ValueError(f"Benchmark name must include dimension, e.g. ackley_1000. Got {name!r}.")
    dim = int(parts[-1])
    kind = "_".join(parts[:-1])
    if kind == "hartmann6":
        problem = SyntheticProblem(
            key, dim, 6, torch.zeros(dim), torch.ones(dim), 3.32237, "hartmann6"
        )
    elif kind == "ackley":
        problem = SyntheticProblem(
            key, dim, dim, torch.full((dim,), -5.0), torch.full((dim,), 5.0), 0.0, "ackley"
        )
    elif kind == "levy":
        problem = SyntheticProblem(
            key, dim, dim, torch.full((dim,), -10.0), torch.full((dim,), 10.0), 0.0, "levy"
        )
    elif kind == "rastrigin":
        problem = SyntheticProblem(
            key, dim, dim, torch.full((dim,), -5.12), torch.full((dim,), 5.12), 0.0, "rastrigin"
        )
    elif kind == "rosenbrock":
        problem = SyntheticProblem(
            key, dim, dim, torch.full((dim,), -2.05), torch.full((dim,), 2.05), 0.0, "rosenbrock"
        )
    elif kind == "griewank":
        problem = SyntheticProblem(
            key, dim, dim, torch.full((dim,), -10.0), torch.full((dim,), 10.0), 0.0, "griewank"
        )
    else:
        raise ValueError(f"Unknown benchmark {name!r}.")
    return problem.to(device=device, dtype=dtype)


def initial_design(
    n: int, dim: int, *, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    engine = torch.quasirandom.SobolEngine(dimension=dim, scramble=True, seed=int(seed))
    return engine.draw(int(n)).to(device=device, dtype=dtype).clamp(0.0, 1.0)


def _ackley(X: torch.Tensor) -> torch.Tensor:
    sq = (X * X).mean(dim=-1)
    cos = torch.cos(2.0 * math.pi * X).mean(dim=-1)
    return -20.0 * torch.exp(-0.2 * torch.sqrt(sq)) - torch.exp(cos) + 20.0 + math.e


def _levy(X: torch.Tensor) -> torch.Tensor:
    w = 1.0 + (X - 1.0) / 4.0
    term1 = torch.sin(math.pi * w[..., 0]).pow(2)
    term3 = (w[..., -1] - 1.0).pow(2) * (1.0 + torch.sin(2.0 * math.pi * w[..., -1]).pow(2))
    mid = (w[..., :-1] - 1.0).pow(2) * (1.0 + 10.0 * torch.sin(math.pi * w[..., :-1] + 1.0).pow(2))
    return term1 + mid.sum(dim=-1) + term3


def _rastrigin(X: torch.Tensor) -> torch.Tensor:
    d = X.shape[-1]
    return 10.0 * d + (X.pow(2) - 10.0 * torch.cos(2.0 * math.pi * X)).sum(dim=-1)


def _rosenbrock(X: torch.Tensor) -> torch.Tensor:
    return (100.0 * (X[..., 1:] - X[..., :-1].pow(2)).pow(2) + (1.0 - X[..., :-1]).pow(2)).sum(
        dim=-1
    )


def _griewank(X: torch.Tensor) -> torch.Tensor:
    d = X.shape[-1]
    idx = torch.arange(1, d + 1, device=X.device, dtype=X.dtype)
    sum_term = X.pow(2).sum(dim=-1) / 4000.0
    prod_term = torch.cos(X / torch.sqrt(idx)).prod(dim=-1)
    return 1.0 + sum_term - prod_term


def _hartmann6_reward(X: torch.Tensor) -> torch.Tensor:
    alpha = X.new_tensor([1.0, 1.2, 3.0, 3.2])
    A = X.new_tensor(
        [
            [10.0, 3.0, 17.0, 3.5, 1.7, 8.0],
            [0.05, 10.0, 17.0, 0.1, 8.0, 14.0],
            [3.0, 3.5, 1.7, 10.0, 17.0, 8.0],
            [17.0, 8.0, 0.05, 10.0, 0.1, 14.0],
        ]
    )
    P = 1.0e-4 * X.new_tensor(
        [
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381],
        ]
    )
    diff2 = (X[..., None, :] - P).pow(2)
    inner = (A * diff2).sum(dim=-1)
    return (alpha * torch.exp(-inner)).sum(dim=-1)
