from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from lite.methods.common.random import RandomStream, rng_scope, seed_for


@dataclass(slots=True)
class TurboState:
    length: float
    local_X: torch.Tensor
    local_y: torch.Tensor
    success_count: int = 0
    failure_count: int = 0
    restart_seed: int = 0

    @property
    def n_local(self) -> int:
        return int(self.local_X.shape[0])


def make_turbo_state(
    X_init: torch.Tensor, y_init: torch.Tensor, cfg: Any, *, seed: int
) -> TurboState:
    return TurboState(
        length=float(cfg.turbo_length_init),
        local_X=X_init.detach().clone(),
        local_y=y_init.detach().clone().reshape(-1),
        restart_seed=seed_for(seed, RandomStream.TURBO_RESTART),
    )


def restart_turbo_state(
    state: TurboState, *, dim: int, device: torch.device, dtype: torch.dtype, cfg: Any, seed: int
) -> None:
    state.length = float(cfg.turbo_length_init)
    state.success_count = 0
    state.failure_count = 0
    state.local_X = torch.empty((0, dim), device=device, dtype=dtype)
    state.local_y = torch.empty((0,), device=device, dtype=dtype)
    state.restart_seed = int(seed)


def append_turbo_observation(
    state: TurboState, candidate: torch.Tensor, y_new: torch.Tensor, cfg: Any, *, seed: int
) -> float:

    candidate = candidate.detach().reshape(-1, candidate.shape[-1])
    y_new = y_new.detach().reshape(-1)
    was_initializing = state.n_local < int(cfg.n_init)
    old_best = float(state.local_y.max().detach().cpu()) if state.n_local > 0 else -float("inf")
    state.local_X = torch.cat([state.local_X, candidate], dim=0)
    state.local_y = torch.cat([state.local_y, y_new], dim=0)
    if not was_initializing:
        _adjust_length(state, old_best, float(y_new.max().detach().cpu()), cfg)
    tr_len = float(state.length)
    if state.length < cfg.turbo_length_min:
        restart_turbo_state(
            state,
            dim=candidate.shape[-1],
            device=candidate.device,
            dtype=candidate.dtype,
            cfg=cfg,
            seed=seed_for(seed, RandomStream.TURBO_RESTART),
        )
    return tr_len


def _failure_tolerance(dim: int, cfg: Any) -> int:
    if cfg.turbo_failure_tolerance > 0:
        return int(cfg.turbo_failure_tolerance)
    return int(math.ceil(max(4.0 / max(1, cfg.batch_size), float(dim) / max(1, cfg.batch_size))))


def _adjust_length(state: TurboState, old_best: float, new_y: float, cfg: Any) -> None:
    threshold = old_best + 1.0e-3 * max(1.0, abs(old_best))
    if new_y > threshold:
        state.success_count += 1
        state.failure_count = 0
    else:
        state.success_count = 0
        state.failure_count += 1
    if state.success_count >= int(cfg.turbo_success_tolerance):
        state.length = min(float(cfg.turbo_length_max), 2.0 * state.length)
        state.success_count = 0
    if state.failure_count >= _failure_tolerance(state.local_X.shape[-1], cfg):
        state.length *= 0.5
        state.failure_count = 0


def _make_exact_gp(train_x: torch.Tensor, train_y: torch.Tensor, cfg: Any):
    from botorch.models.gpytorch import GPyTorchModel
    from gpytorch.constraints import Interval
    from gpytorch.distributions import MultivariateNormal
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.means import ConstantMean
    from gpytorch.models import ExactGP

    class GP(ExactGP, GPyTorchModel):
        _num_outputs = 1

        def __init__(
            self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: GaussianLikelihood
        ) -> None:
            ard_dims = train_x.shape[-1] if cfg.use_ard else None
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = ConstantMean()
            base_kernel = MaternKernel(
                nu=2.5,
                ard_num_dims=ard_dims,
                lengthscale_constraint=Interval(
                    float(cfg.turbo_lengthscale_min), float(cfg.turbo_lengthscale_max)
                ),
            )
            self.covar_module = ScaleKernel(
                base_kernel,
                outputscale_constraint=Interval(
                    float(cfg.turbo_outputscale_min), float(cfg.turbo_outputscale_max)
                ),
            )

        def forward(self, x: torch.Tensor) -> MultivariateNormal:
            return MultivariateNormal(self.mean_module(x), self.covar_module(x))

    likelihood = GaussianLikelihood(
        noise_constraint=Interval(float(cfg.turbo_noise_min), float(cfg.turbo_noise_max))
    ).to(device=train_x.device, dtype=train_x.dtype)
    model = GP(train_x, train_y, likelihood).to(device=train_x.device, dtype=train_x.dtype)
    model.initialize(
        **{
            "covar_module.outputscale": torch.as_tensor(
                1.0, device=train_x.device, dtype=train_x.dtype
            ),
            "covar_module.base_kernel.lengthscale": torch.as_tensor(
                0.5, device=train_x.device, dtype=train_x.dtype
            ),
            "likelihood.noise": torch.as_tensor(0.005, device=train_x.device, dtype=train_x.dtype),
        }
    )
    return model


def _fit_turbo_gp(train_x: torch.Tensor, train_y: torch.Tensor, cfg: Any):
    import gpytorch
    from gpytorch.mlls import ExactMarginalLogLikelihood

    model = _make_exact_gp(train_x, train_y, cfg)
    model.train()
    model.likelihood.train()
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.turbo_gp_lr))
    with gpytorch.settings.max_cholesky_size(int(cfg.turbo_max_cholesky_size)):
        for _ in range(max(1, int(cfg.turbo_gp_train_steps))):
            opt.zero_grad(set_to_none=True)
            output = model(train_x)
            loss = -mll(output, train_y).sum()
            loss.backward()
            opt.step()
    model.eval()
    model.likelihood.eval()
    return model


def _standardize_y(y: torch.Tensor) -> torch.Tensor:
    med = y.median()
    std = y.std(unbiased=False).clamp_min(1.0e-6)
    return (y - med) / std


def _restart_initial_candidate(
    state: TurboState, dim: int, cfg: Any, *, device: torch.device, dtype: torch.dtype, q: int = 1
) -> torch.Tensor:

    q = max(1, int(q))
    engine = torch.quasirandom.SobolEngine(
        dimension=dim, scramble=True, seed=int(state.restart_seed)
    )
    pts = engine.draw(max(int(cfg.n_init), state.n_local + q)).to(device=device, dtype=dtype)
    return pts[state.n_local : state.n_local + q].clamp(0.0, 1.0)


def select_turbo_candidate(state: TurboState, cfg: Any, *, seed: int, q: int = 1) -> torch.Tensor:
    import gpytorch

    dim = int(state.local_X.shape[-1]) if state.local_X.ndim == 2 else 0
    if dim == 0:
        raise ValueError("TurboState has invalid local_X shape.")
    device, dtype = state.local_X.device, state.local_X.dtype
    q = max(1, int(q))
    if state.n_local < int(cfg.n_init):
        return _restart_initial_candidate(state, dim, cfg, device=device, dtype=dtype, q=q)

    local_X = state.local_X
    local_y = state.local_y.reshape(-1)
    if state.n_local > int(cfg.turbo_max_local_points):
        center = local_X[torch.argmax(local_y)]
        d2 = torch.sum((local_X - center) ** 2, dim=-1)
        keep = torch.topk(-d2, k=int(cfg.turbo_max_local_points)).indices
        local_X = local_X[keep]
        local_y = local_y[keep]

    with rng_scope(seed_for(seed, RandomStream.TRAINING), device=device):
        model = _fit_turbo_gp(local_X, _standardize_y(local_y), cfg)
    x_center = local_X[torch.argmax(local_y)].detach()
    lengthscale = (
        model.covar_module.base_kernel.lengthscale.detach()
        .reshape(-1)
        .to(device=device, dtype=dtype)
    )
    if lengthscale.numel() == 1:
        weights = torch.ones(dim, device=device, dtype=dtype)
    else:
        weights = lengthscale / lengthscale.mean().clamp_min(torch.finfo(dtype).eps)
        weights = weights / torch.prod(weights.pow(1.0 / dim)).clamp_min(torch.finfo(dtype).eps)
    lb = (x_center - 0.5 * state.length * weights).clamp(0.0, 1.0)
    ub = (x_center + 0.5 * state.length * weights).clamp(0.0, 1.0)

    n_cand = (
        int(cfg.turbo_n_candidates)
        if int(cfg.turbo_n_candidates) > 0
        else min(5000, max(2000, 200 * dim))
    )
    sobol = torch.quasirandom.SobolEngine(
        dim, scramble=True, seed=seed_for(seed, RandomStream.TURBO_CANDIDATES)
    )
    pert = sobol.draw(n_cand).to(device=device, dtype=dtype)
    pert = lb + (ub - lb) * pert

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed_for(seed, RandomStream.TURBO_MASK))
    prob_perturb = min(20.0 / float(dim), 1.0)
    mask_cpu = torch.rand((n_cand, dim), generator=gen) <= prob_perturb
    empty = torch.nonzero(mask_cpu.sum(dim=1) == 0, as_tuple=False).reshape(-1)
    if empty.numel() > 0:
        cols = torch.randint(0, dim, (int(empty.numel()),), generator=gen)
        mask_cpu[empty, cols] = True
    mask = mask_cpu.to(device=device)
    X_cand = x_center.expand(n_cand, dim).clone()
    X_cand[mask] = pert[mask]

    with (
        rng_scope(seed_for(seed, RandomStream.TURBO_POSTERIOR), device=device),
        torch.no_grad(),
        gpytorch.settings.max_cholesky_size(int(cfg.turbo_max_cholesky_size)),
    ):
        samples = model.likelihood(model(X_cand)).sample(torch.Size([q])).reshape(q, n_cand)

        chosen: list[int] = []
        work = samples.clone()
        for i in range(q):
            idx = int(torch.argmax(work[i]).detach().cpu())
            chosen.append(idx)
            work[:, idx] = -float("inf")
        best = torch.as_tensor(chosen, device=device, dtype=torch.long)
    return X_cand[best].detach().clamp(0.0, 1.0)


def _local_subset_for_turbo(state: TurboState, cfg: Any) -> tuple[torch.Tensor, torch.Tensor]:
    local_X = state.local_X
    local_y = state.local_y.reshape(-1)
    if state.n_local > int(cfg.turbo_max_local_points):
        center = local_X[torch.argmax(local_y)]
        d2 = torch.sum((local_X - center) ** 2, dim=-1)
        keep = torch.topk(-d2, k=int(cfg.turbo_max_local_points)).indices
        local_X = local_X[keep]
        local_y = local_y[keep]
    return local_X, local_y


def _turbo_bounds_from_model(
    state: TurboState, local_X: torch.Tensor, local_y: torch.Tensor, model
) -> torch.Tensor:
    dim = int(local_X.shape[-1])
    device, dtype = local_X.device, local_X.dtype
    x_center = local_X[torch.argmax(local_y)].detach()
    lengthscale = (
        model.covar_module.base_kernel.lengthscale.detach()
        .reshape(-1)
        .to(device=device, dtype=dtype)
    )
    if lengthscale.numel() == 1:
        weights = torch.ones(dim, device=device, dtype=dtype)
    else:
        weights = lengthscale / lengthscale.mean().clamp_min(torch.finfo(dtype).eps)
        weights = weights / torch.prod(weights.pow(1.0 / dim)).clamp_min(torch.finfo(dtype).eps)
    lb = (x_center - 0.5 * float(state.length) * weights).clamp(0.0, 1.0)
    ub = (x_center + 0.5 * float(state.length) * weights).clamp(0.0, 1.0)
    return torch.stack([lb, ub])


def select_turbo_logei_candidate(
    state: TurboState, cfg: Any, *, seed: int
) -> tuple[torch.Tensor, float, float]:

    import time
    import warnings

    import gpytorch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.exceptions.warnings import OptimizationWarning
    from botorch.optim import optimize_acqf

    dim = int(state.local_X.shape[-1]) if state.local_X.ndim == 2 else 0
    if dim == 0:
        raise ValueError("TurboState has invalid local_X shape.")
    device, dtype = state.local_X.device, state.local_X.dtype
    if state.n_local < int(cfg.n_init):
        return (
            _restart_initial_candidate(state, dim, cfg, device=device, dtype=dtype, q=1),
            0.0,
            0.0,
        )

    local_X, local_y = _local_subset_for_turbo(state, cfg)
    train_y = _standardize_y(local_y)
    t0 = time.perf_counter()
    with rng_scope(seed_for(seed, RandomStream.TRAINING), device=device):
        model = _fit_turbo_gp(local_X, train_y, cfg)
    fit_t = time.perf_counter() - t0
    bounds = _turbo_bounds_from_model(state, local_X, local_y, model)
    best_f = train_y.max().detach()

    acqf = LogExpectedImprovement(model=model, best_f=best_f)
    raw = int(cfg.acq_raw_samples)
    restarts = int(cfg.acq_restarts)
    maxiter = int(cfg.acq_maxiter)
    t1 = time.perf_counter()
    acquisition_seed = seed_for(seed, RandomStream.ACQUISITION)
    with (
        rng_scope(acquisition_seed, device=device),
        gpytorch.settings.max_cholesky_size(int(cfg.turbo_max_cholesky_size)),
    ):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Optimization failed in `gen_candidates_scipy`.*",
                category=RuntimeWarning,
            )
            warnings.filterwarnings("ignore", category=OptimizationWarning)
            candidate, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=restarts,
                raw_samples=raw,
                options={"maxiter": maxiter, "seed": acquisition_seed},
                sequential=True,
            )
    acq_t = time.perf_counter() - t1
    return candidate.detach().clamp(0.0, 1.0), fit_t, acq_t
