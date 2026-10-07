from __future__ import annotations

import warnings
from typing import Any

import torch

from lite.experiments.bo.config import BOConfig
from lite.methods.common.random import RandomStream, rng_scope, seed_for


def make_log_ei(model, best_f: torch.Tensor):
    from botorch.acquisition.analytic import LogExpectedImprovement

    return LogExpectedImprovement(model=model, best_f=best_f)


def _make_sobol_qmc_sampler(num_samples: int, *, seed: int | None = None):
    from botorch.sampling.normal import SobolQMCNormalSampler

    return SobolQMCNormalSampler(sample_shape=torch.Size([max(1, num_samples)]), seed=seed)


def make_vbo_acquisition(model, best_f: torch.Tensor, cfg: BOConfig, *, seed: int | None = None):
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement

    if cfg.vbo_acq in {"log_ei", "ei"}:
        return make_log_ei(model, best_f)
    if cfg.vbo_acq not in {"log_nei", "qlognei", "nei"}:
        raise ValueError(f"Unsupported VBO acquisition: {cfg.vbo_acq}")
    sampler = _make_sobol_qmc_sampler(
        cfg.vbo_nei_mc_samples,
        seed=None if seed is None else seed_for(seed, RandomStream.ACQUISITION_MC),
    )
    return qLogNoisyExpectedImprovement(
        model=model,
        X_baseline=model.train_inputs[0].detach(),
        sampler=sampler,
        prune_baseline=cfg.vbo_nei_prune_baseline,
        cache_root=cfg.vbo_nei_cache_root,
    )


def optimize_log_ei(
    model,
    bounds: torch.Tensor,
    best_f: torch.Tensor,
    cfg: BOConfig,
    *,
    seed: int,
    method: str = "vbo",
) -> torch.Tensor:
    with rng_scope(seed, device=bounds.device):
        from botorch.optim import optimize_acqf

        if method == "vbo":
            acqf = make_vbo_acquisition(model, best_f, cfg, seed=seed)
        else:
            acqf = make_log_ei(model, best_f)

        if method == "vbo":
            raw = int(cfg.vbo_acq_raw_samples)
            restarts = int(cfg.vbo_acq_restarts)
            maxiter = int(cfg.vbo_acq_maxiter)
            options: dict[str, Any] = {
                "nonnegative": False,
                "sample_around_best": bool(cfg.vbo_acq_sample_around_best),
                "sample_around_best_sigma": float(cfg.vbo_acq_sample_around_best_sigma),
                "maxiter": maxiter,
                "batch_limit": int(cfg.vbo_acq_batch_limit),
            }
            retry = bool(cfg.vbo_acq_retry_on_optimization_warning)
        elif method in {"lite"}:
            raw = int(cfg.lite_acq_raw_samples)
            restarts = int(cfg.lite_acq_restarts)
            maxiter = int(cfg.lite_acq_maxiter)
            options = {"maxiter": maxiter, "batch_limit": cfg.lite_prediction_batch_size}
            retry = True
        elif method in {"tera", "tera_batched", "tera-target", "tera-target-pred"}:
            raw = int(cfg.tera_acq_raw_samples)
            restarts = int(cfg.tera_acq_restarts)
            maxiter = int(cfg.tera_acq_maxiter)
            options = {"maxiter": maxiter}
            if method == "tera_batched":
                options["batch_limit"] = cfg.tera_prediction_batch_size
            retry = True
        else:
            raw = int(cfg.acq_raw_samples)
            restarts = int(cfg.acq_restarts)
            maxiter = int(cfg.acq_maxiter)
            options = {"maxiter": maxiter}
            retry = True

        options["seed"] = seed

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Optimization failed in `gen_candidates_scipy`.*",
            )
            warnings.filterwarnings(
                "ignore",
                category=RuntimeWarning,
                message=r"Optimization failed on the second try.*",
            )
            candidate, _ = optimize_acqf(
                acq_function=acqf,
                bounds=bounds,
                q=1,
                num_restarts=restarts,
                raw_samples=raw,
                options=options,
                sequential=True,
                retry_on_optimization_warning=retry,
            )
        return candidate.detach().clamp(bounds[0], bounds[1])
