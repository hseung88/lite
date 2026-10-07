from __future__ import annotations

import torch

from lite.experiments.gp_sim.config import ExperimentConfig
from lite.methods.common.data import SimulatedDataset
from lite.methods.common.deroos import (
    full_value_grad_covariance,
    sample_value_grad_observations_deroos,
)
from lite.methods.common.ordering import maximin_ordering
from lite.methods.common.random import RandomStream, seed_for
from lite.methods.common.utils import (
    calibrate_isotropic_lengthscale_from_inputs,
    cholesky_with_jitter,
    dtype_from_name,
    resolve_lengthscale,
    scale_inputs,
    split_value_grad,
)


def _make_design(
    n: int, d: int, *, design: str, seed: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if design == "sobol":
        eng = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
        return eng.draw(n).to(device=device, dtype=dtype).contiguous()
    if design == "random":
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        return torch.rand((n, d), generator=gen, device=device, dtype=dtype)
    raise ValueError(f"Unknown design: {design}")


def _sample_observations(
    cfg: ExperimentConfig, X_train: torch.Tensor, lengthscale: torch.Tensor, *, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
    n, d = X_train.shape
    obs_dim = n * (d + 1)
    gen = torch.Generator(device=X_train.device)
    gen.manual_seed(seed)

    if cfg.sampling == "zeros":
        z = torch.zeros(obs_dim, device=X_train.device, dtype=X_train.dtype)
        f, g = split_value_grad(z, d)
        return f, g, z, "zeros"

    if cfg.sampling == "deroos":
        n2 = n * n
        if n2 > cfg.deroos_sampling_max_n2:
            raise RuntimeError(
                f"deRoos exact sampling requires an n^2 by n^2 square-root update with n^2={n2}. "
                f"Increase deroos_sampling_max_n2 or lower n_train."
            )
        f, g, z = sample_value_grad_observations_deroos(
            X_train,
            lengthscale,
            cfg.outputscale,
            cfg.kernel,
            sigma_f=cfg.sigma_f,
            sigma_g=0.0,
            generator=gen,
        )
        if cfg.sigma_g:
            noise = torch.randn(g.shape, generator=gen, device=g.device, dtype=g.dtype)
            g = g + cfg.sigma_g * noise
            z = torch.cat([f[:, None], g], dim=-1).reshape(-1).contiguous()
        return f, g, z, "deroos"

    if cfg.sampling != "dense":
        raise ValueError(f"Unknown sampling backend: {cfg.sampling}")

    if obs_dim > cfg.dense_sampling_max_obs_dim:
        raise RuntimeError(
            f"Dense exact sampling would require an {obs_dim} x {obs_dim} covariance. "
            f"Use sampling: deroos for exact high-dimensional derivative-GP simulation, "
            f"increase dense_sampling_max_obs_dim, or lower n/d."
        )
    Kzz = full_value_grad_covariance(
        X_train,
        lengthscale,
        cfg.outputscale,
        cfg.kernel,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
    )
    L = cholesky_with_jitter(Kzz)
    eps = torch.randn((obs_dim,), generator=gen, device=X_train.device, dtype=X_train.dtype)
    z = L @ eps
    f, g = split_value_grad(z, d)
    return f, g, z, "dense"


def simulate_dataset(cfg: ExperimentConfig, d: int, *, repeat: int) -> SimulatedDataset:
    with torch.no_grad():
        device = torch.device(cfg.sampling_device or cfg.device)
        prediction_device = torch.device(cfg.device)
        dtype = dtype_from_name(cfg.dtype)

        X_train_raw = _make_design(
            cfg.n_train,
            d,
            design=cfg.design,
            seed=seed_for(cfg.seed, RandomStream.GP_TRAIN_INPUTS, repeat, d),
            device=device,
            dtype=dtype,
        )
        X_eval = _make_design(
            cfg.n_eval,
            d,
            design=cfg.design,
            seed=seed_for(cfg.seed, RandomStream.GP_EVAL_INPUTS, repeat, d),
            device=device,
            dtype=dtype,
        )

        if cfg.target_median_correlation is not None:
            if cfg.use_ard:
                raise ValueError(
                    "target_median_correlation calibration currently supports isotropic metrics only."
                )
            probe = torch.cat([X_train_raw, X_eval], dim=0)
            lengthscale = calibrate_isotropic_lengthscale_from_inputs(
                probe, cfg.kernel, cfg.target_median_correlation
            ).to(device=device, dtype=dtype)
        else:
            lengthscale = resolve_lengthscale(
                cfg.lengthscale, d, cfg.use_ard, device=device, dtype=dtype
            )

        X_train_scaled_raw = scale_inputs(X_train_raw, lengthscale)
        order = maximin_ordering(X_train_scaled_raw)
        X_train = X_train_raw[order].contiguous()
        X_train_scaled = X_train_scaled_raw[order].contiguous()
        X_eval_scaled = scale_inputs(X_eval, lengthscale).contiguous()

        observation_seed = seed_for(cfg.seed, RandomStream.GP_OBSERVATIONS, repeat, d)
        f_eval_true = None
        train_jitter = eval_jitter = 0.0
        if cfg.sample_eval:
            raise ValueError("Joint test-value sampling is not part of the retained experiments.")
        f_obs, g_obs, z_obs, backend = _sample_observations(
            cfg, X_train, lengthscale, seed=observation_seed
        )

        def move(tensor):
            return tensor.to(prediction_device).contiguous()

        return SimulatedDataset(
            X_train=move(X_train),
            X_train_scaled=move(X_train_scaled),
            X_eval=move(X_eval),
            X_eval_scaled=move(X_eval_scaled),
            lengthscale=move(lengthscale),
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            kernel_name=cfg.kernel,
            f_train_obs=move(f_obs),
            g_train_obs=move(g_obs),
            z_train_obs=move(z_obs),
            sampling_backend=backend,
            f_eval_true=None if f_eval_true is None else move(f_eval_true),
            sampling_train_jitter=train_jitter,
            sampling_eval_jitter=eval_jitter,
        )
