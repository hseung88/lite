from __future__ import annotations

import math
from typing import Any

import torch

from lite.methods.common.bo import initial_lengthscale


def _gp_modules(
    dim: int, cfg: Any, *, device: torch.device, dtype: torch.dtype, method: str = "vbo"
):
    import gpytorch
    from gpytorch.constraints import GreaterThan
    from gpytorch.priors import LogNormalPrior

    is_vbo = method == "vbo"
    lengthscale_prior = None
    noise_prior = None
    if is_vbo and cfg.vbo_use_priors:
        loc = float(cfg.vbo_lengthscale_prior_loc) + 0.5 * math.log(float(dim))
        lengthscale_prior = LogNormalPrior(loc=loc, scale=float(cfg.vbo_lengthscale_prior_scale))
        noise_prior = LogNormalPrior(
            loc=float(cfg.vbo_noise_prior_loc), scale=float(cfg.vbo_noise_prior_scale)
        )

    kernel_name = str(cfg.vbo_kernel if is_vbo else cfg.kernel)
    ls_constraint_value = float(cfg.vbo_lengthscale_constraint if is_vbo else cfg.min_noise_var)
    noise_constraint_value = float(cfg.vbo_noise_constraint if is_vbo else cfg.min_noise_var)

    base_kwargs: dict[str, Any] = {
        "ard_num_dims": dim if cfg.use_ard else None,
        "lengthscale_constraint": GreaterThan(ls_constraint_value),
    }
    if lengthscale_prior is not None:
        base_kwargs["lengthscale_prior"] = lengthscale_prior

    if kernel_name == "matern52":
        base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, **base_kwargs).to(
            device=device, dtype=dtype
        )
    elif kernel_name == "rbf":
        base_kernel = gpytorch.kernels.RBFKernel(**base_kwargs).to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported kernel: {kernel_name}")

    if not (is_vbo and str(cfg.vbo_lengthscale_init) == "gpytorch_default"):
        ell = initial_lengthscale(dim, cfg, device=device, dtype=dtype, method=method)
        base_kernel.initialize(lengthscale=ell.view(1, -1) if cfg.use_ard else ell.view(1, 1))

    if is_vbo and not bool(cfg.vbo_use_outputscale):
        covar_module = base_kernel
    else:
        covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel,
            outputscale_constraint=GreaterThan(cfg.min_noise_var),
        ).to(device=device, dtype=dtype)
        covar_module.initialize(
            outputscale=torch.as_tensor(cfg.outputscale_init, device=device, dtype=dtype)
        )
        if is_vbo and cfg.vbo_fix_outputscale:
            covar_module.raw_outputscale.requires_grad_(False)

    like_kwargs: dict[str, Any] = {
        "noise_constraint": gpytorch.constraints.GreaterThan(noise_constraint_value)
    }
    if noise_prior is not None:
        like_kwargs["noise_prior"] = noise_prior
    likelihood = gpytorch.likelihoods.GaussianLikelihood(**like_kwargs).to(
        device=device, dtype=dtype
    )
    if (not is_vbo) or bool(cfg.vbo_initialize_noise):
        noise_init = float(cfg.vbo_noise_var_init if is_vbo else cfg.noise_var_init)
        noise_lower = float(cfg.vbo_noise_constraint if is_vbo else cfg.min_noise_var)

        noise_init = max(noise_init, 1.01 * noise_lower)
        likelihood.initialize(noise=torch.as_tensor(noise_init, device=device, dtype=dtype))
    return covar_module, likelihood


def _fit_vbo_mll(mll, cfg: Any) -> None:
    from botorch.fit import fit_gpytorch_mll

    options = {} if cfg.vbo_fit_maxiter is None else {"maxiter": int(cfg.vbo_fit_maxiter)}
    fit_gpytorch_mll(mll, optimizer_kwargs={"options": options})


def fit_single_task_gp(
    train_X: torch.Tensor, train_Y: torch.Tensor, cfg: Any, *, method: str = "vbo"
):
    from botorch.models import SingleTaskGP
    from gpytorch.mlls import ExactMarginalLogLikelihood

    train_Y = train_Y.reshape(-1, 1)
    covar_module, likelihood = _gp_modules(
        train_X.shape[-1], cfg, device=train_X.device, dtype=train_X.dtype, method=method
    )
    model = SingleTaskGP(train_X, train_Y, covar_module=covar_module, likelihood=likelihood)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    model.train()
    mll.train()
    if method == "vbo" and str(cfg.vbo_fit_backend) == "botorch_mll":
        _fit_vbo_mll(mll, cfg)
    else:
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.Adam(params, lr=cfg.gp_lr, weight_decay=cfg.gp_weight_decay)
        for _ in range(max(0, int(cfg.gp_train_steps))):
            opt.zero_grad(set_to_none=True)
            loss = -mll(model(train_X), train_Y.squeeze(-1)).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
    return model.eval()
