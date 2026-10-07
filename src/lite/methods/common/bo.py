from __future__ import annotations

import math
from typing import Any

import torch


def initial_lengthscale(
    dim: int,
    cfg: Any,
    *,
    device: torch.device,
    dtype: torch.dtype,
    method: str = "vbo",
) -> torch.Tensor:
    if method == "vbo":
        init = cfg.vbo_lengthscale_init
        base = cfg.vbo_base_lengthscale
        lower = getattr(cfg, "vbo_lengthscale_constraint", 1.0e-4)
        upper = None
    elif method == "tera":
        init = cfg.tera_lengthscale_init
        base = cfg.tera_base_lengthscale
        lower = cfg.tera_lengthscale_min
        upper = cfg.tera_lengthscale_max
    else:
        init = cfg.lengthscale_init
        base = cfg.base_lengthscale
        lower = cfg.min_noise_var
        upper = None

    if init == "d_scaled":
        value = base * math.sqrt(float(dim))
    elif init == "prior_mode":
        if method != "vbo":
            raise ValueError("lengthscale_init='prior_mode' is only defined for VBO.")

        value = math.exp(
            float(cfg.vbo_lengthscale_prior_loc)
            + 0.5 * math.log(float(dim))
            - float(cfg.vbo_lengthscale_prior_scale) ** 2
        )
    elif init == "base":
        value = base
    elif init in {"gpytorch_default", "interval_midpoint"}:
        if upper is None:
            value = float(lower) + math.log(2.0)
        else:
            value = 0.5 * (float(lower) + float(upper))
    else:
        value = float(init)

    shape = (dim,) if cfg.use_ard else (1,)
    return torch.full(shape, float(value), device=device, dtype=dtype)
