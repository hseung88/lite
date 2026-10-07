from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import yaml

from lite.experiments.config import normalize_config
from lite.experiments.md22.observation_noise import config_noise_fractions
from lite.experiments.options import expand_yaml, method_prefix
from lite.methods.names import normalize_md22_method_name


@dataclass(slots=True)
class MD22Config:
    def __post_init__(self):
        config_noise_fractions(self)
        self.methods = list(dict.fromkeys(map(normalize_md22_method_name, self.methods)))

    experiment_name: str = "lite.experiments.md22"
    data_dir: str = "data/md22"
    datasets: list[str] = field(default_factory=lambda: ["DHA"])
    methods: list[str] = field(
        default_factory=lambda: ["standard_gp", "dsoftki", "ddsvgp", "tera", "tera_batched", "lite"]
    )
    seeds: list[int] = field(default_factory=lambda: [6535, 8830, 92357])
    train_frac: float = 0.9
    test_frac: float = 0.1
    n_train: int | None = None
    n_test: int | None = None
    device: str = "cuda"
    dtype: str = "float32"
    kernel: str = "rbf"
    outputscale: float = 1.0

    sigma_f: float = 1e-3
    sigma_g: float = 1e-3
    observation_noise_fraction: float = 0.0
    energy_noise_fraction: float | None = None
    force_noise_fraction: float | None = None
    observation_noise_seed: int | None = None
    initialize_noise_from_observations: bool = True
    lengthscale: float | list[float] | None = None
    lengthscale_init: str = "median"
    lengthscale_init_max_points: int = 2048
    use_ard: bool = False
    x_scale: float = 3.0
    preprocessing_version: str = "md22_v2_float64_train_stats_chain_rule"
    m: int = 20
    lite_m: int | None = None
    tera_m: int | None = None
    vecchia_m: int | None = None

    standard_gp_max_train: int = 100000
    out_csv_name: str = "results.csv"
    verbose: bool = False
    log_training_curves: bool = False
    curve_log_every: int = 5

    standard_gp_train_epochs: int = 0
    standard_gp_train_steps: int = 50

    standard_gp_lr: float | None = None
    standard_gp_lr_default: float = 0.01
    standard_gp_lr_by_dataset: dict[str, float] = field(
        default_factory=lambda: {"double-walled-nanotube": 0.005}
    )
    standard_gp_weight_decay: float = 0.0
    standard_gp_learn_lengthscale: bool = True
    standard_gp_learn_outputscale: bool = True
    standard_gp_learn_sigma_f: bool = True
    standard_gp_min_sigma_f: float = 1e-6
    standard_gp_log_every: int = 0

    tera_train_steps: int = 0
    tera_train_epochs: int = 1

    tera_graph_refresh_epochs: int = 0
    tera_prediction_batch_size: int = 256
    tera_train_batch_size: int = 256
    tera_lr: float = 0.01
    tera_weight_decay: float = 0.0
    tera_learn_lengthscale: bool = True
    tera_learn_outputscale: bool = True
    tera_learn_sigma_f: bool = True
    tera_learn_sigma_g: bool = True
    tera_min_sigma_f: float = 1e-6
    tera_min_sigma_g: float = 0.0
    tera_log_every: int = 0
    tera_gradient_noise_model: Literal["iid", "scaled"] = "iid"

    lite_train_steps: int = 0
    lite_train_epochs: int = 1

    lite_graph_refresh_epochs: int = 0
    lite_train_batch_size: int = 256
    lite_lr: float = 0.01
    lite_weight_decay: float = 0.0
    lite_learn_lengthscale: bool = True
    lite_learn_outputscale: bool = True
    lite_learn_sigma_f: bool = True
    lite_learn_sigma_g: bool = True
    lite_min_sigma_f: float = 1e-6
    lite_min_sigma_g: float = 0.0
    lite_log_every: int = 0
    lite_gradient_noise_model: Literal["iid", "scaled"] = "iid"
    lite_prediction_batch_size: int = 256

    vecchia_train_steps: int = 0
    vecchia_train_epochs: int = 1
    vecchia_graph_refresh_epochs: int = 0
    vecchia_train_batch_size: int = 32
    vecchia_prediction_batch_size: int = 32
    vecchia_lr: float = 0.01
    vecchia_weight_decay: float = 0.0
    vecchia_learn_lengthscale: bool = True
    vecchia_learn_outputscale: bool = True
    vecchia_learn_sigma_f: bool = True
    vecchia_min_sigma_f: float = 1e-6
    vecchia_log_every: int = 0

    baseline_joint_scale: bool = True
    baseline_num_workers: int = 0
    dsoftki_train_epochs: int = 50
    dsoftki_num_inducing: int = 512
    dsoftki_batch_size: int | None = None
    dsoftki_prediction_batch_size: int = 512
    dsoftki_fit_chunk_size: int = 256
    dsoftki_lr: float | None = None
    dsoftki_dtype: str = "float32"
    dsoftki_use_ard: bool | None = None
    dsoftki_noise: float | None = None
    dsoftki_deriv_noise: float | None = None
    dsoftki_learn_noise: bool = True
    dsoftki_cg_tolerance: float = 1e-5
    ddsvgp_train_epochs: int = 50
    ddsvgp_num_inducing: int = 512
    ddsvgp_num_directions: int = 2
    ddsvgp_batch_size: int | None = None
    ddsvgp_prediction_batch_size: int = 512
    ddsvgp_lr: float | None = None
    ddsvgp_dtype: str = "float32"
    ddsvgp_noise: float | None = None
    ddsvgp_mll_type: Literal["PLL", "ELBO"] = "PLL"


def resolve_dataset_config(cfg: MD22Config, d: int) -> MD22Config:
    """Resolve unspecified baseline options using the TERA MD22 schedule."""
    cfg = replace(cfg)
    if d < 1:
        raise ValueError("Input dimension must be positive.")
    if d >= 1000:
        batch, dsoftki_lr, ddsvgp_lr = 128, 0.001, 0.0015
    elif d >= 300:
        batch, dsoftki_lr, ddsvgp_lr = 256, 0.002, 0.003
    elif d >= 180:
        batch, dsoftki_lr, ddsvgp_lr = 512, 0.004, 0.006
    else:
        batch, dsoftki_lr, ddsvgp_lr = 1024, 0.008, 0.012
    noise = cfg.sigma_f if cfg.dsoftki_noise is None else cfg.dsoftki_noise
    resolved = replace(
        cfg,
        dsoftki_batch_size=batch if cfg.dsoftki_batch_size is None else cfg.dsoftki_batch_size,
        ddsvgp_batch_size=batch if cfg.ddsvgp_batch_size is None else cfg.ddsvgp_batch_size,
        dsoftki_lr=dsoftki_lr if cfg.dsoftki_lr is None else cfg.dsoftki_lr,
        ddsvgp_lr=ddsvgp_lr if cfg.ddsvgp_lr is None else cfg.ddsvgp_lr,
        dsoftki_use_ard=cfg.use_ard if cfg.dsoftki_use_ard is None else cfg.dsoftki_use_ard,
        dsoftki_noise=noise,
        dsoftki_deriv_noise=noise * d
        if cfg.dsoftki_deriv_noise is None
        else cfg.dsoftki_deriv_noise,
        ddsvgp_noise=cfg.sigma_f if cfg.ddsvgp_noise is None else cfg.ddsvgp_noise,
    )
    for prefix, method in (("dsoftki", "dsoftki"), ("ddsvgp", "ddsvgp")):
        if method not in cfg.methods:
            continue
        for suffix in ("train_epochs", "num_inducing", "batch_size", "prediction_batch_size", "lr"):
            value = getattr(resolved, f"{prefix}_{suffix}")
            if value <= 0:
                raise ValueError(f"{prefix}_{suffix} must be positive, got {value}.")
    if cfg.ddsvgp_mll_type not in {"PLL", "ELBO"}:
        raise ValueError("ddsvgp_mll_type must be PLL or ELBO.")
    return resolved


def resolve_method_config(cfg: MD22Config, method: str) -> MD22Config:
    m = getattr(cfg, f"{method_prefix(method)}_m", None)
    return cfg if m is None or m == cfg.m else replace(cfg, m=m)


def load_config(path: str | Path) -> MD22Config:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = normalize_config(yaml.safe_load(f) or {})
    return MD22Config(**expand_yaml(raw, "md22", MD22Config))
