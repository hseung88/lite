from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

from lite.experiments.config import normalize_config
from lite.experiments.options import expand_yaml

BOMethod = Literal[
    "sobol",
    "vbo",
    "turbo",
    "turbo-logei",
    "tera",
    "tera_batched",
    "tera-target",
    "tera-target-pred",
    "lite",
]
KernelName = Literal["matern52", "rbf"]
VBOAcquisitionName = Literal["log_nei", "log_ei"]
VBOFitBackend = Literal["botorch_mll", "adam"]


@dataclass(slots=True)
class BOConfig:
    def __post_init__(self):
        self.methods = _validate_methods(self.methods)

    experiment_name: str = "bo_synthetic"
    benchmarks: list[str] = field(default_factory=lambda: ["levy_200", "ackley_200"])
    methods: list[BOMethod] = field(
        default_factory=lambda: ["vbo", "turbo-logei", "tera_batched", "lite"]
    )
    seeds: list[int] = field(default_factory=lambda: [1, 27, 42, 86, 99])
    budget: int = 150
    n_init: int = 30
    batch_size: int = 1
    device: str = "cuda"
    dtype: str = "float64"
    objective_noise_std: float = 0.0
    standardize_y: bool = True
    maximize: bool = True
    continue_on_error: bool = False

    kernel: KernelName = "matern52"
    use_ard: bool = True
    lengthscale_init: str = "d_scaled"
    base_lengthscale: float = 0.5
    outputscale_init: float = 1.0
    noise_var_init: float = 1.0e-6
    min_noise_var: float = 1.0e-8
    gp_train_steps: int = 50
    gp_lr: float = 0.01
    gp_weight_decay: float = 0.0

    vbo_use_priors: bool = True
    vbo_lengthscale_prior_loc: float = 1.4
    vbo_lengthscale_prior_scale: float = 1.73205
    vbo_noise_prior_loc: float = -4.0
    vbo_noise_prior_scale: float = 1.0

    vbo_kernel: KernelName = "rbf"
    vbo_use_outputscale: bool = False

    vbo_lengthscale_init: str = "gpytorch_default"
    vbo_base_lengthscale: float = 0.5
    vbo_fix_outputscale: bool = True
    vbo_lengthscale_constraint: float = 1.0e-4
    vbo_noise_constraint: float = 1.0e-4

    vbo_initialize_noise: bool = False
    vbo_noise_var_init: float = 1.0e-4

    vbo_fit_backend: VBOFitBackend = "botorch_mll"
    vbo_fit_maxiter: int | None = None
    vbo_acq_raw_samples: int = 512
    vbo_acq_restarts: int = 4
    vbo_acq_maxiter: int = 300
    vbo_acq_batch_limit: int = 64
    vbo_acq_sample_around_best: bool = True
    vbo_acq_sample_around_best_sigma: float = 0.1
    vbo_acq_retry_on_optimization_warning: bool = False

    vbo_acq: VBOAcquisitionName = "log_nei"
    vbo_nei_mc_samples: int = 512
    vbo_nei_prune_baseline: bool = True
    vbo_nei_cache_root: bool = True

    acq: str = "log_ei"
    acq_raw_samples: int = 512
    acq_restarts: int = 4
    acq_maxiter: int = 300

    tera_m: int = 20
    tera_train_steps: int = 20

    tera_refit_every: int = 20
    tera_initial_train_steps: int = 50
    tera_update_train_steps: int = 10
    tera_prediction_batch_size: int = 256
    tera_train_batch_size: int = 256
    tera_lr: float = 0.01
    tera_learn_outputscale: bool = True
    tera_learn_lengthscale: bool = True
    tera_lengthscale_init: str = "base"
    tera_base_lengthscale: float = 0.5
    tera_lengthscale_min: float = 0.003
    tera_lengthscale_max: float = 8.0
    tera_learn_sigma_f: bool = False
    tera_learn_sigma_g: bool = False
    tera_sigma_f: float = 1.0e-3
    tera_sigma_g: float = 1.0e-3
    tera_gradient_noise_model: Literal["iid", "scaled"] = "scaled"
    tera_acq_raw_samples: int = 256
    tera_acq_restarts: int = 4
    tera_acq_maxiter: int = 30
    tera_target_batch_size: int = 512

    lite_m: int = 20
    lite_train_steps: int = 20

    lite_refit_every: int = 20
    lite_initial_train_steps: int = 50
    lite_update_train_steps: int = 10
    lite_train_batch_size: int = 256
    lite_lr: float = 0.01
    lite_learn_outputscale: bool = True
    lite_learn_lengthscale: bool = True
    lite_lengthscale_init: str = "base"
    lite_base_lengthscale: float = 0.5
    lite_lengthscale_min: float = 0.003
    lite_lengthscale_max: float = 8.0
    lite_learn_sigma_f: bool = False
    lite_learn_sigma_g: bool = False
    lite_sigma_f: float = 1.0e-3
    lite_sigma_g: float = 1.0e-3
    lite_gradient_noise_model: Literal["iid", "scaled"] = "scaled"
    lite_acq_raw_samples: int = 256
    lite_acq_restarts: int = 4
    lite_acq_maxiter: int = 30
    lite_prediction_batch_size: int = 512

    turbo_length_init: float = 0.8
    turbo_length_min: float = 0.5**7
    turbo_length_max: float = 1.6
    turbo_success_tolerance: int = 3
    turbo_failure_tolerance: int = 0
    turbo_max_local_points: int = 2048
    turbo_n_candidates: int = 0
    turbo_gp_train_steps: int = 50
    turbo_gp_lr: float = 0.1
    turbo_min_cuda: int = 1024
    turbo_max_cholesky_size: int = 2000
    turbo_noise_min: float = 5.0e-4
    turbo_noise_max: float = 0.2
    turbo_lengthscale_min: float = 0.005
    turbo_lengthscale_max: float = 2.0
    turbo_outputscale_min: float = 0.05
    turbo_outputscale_max: float = 20.0

    out_csv_name: str = "results.csv"

    lassodna_data_path: str | None = None
    lassodna_data_home: str | None = None
    lassodna_split_seed: int = 42
    lassodna_n_splits: int = 5
    lassodna_test_size: float = 0.15
    lassodna_tol: float = 1e-8
    lassodna_max_iter: int = 1000
    lassodna_backend: Literal["celer", "torch"] = "celer"
    lassodna_torch_max_iter: int = 20000


def _validate_methods(methods: list[str]) -> list[BOMethod]:
    methods = ["lite" if m.strip().lower() == "lite_normalized" else m.strip() for m in methods]
    allowed = {
        "sobol",
        "vbo",
        "turbo",
        "turbo-logei",
        "tera",
        "tera_batched",
        "tera-target",
        "tera-target-pred",
        "lite",
    }
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"Unknown BO method(s): {bad}. Allowed: {sorted(allowed)}")
    out: list[BOMethod] = []
    for method in methods:
        name = method
        if name not in out:
            out.append(name)
    return out


def load_config(path: str | Path) -> BOConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw: dict[str, Any] = normalize_config(yaml.safe_load(f) or {})
    if "methods" in raw:
        raw["methods"] = _validate_methods(raw["methods"])

    return BOConfig(**expand_yaml(raw, "bo", BOConfig))
