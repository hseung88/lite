from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from lite.experiments.config import normalize_config
from lite.methods.names import normalize_method_name

KernelName = Literal["rbf", "matern52"]
MethodName = Literal[
    "Standard GP",
    "Standard dGP",
    "Standard dGP-deRoos",
    "Vecchia GP",
    "Vecchia dGP",
    "Vecchia dGP-deRoos",
    "TERA",
    "TERA (batched)",
    "LITE",
]

METHOD_STANDARD_GP: MethodName = "Standard GP"
METHOD_STANDARD_DGP: MethodName = "Standard dGP"
METHOD_STANDARD_DGP_DEROOS: MethodName = "Standard dGP-deRoos"
METHOD_VECCHIA_GP: MethodName = "Vecchia GP"
METHOD_VECCHIA_DGP: MethodName = "Vecchia dGP"
METHOD_VECCHIA_DGP_DEROOS: MethodName = "Vecchia dGP-deRoos"
METHOD_TERA: MethodName = "TERA"
METHOD_TERA_BATCHED: MethodName = "TERA (batched)"
METHOD_LITE: MethodName = "LITE"

METHOD_ORDER: tuple[MethodName, ...] = (
    METHOD_STANDARD_GP,
    METHOD_STANDARD_DGP,
    METHOD_STANDARD_DGP_DEROOS,
    METHOD_VECCHIA_GP,
    METHOD_VECCHIA_DGP,
    METHOD_VECCHIA_DGP_DEROOS,
    METHOD_TERA,
    METHOD_TERA_BATCHED,
    METHOD_LITE,
)
ALL_METHOD_NAMES = set(METHOD_ORDER)


def validate_methods(methods: list[str]) -> list[MethodName]:
    aliases = {
        "lite": METHOD_LITE,
        "tera": METHOD_TERA,
        "tera_batched": METHOD_TERA_BATCHED,
        "tera (batched)": METHOD_TERA_BATCHED,
        "vecchia": METHOD_VECCHIA_GP,
        "vecchia_gp": METHOD_VECCHIA_GP,
        "vecchia gp": METHOD_VECCHIA_GP,
    }
    methods = [aliases.get(m.strip().lower(), m.strip()) for m in methods]
    methods = [normalize_method_name(m) for m in methods]
    bad = [m for m in methods if m not in ALL_METHOD_NAMES]
    if bad:
        allowed = ", ".join(METHOD_ORDER)
        raise ValueError(f"Unknown method name(s): {bad}. Allowed methods: {allowed}")
    normalized: list[MethodName] = []
    for method in methods:
        name = method
        if name not in normalized:
            normalized.append(name)
    return normalized


@dataclass(slots=True)
class ExperimentConfig:
    experiment_name: str
    kernel: KernelName
    use_ard: bool
    lengthscale: float | list[float] | None
    outputscale: float
    sigma_f: float
    sigma_g: float
    n_train: int
    n_eval: int
    m: int | str
    d_values: list[int]
    repeats: int
    seed: int
    device: str
    dtype: str
    methods: list[MethodName]
    lite_prediction_batch_size: int = 256
    tera_prediction_batch_size: int = 256
    vecchia_prediction_batch_size: int | None = None
    sample_eval: bool = False
    sampling_device: str | None = None
    sampling_chunk_size: int = 8
    n_train_values: list[int] | None = None
    m_values: list[int] | None = None
    target_median_correlation: float | None = None
    design: Literal["sobol", "random"] = "sobol"
    sampling: Literal["dense", "deroos", "zeros"] = "dense"
    dense_sampling_max_obs_dim: int = 20000
    deroos_sampling_max_n2: int = 6400
    standard_dgp_dense_max_d: int = 100
    standard_dgp_dense_max_obs_dim: int = 25000
    standard_dgp_deroos_max_n: int = 100
    vecchia_dense_max_local_dim: int = 5000
    evaluation_mode: Literal["standard", "expected_mse"] = "standard"
    knn_query_batch_size: int = 32
    knn_train_chunk_size: int = 4096
    lengthscale_probe_size: int = 2048
    measurement_repeats: int = 3
    warmup_batches: int = 2

    def __post_init__(self) -> None:
        if self.evaluation_mode not in {"standard", "expected_mse"}:
            raise ValueError("evaluation_mode must be standard or expected_mse")
        if (
            min(self.knn_query_batch_size, self.knn_train_chunk_size, self.measurement_repeats) < 1
            or self.lengthscale_probe_size < 2
            or self.warmup_batches < 0
        ):
            raise ValueError("Invalid nearest-neighbor, calibration, or timing setting")
        if self.evaluation_mode == "expected_mse":
            if self.sample_eval:
                raise ValueError("expected_mse does not sample test values; set sample_eval: false")
            if self.sampling != "zeros":
                raise ValueError("expected_mse requires sampling: zeros (timing placeholders only)")
            if any(
                method not in {METHOD_VECCHIA_GP, METHOD_LITE, METHOD_TERA_BATCHED}
                for method in self.methods
            ):
                raise ValueError("expected_mse supports vecchia, lite and tera_batched only")
            minimum_n = min(self.n_train_values or [self.n_train])
            if any(
                not isinstance(m, int) or not 1 <= m <= minimum_n for m in self.m_values or [self.m]
            ):
                raise ValueError("expected_mse requires integer m values between 1 and n_train")
            if any(d < 1 for d in self.d_values):
                raise ValueError("Input dimensions must be positive")
        if self.sample_eval and self.sampling != "dense":
            raise ValueError("sample_eval requires dense joint GP sampling, not zeros/deRoos.")
        if (
            min(
                self.n_train,
                self.n_eval,
                self.repeats,
                self.lite_prediction_batch_size,
                self.tera_prediction_batch_size,
                self.sampling_chunk_size,
            )
            < 1
        ):
            raise ValueError(
                "Sample counts, repeats, batches, and sampling chunk size must be positive."
            )
        if (
            self.vecchia_prediction_batch_size is not None
            and self.vecchia_prediction_batch_size < 1
        ):
            raise ValueError(
                "Vecchia prediction batch size must be positive or null (inherits LITE)."
            )
        if self.sigma_f < 0 or self.sigma_g < 0 or self.outputscale <= 0:
            raise ValueError(
                "Noise standard deviations must be nonnegative and outputscale positive."
            )
        if self.sampling_device not in {None, "cpu", "cuda"}:
            raise ValueError("sampling_device must be cpu, cuda, or null.")
        if self.sample_eval:
            minimum_n = min(self.n_train_values or [self.n_train])
            for m in self.m_values or [self.m]:
                if isinstance(m, int) and not 1 <= m <= minimum_n:
                    raise ValueError("RMSE scaling requires 1 <= m <= each training sample count.")


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = normalize_config(yaml.safe_load(f) or {})
    if "methods" in raw:
        raw["methods"] = validate_methods(raw["methods"])
    return ExperimentConfig(**raw)
