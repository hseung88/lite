import math
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class SymmetryConfig:
    n_train: int = 1024
    m: int = 20
    n_targets: int = 200
    d_values: list[int] = field(default_factory=lambda: [50, 100, 200, 500, 1000])
    repeats: int = 5
    seed: int = 42
    kernel: str = "matern52"
    outputscale: float = 1.0
    sigma_f: float = 0.1
    sigma_g: float = 0.1
    target_median_correlation: float = 0.3
    rank_rtol: float = 1e-12
    device: str = "cuda"
    dtype: str = "float64"

    def validate(self):
        if self.m < 2 or self.n_train <= self.m:
            raise ValueError("Require 2 <= m < n_train; m is never silently truncated.")
        if not 1 <= self.n_targets <= self.n_train - self.m or self.repeats < 1:
            raise ValueError("Require positive repeats and 1 <= n_targets <= n_train - m.")
        if not self.d_values or len(set(self.d_values)) != len(self.d_values):
            raise ValueError("Dimensions must be nonempty and unique.")
        if any(d < self.m for d in self.d_values):
            raise ValueError("Require d >= m for the compound-symmetry experiment.")
        if self.kernel not in {"rbf", "matern52"} or self.dtype != "float64":
            raise ValueError("Use rbf or matern52 and float64.")
        if not math.isfinite(self.outputscale) or self.outputscale <= 0:
            raise ValueError("outputscale must be a positive finite variance.")
        if any(not math.isfinite(s) or s <= 0 for s in (self.sigma_f, self.sigma_g)):
            raise ValueError(
                "sigma_f and sigma_g must be positive finite noise standard deviations."
            )
        if not 0 < self.target_median_correlation < 1 or not 0 < self.rank_rtol < 1:
            raise ValueError("Correlation and rank_rtol must lie strictly between zero and one.")
        if self.seed < 0:
            raise ValueError("seed must be nonnegative.")


def load_config(path: str | Path) -> SymmetryConfig:
    with open(path, encoding="utf-8") as f:
        cfg = SymmetryConfig(**(yaml.safe_load(f) or {}))
    cfg.validate()
    return cfg
