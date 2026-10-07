from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class Prediction:
    y_mean: torch.Tensor
    y_var: torch.Tensor | None = None
