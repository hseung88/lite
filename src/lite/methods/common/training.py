from dataclasses import dataclass

import torch


@dataclass
class RegressionData:
    X_train: torch.Tensor
    y_train: torch.Tensor
    g_train: torch.Tensor | None = None
    X_test: torch.Tensor | None = None

    def __post_init__(self):
        if self.X_test is None:
            self.X_test = self.X_train[:0]
