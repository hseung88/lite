from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from lite.methods.common.data import PredictiveMarginals, SimulatedDataset
from lite.methods.common.ordering import knn_to_eval
from lite.methods.common.utils import scale_inputs


class MarginalPredictor(ABC):
    @abstractmethod
    def build(self, data: SimulatedDataset) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict_f_marginals(self, X_eval) -> PredictiveMarginals:
        raise NotImplementedError


class NeighborhoodPredictor(MarginalPredictor):
    def predict_f_marginals(self, X_eval: torch.Tensor) -> PredictiveMarginals:
        if self.data is None:
            raise RuntimeError("build() must be called before prediction.")
        data = self.data
        X_eval_scaled = scale_inputs(X_eval, data.lengthscale)
        neighborhoods = knn_to_eval(data.X_train_scaled, X_eval_scaled, self.m)
        means: list[torch.Tensor] = []
        vars_: list[torch.Tensor] = []
        with torch.no_grad():
            for j, idx in enumerate(neighborhoods):
                mean_j, var_j = self._predict_one(
                    x_eval=X_eval[j : j + 1],
                    x_eval_scaled=X_eval_scaled[j : j + 1],
                    idx=idx,
                )
                means.append(mean_j)
                vars_.append(var_j)
        return PredictiveMarginals(
            mean=torch.stack(means).contiguous(), var=torch.stack(vars_).contiguous()
        )
