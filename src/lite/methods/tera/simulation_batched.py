from __future__ import annotations

from dataclasses import replace

from lite.methods.common.data import SimulatedDataset
from lite.methods.tera.batched_prediction import BatchedTERAPredictor


class BatchedTERASimulationPredictor(BatchedTERAPredictor):
    def __init__(self, m: int, *, prediction_batch_size: int = 256) -> None:
        super().__init__(
            m=m, gradient_noise_model="iid", prediction_batch_size=prediction_batch_size
        )

    def build(self, data: SimulatedDataset) -> None:
        super().build(replace(data, sigma_g=data.sigma_g**2))
