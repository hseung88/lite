from __future__ import annotations

import math

import torch

from lite.methods.common.data import PredictiveMarginals
from lite.methods.common.profiling import step_scope
from lite.methods.lite.posterior import conditional_moments, predict_marginals
from lite.methods.tera.model import TERAModel


class LITEPredictor:
    def __init__(
        self, m, gradient_noise_model, prediction_batch_size=256, normalize_directions=False
    ):
        self.m = m
        self.gradient_noise_model = gradient_noise_model
        self.prediction_batch_size = prediction_batch_size
        self.normalize_directions = normalize_directions
        self.data = None

    def build(self, data):
        self.data = data

    def predict_f_marginals(self, X_eval):
        if self.data is None:
            raise RuntimeError("fit() must be called before prediction")
        data = self.data
        with torch.no_grad():
            mean, var = predict_marginals(
                X_eval,
                data.X_train,
                data.f_train_obs,
                data.g_train_obs,
                lengthscale=data.lengthscale,
                outputscale=data.outputscale,
                noise_y_var=data.sigma_f**2,
                noise_g_var=data.sigma_g,
                kernel=data.kernel_name,
                gradient_noise_model=self.gradient_noise_model,
                m=self.m,
                prediction_batch_size=self.prediction_batch_size,
                normalize_directions=self.normalize_directions,
                train_scaled=data.X_train_scaled,
            )
        return PredictiveMarginals(mean=mean, var=var)


def batch_nll(
    *,
    state,
    target_positions,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g,
    kernel,
    gradient_noise_model,
    normalize_directions=False,
    profile_regions=False,
):

    positions = target_positions.to(device="cpu", dtype=torch.long)
    counts = positions.clamp_max(state.neighbors.shape[1])
    total = 0 * (lengthscale.sum() + outputscale + sigma_f + sigma_g)
    for m in counts.unique().tolist():
        pos = positions[counts == m].to(state.X.device)
        with step_scope("gather_gram", profile_regions):
            ids = state.neighbors[pos, :m]
            inputs = (state.X[pos], state.X[ids], state.y[ids], state.g[ids])
        mean, var = conditional_moments(
            *inputs,
            lengthscale=lengthscale,
            outputscale=outputscale,
            noise_y_var=sigma_f,
            noise_g_var=sigma_g,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            freeze_directions=True,
            profile_regions=profile_regions,
            normalize_directions=normalize_directions,
        )
        del inputs
        var = var + sigma_f
        total = (
            total
            + 0.5 * (var.log() + (state.y[pos] - mean).square() / var + math.log(2 * math.pi)).sum()
        )
    return total / target_positions.numel()


class LITEModel(TERAModel):
    name = "LITE"

    def __init__(
        self,
        *,
        train_batch_size=None,
        prediction_batch_size=256,
        lengthscale_min=None,
        lengthscale_max=None,
        normalize_directions=False,
        **kwargs,
    ):
        if prediction_batch_size <= 0:
            raise ValueError("prediction_batch_size must be positive")
        if any(v is not None and v <= 0 for v in (lengthscale_min, lengthscale_max)):
            raise ValueError("lengthscale bounds must be positive")
        if (
            lengthscale_min is not None
            and lengthscale_max is not None
            and lengthscale_min > lengthscale_max
        ):
            raise ValueError("lengthscale_min must not exceed lengthscale_max")
        self.prediction_batch_size = prediction_batch_size
        self.normalize_directions = normalize_directions
        self.lengthscale_min = lengthscale_min
        self.lengthscale_max = lengthscale_max
        super().__init__(
            train_batch_size=train_batch_size,
            prediction_batch_size=prediction_batch_size,
            lengthscale_min=lengthscale_min,
            lengthscale_max=lengthscale_max,
            **kwargs,
        )
        if self.m < 0 or self.train_batch_size <= 0:
            raise ValueError("m must be nonnegative and train_batch_size must be positive")

    def _make_predictor(self):
        return LITEPredictor(
            self.m, self.gradient_noise_model, self.prediction_batch_size, self.normalize_directions
        )

    def _batch_nll(self, **kwargs):
        return batch_nll(**kwargs, normalize_directions=self.normalize_directions)

    def _likelihood_lengthscale(self, lengthscale):

        return lengthscale

    def _apply_lengthscale_bounds(self, lengthscale):
        if self.lengthscale_min is not None:
            lengthscale = lengthscale.clamp_min(self.lengthscale_min)
        if self.lengthscale_max is not None:
            lengthscale = lengthscale.clamp_max(self.lengthscale_max)
        return lengthscale

    @torch.no_grad()
    def _project_log_lengthscale_(self, log_lengthscale):
        if self.lengthscale_min is not None:
            log_lengthscale.clamp_(min=math.log(self.lengthscale_min))
        if self.lengthscale_max is not None:
            log_lengthscale.clamp_(max=math.log(self.lengthscale_max))
