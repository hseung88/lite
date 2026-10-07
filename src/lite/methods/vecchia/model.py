from __future__ import annotations

import math

import torch

from lite.methods.common.data import PredictiveMarginals
from lite.methods.common.linalg import cholesky_psd
from lite.methods.common.training import RegressionData
from lite.methods.lite.posterior import _radial, nearest_neighbors
from lite.methods.tera.model import TERAModel


def conditional_moments(
    targets, neighbors, values, *, lengthscale, outputscale, noise_y_var, kernel, jitter=1e-8
):
    """Function-only conditional using the same geometry and jitter as LITE."""
    ell = torch.as_tensor(lengthscale, dtype=targets.dtype, device=targets.device)
    os = torch.as_tensor(outputscale, dtype=targets.dtype, device=targets.device)
    batch, m = neighbors.shape[:2]
    if m == 0:
        return targets.new_zeros(batch), os.expand(batch)
    delta = (neighbors - targets[:, None]) / ell
    gram = delta @ delta.transpose(-1, -2)
    diagonal = gram.diagonal(dim1=-2, dim2=-1)
    distances = (diagonal[:, :, None] + diagonal[:, None, :] - 2 * gram).clamp_min(0)
    covariance, _, _ = _radial(distances, os, kernel)
    cross, _, _ = _radial(diagonal, os, kernel)
    covariance = covariance + noise_y_var * torch.eye(m, dtype=targets.dtype, device=targets.device)
    factor = cholesky_psd(covariance, jitter=jitter)
    solution = torch.cholesky_solve(torch.stack((values, cross), dim=-1), factor)
    mean = (cross * solution[..., 0]).sum(-1)
    variance = (os - (cross * solution[..., 1]).sum(-1)).clamp_min(jitter)
    return mean, variance


def batch_nll(
    *,
    state,
    target_positions,
    lengthscale,
    outputscale,
    sigma_f,
    sigma_g=None,
    kernel,
    gradient_noise_model=None,
    **kwargs,
):
    positions = target_positions.to(device="cpu", dtype=torch.long)
    counts = positions.clamp_max(state.neighbors.shape[1])
    total = 0 * (lengthscale.sum() + outputscale + sigma_f)
    for m in counts.unique().tolist():
        pos = positions[counts == m].to(state.X.device)
        ids = state.neighbors[pos, :m]
        mean, variance = conditional_moments(
            state.X[pos],
            state.X[ids],
            state.y[ids],
            lengthscale=lengthscale,
            outputscale=outputscale,
            noise_y_var=sigma_f,
            kernel=kernel,
        )
        variance = variance + sigma_f
        total = (
            total
            + 0.5
            * (
                variance.log() + (state.y[pos] - mean).square() / variance + math.log(2 * math.pi)
            ).sum()
        )
    return total / positions.numel()


class VecchiaGPPredictor:
    def __init__(self, m, prediction_batch_size):
        self.m = m
        self.prediction_batch_size = prediction_batch_size
        self.data = None

    def build(self, data):
        self.data = data

    @torch.no_grad()
    def predict_f_marginals(self, targets):
        if self.data is None:
            raise RuntimeError("fit() must be called before prediction")
        data = self.data
        means, variances = [], []
        for start in range(0, len(targets), self.prediction_batch_size):
            batch = targets[start : start + self.prediction_batch_size]
            ids = nearest_neighbors(data.X_train_scaled, batch / data.lengthscale, self.m)
            mean, variance = conditional_moments(
                batch,
                data.X_train[ids],
                data.f_train_obs[ids],
                lengthscale=data.lengthscale,
                outputscale=data.outputscale,
                noise_y_var=data.sigma_f**2,
                kernel=data.kernel_name,
            )
            means.append(mean)
            variances.append(variance)
        if not means:
            return PredictiveMarginals(mean=targets.new_empty(0), var=targets.new_empty(0))
        return PredictiveMarginals(mean=torch.cat(means), var=torch.cat(variances))


class VecchiaGPModel(TERAModel):
    """Same ordering, neighborhoods and optimizer as LITE, without gradient blocks."""

    name = "Vecchia GP"

    def __init__(
        self,
        *,
        sigma_g=0.0,
        learn_sigma_g=False,
        min_sigma_g=0.0,
        gradient_noise_model="iid",
        **kwargs,
    ):
        super().__init__(
            sigma_g=0.0, learn_sigma_g=False, min_sigma_g=0.0, gradient_noise_model="iid", **kwargs
        )

    def _make_predictor(self):
        return VecchiaGPPredictor(self.m, self.prediction_batch_size)

    def _batch_nll(self, **kwargs):
        return batch_nll(**kwargs)

    def fit(self, split):
        function_data = RegressionData(
            X_train=split.X_train,
            y_train=split.y_train,
            g_train=split.X_train.new_empty((len(split.X_train), 0)),
            X_test=split.X_test,
        )
        super().fit(function_data)
