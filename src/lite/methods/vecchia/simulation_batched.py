from __future__ import annotations

import torch

from lite.methods.common.data import PredictiveMarginals
from lite.methods.common.linalg import cholesky_psd
from lite.methods.common.utils import scale_inputs
from lite.methods.lite.posterior import _radial, nearest_neighbors


class BatchedVecchiaSimulationPredictor:
    """Function-only local prediction with the same neighborhoods as LITE/TERA."""

    name = "Vecchia GP"

    def __init__(self, m, prediction_batch_size=256, *, jitter=1e-8):
        if m < 0 or prediction_batch_size < 1 or jitter < 0:
            raise ValueError("Invalid neighborhood, batch size, or jitter")
        self.m = m
        self.prediction_batch_size = prediction_batch_size
        self.jitter = jitter
        self.data = None

    def build(self, data):
        self.data = data

    def _predict_batch(self, targets, ids, return_expected_mse):
        data = self.data
        batch, m = ids.shape
        prior = targets.new_tensor(data.outputscale)
        if m == 0:
            return targets.new_zeros(batch), prior.expand(batch), prior.expand(batch)
        delta = (data.X_train[ids] - targets[:, None]) / data.lengthscale
        gram = delta @ delta.transpose(-1, -2)
        diagonal = gram.diagonal(dim1=-2, dim2=-1)
        distance = (diagonal[:, :, None] + diagonal[:, None, :] - 2 * gram).clamp_min(0)
        covariance, _, _ = _radial(distance, prior, data.kernel_name)
        cross, _, _ = _radial(diagonal, prior, data.kernel_name)
        covariance = covariance + data.sigma_f**2 * torch.eye(
            m, device=targets.device, dtype=targets.dtype
        )
        factor = cholesky_psd(covariance, jitter=self.jitter)
        solution = torch.cholesky_solve(torch.stack((data.f_train_obs[ids], cross), -1), factor)
        weights = solution[..., 1]
        mean = (cross * solution[..., 0]).sum(-1)
        cross_term = (cross * weights).sum(-1)
        variance = (prior - cross_term).clamp_min(torch.finfo(targets.dtype).eps)
        risk = None
        if return_expected_mse:
            quadratic = (weights * (covariance @ weights.unsqueeze(-1)).squeeze(-1)).sum(-1)
            risk = prior - 2 * cross_term + quadratic
        return mean, variance, risk

    @torch.no_grad()
    def predict_f_marginals(self, targets, *, neighborhoods=None, return_expected_mse=False):
        if self.data is None:
            raise RuntimeError("build() must precede prediction")
        if neighborhoods is None:
            ids = nearest_neighbors(
                self.data.X_train_scaled, scale_inputs(targets, self.data.lengthscale), self.m
            )
            neighborhoods = list(ids.unbind(0))
        if len(neighborhoods) != len(targets):
            raise ValueError("One conditioning set is required for each target")
        means, variances, risks = [], [], []
        for start in range(0, len(targets), self.prediction_batch_size):
            batch_targets = targets[start : start + self.prediction_batch_size]
            sets = neighborhoods[start : start + self.prediction_batch_size]
            counts = [len(ids) for ids in sets]
            mean, variance = batch_targets.new_empty(len(sets)), batch_targets.new_empty(len(sets))
            risk = batch_targets.new_empty(len(sets)) if return_expected_mse else None
            for count in sorted(set(counts)):
                positions = [i for i, n in enumerate(counts) if n == count]
                ids = torch.stack([sets[i] for i in positions]).to(targets.device)
                result = self._predict_batch(batch_targets[positions], ids, return_expected_mse)
                mean[positions], variance[positions] = result[:2]
                if return_expected_mse:
                    risk[positions] = result[2]
            means.append(mean)
            variances.append(variance)
            if return_expected_mse:
                risks.append(risk)
        return PredictiveMarginals(
            mean=torch.cat(means) if means else targets.new_empty(0),
            var=torch.cat(variances) if variances else targets.new_empty(0),
            expected_mse=(torch.cat(risks) if risks else targets.new_empty(0))
            if return_expected_mse
            else None,
        )
