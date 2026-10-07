from dataclasses import dataclass

import torch


@dataclass
class CovarianceSolver:
    keep: torch.Tensor
    scale: torch.Tensor
    chol: torch.Tensor | None
    eigvec: torch.Tensor | None
    eigval: torch.Tensor | None
    rank: int

    def solve(self, rhs):
        vector = rhs.ndim == 1
        b = rhs[:, None] if vector else rhs
        scaled = b[self.keep] / self.scale[:, None]
        if self.chol is not None:
            result = torch.cholesky_solve(scaled, self.chol)
        else:
            result = self.eigvec @ ((self.eigvec.T @ scaled) / self.eigval[:, None])
        out = torch.zeros_like(b)
        out[self.keep] = result / self.scale[:, None]
        return out[:, 0] if vector else out


def factor_covariance(K, *, rtol=1e-12):

    K = (K + K.T) / 2
    diag = K.diagonal()
    if not torch.isfinite(K).all() or (diag < 0).any():
        raise ValueError("Covariance must be finite with nonnegative diagonal.")
    keep = torch.where(diag > 0)[0]
    scale = diag[keep].sqrt()
    R = K[keep[:, None], keep] / scale[:, None] / scale[None, :]
    L, info = torch.linalg.cholesky_ex(R)

    if info.item() == 0 and (L.diagonal().square() > rtol).all():
        return CovarianceSolver(keep, scale, L, None, None, len(keep))
    vals, vecs = torch.linalg.eigh(R)
    cutoff = rtol * vals.abs().max().clamp_min(1)
    if (vals < -10 * cutoff).any():
        raise RuntimeError("Covariance has materially negative eigenvalues.")
    mask = vals > cutoff
    return CovarianceSolver(keep, scale, None, vecs[:, mask], vals[mask], int(mask.sum()))


def condition_scalar(K, cross, observations, prior_var, *, rtol=1e-12):
    solver = factor_covariance(K, rtol=rtol)
    weights = solver.solve(cross)
    mean = weights @ observations
    variance = prior_var - cross @ weights
    tol = 100 * rtol * max(1.0, float(prior_var))
    if float(variance) < -tol:
        raise RuntimeError("Substantially negative predictive variance.")
    return mean, variance.clamp_min(torch.finfo(K.dtype).eps), solver.rank
