import torch

from lite.methods.common.covariance import condition_scalar as condition_scalar
from lite.methods.common.covariance import factor_covariance
from lite.methods.lite.posterior import _radial, coefficient_rows


def raw_lite_directions(
    target, neighbors, *, lengthscale, outputscale, kernel="matern52", sigma_f=0.0
):

    delta = neighbors - target
    scaled = delta / lengthscale
    H = scaled @ scaled.T
    h = H.diagonal()
    K, alpha, _ = _radial((h[:, None] + h[None, :] - 2 * H).clamp_min(0), outputscale, kernel)
    k, ai, _ = _radial(h, outputscale, kernel)
    if sigma_f:
        K = K + sigma_f**2 * torch.eye(len(neighbors), device=K.device, dtype=K.dtype)
    w = factor_covariance(K).solve(k)
    C, _ = coefficient_rows(H, alpha, w, ai)
    directions = (C @ delta).reshape(len(neighbors), 2, -1)
    return directions[:, 0], directions[:, 1]


def directional_blocks(
    target,
    neighbors,
    directions,
    sites,
    *,
    lengthscale,
    outputscale,
    kernel="matern52",
    sigma_f=0.0,
    sigma_g=0.0,
):

    ell = torch.as_tensor(lengthscale, device=neighbors.device, dtype=neighbors.dtype)
    delta = (neighbors[:, None] - neighbors[None, :]) / ell
    Kff, alpha, beta = _radial(delta.square().sum(-1), outputscale, kernel)
    U = directions / ell

    Kfs = alpha[:, sites] * torch.einsum("aqd,qd->aq", delta[:, sites], U)
    dd = delta[sites[:, None], sites[None, :]]
    p_left = torch.einsum("aqd,ad->aq", dd, U)
    p_right = torch.einsum("aqd,qd->aq", dd, U)
    Kss = alpha[sites[:, None], sites[None, :]] * (U @ U.T)
    Kss += beta[sites[:, None], sites[None, :]] * p_left * p_right
    dt = (target - neighbors) / ell
    k, at, _ = _radial(dt.square().sum(-1), outputscale, kernel)
    ks = at[sites] * (dt[sites] * U).sum(-1)
    # Noise is projected from the same observed gradient at each site.
    # Directions at one site have correlated noise, not independent errors.
    if sigma_f:
        Kff = Kff + sigma_f**2 * torch.eye(len(neighbors), device=Kff.device, dtype=Kff.dtype)
    if sigma_g:
        same_site = sites[:, None] == sites[None, :]
        Kss = Kss + sigma_g**2 * (directions @ directions.T) * same_site
    K = torch.cat([torch.cat([Kff, Kfs], 1), torch.cat([Kfs.T, Kss], 1)], 0)
    return K, torch.cat([k, ks])


def directional_posterior(
    target,
    neighbors,
    values,
    gradients,
    directions,
    sites,
    *,
    lengthscale,
    outputscale,
    kernel="matern52",
    rtol=1e-12,
    sigma_f=0.0,
    sigma_g=0.0,
):
    K, cross = directional_blocks(
        target,
        neighbors,
        directions,
        sites,
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        sigma_f=sigma_f,
        sigma_g=sigma_g,
    )
    obs = torch.cat([values, (directions * gradients[sites]).sum(-1)])
    return condition_scalar(K, cross, obs, outputscale, rtol=rtol)
