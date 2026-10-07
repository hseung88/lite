import torch

"""
Copyright (c) 2022, Wesley Maddox, Andres Potapczynski, Andrew Gordon Wilson
All rights reserved.

This file contains modifications to original binary 
"""


def create_placeholders(rhs, residual, batch_shape):
    precond_residual = residual.clone()
    curr_conjugate_vec = precond_residual
    residual_inner_prod = precond_residual.mul(residual).sum(-2, keepdim=True)

    mul_storage = torch.empty_like(residual)
    alpha = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=residual.dtype, device=residual.device)
    beta = torch.empty_like(alpha)
    is_zero = torch.empty(*batch_shape, 1, rhs.size(-1), dtype=torch.bool, device=residual.device)

    return (
        curr_conjugate_vec,
        residual_inner_prod,
        mul_storage,
        beta,
        alpha,
        is_zero,
    )


def initialize_cg(matmul_closure, rhs, stop_updating_after):
    initial_guess = torch.zeros_like(rhs)

    residual = rhs - matmul_closure(initial_guess)
    batch_shape = residual.shape[:-2]

    result = initial_guess.expand_as(residual).contiguous()

    residual_norm = residual.norm(2, dim=-2, keepdim=True)
    has_converged = torch.lt(residual_norm, stop_updating_after)

    state = (result, has_converged, residual, batch_shape, residual_norm)
    out = create_placeholders(rhs, residual, batch_shape)
    return state, out


def take_cg_step(Ap0, x0, r0, gamma0, p0, alpha, beta, mul_storage, has_converged, eps, is_zero):
    torch.mul(p0, Ap0, out=mul_storage)
    torch.sum(mul_storage, dim=-2, keepdim=True, out=alpha)

    torch.lt(alpha, eps, out=is_zero)
    alpha.masked_fill_(is_zero, 1)
    torch.div(gamma0, alpha, out=alpha)
    alpha.masked_fill_(is_zero, 0)
    alpha.masked_fill_(has_converged, 0)

    torch.addcmul(r0, -alpha, Ap0, out=r0)

    precond_residual = r0.clone()

    x0 = torch.addcmul(x0, alpha, p0, out=x0)

    beta.resize_as_(gamma0).copy_(gamma0)
    torch.mul(r0, precond_residual, out=mul_storage)
    torch.sum(mul_storage, -2, keepdim=True, out=gamma0)
    torch.lt(beta, eps, out=is_zero)
    beta.masked_fill_(is_zero, 1)
    torch.div(gamma0, beta, out=beta)
    beta.masked_fill_(is_zero, 0)

    p0.mul_(beta).add_(precond_residual)


def cond_fn(
    k, max_iter, tolerance, residual, has_converged, residual_norm, stop_updating_after, rhs_is_zero
):
    torch.norm(residual, 2, dim=-2, keepdim=True, out=residual_norm)
    residual_norm.masked_fill_(rhs_is_zero, 0)
    torch.lt(residual_norm, stop_updating_after, out=has_converged)
    flag = k >= min(10, max_iter - 1) and bool(residual_norm.mean() < tolerance)
    return flag


def linear_cg(
    matmul_closure,
    rhs,
    tolerance=None,
    eps=1e-10,
    stop_updating_after=1e-10,
    max_iter=None,
):
    rhs_norm = rhs.norm(2, dim=-2, keepdim=True)
    rhs_is_zero = rhs_norm.lt(eps)
    rhs_norm = rhs_norm.masked_fill_(rhs_is_zero, 1)
    rhs = rhs.div(rhs_norm)

    state, out = initialize_cg(matmul_closure, rhs, stop_updating_after)
    x0, has_converged, r0, batch_shape, residual_norm = state
    (p0, gamma0, mul_storage, beta, alpha, is_zero) = out

    for k in range(max_iter):
        Ap0 = matmul_closure(p0)
        take_cg_step(
            Ap0=Ap0,
            x0=x0,
            r0=r0,
            gamma0=gamma0,
            p0=p0,
            alpha=alpha,
            beta=beta,
            mul_storage=mul_storage,
            has_converged=has_converged,
            eps=eps,
            is_zero=is_zero,
        )

        if cond_fn(
            k,
            max_iter,
            tolerance,
            r0,
            has_converged,
            residual_norm,
            stop_updating_after,
            rhs_is_zero,
        ):
            break

    x0 = x0.mul(rhs_norm)
    return x0
