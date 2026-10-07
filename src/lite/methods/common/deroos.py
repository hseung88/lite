from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch

from lite.methods.common.utils import cholesky_with_jitter

_L_DENSE_CACHE: Dict[Tuple[int, str, torch.dtype], torch.Tensor] = {}
_C_COLIDX_CACHE: Dict[Tuple[int, str], torch.Tensor] = {}


def _cache_key_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda:{device.index if device.index is not None else 0}"
    return "cpu"


def inv_lengthscale_sq(
    lengthscale: torch.Tensor, d: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if lengthscale.numel() == 1:
        return (
            (1.0 / (lengthscale * lengthscale)).reshape(1).repeat(d).to(device=device, dtype=dtype)
        )
    return (1.0 / (lengthscale * lengthscale)).to(device=device, dtype=dtype)


def k_kp_kpp_from_r(
    r: torch.Tensor, kernel: str, outputscale: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    os = torch.as_tensor(outputscale, device=r.device, dtype=r.dtype)
    if kernel == "rbf":
        k = os * torch.exp(-0.5 * r)
        kp = -0.5 * k
        kpp = 0.25 * k
        return k, kp, kpp
    if kernel == "matern52":
        a = torch.sqrt(torch.clamp(r, min=1e-12)) * math.sqrt(5.0)
        ea = torch.exp(-a)
        k = os * (1.0 + a + (a * a) / 3.0) * ea
        kp = -(5.0 / 6.0) * os * (1.0 + a) * ea
        kpp = (25.0 / 12.0) * os * ea
        return k, kp, kpp
    raise ValueError(f"Unknown kernel: {kernel}")


def function_covariance(X1, X2, lengthscale, outputscale, kernel):
    center = 0.5 * (X1.mean(dim=0, keepdim=True) + X2.mean(dim=0, keepdim=True))

    if lengthscale.numel() == 1:
        X1s = (X1 - center) / lengthscale.reshape(1)
        X2s = (X2 - center) / lengthscale.reshape(1)
    else:
        X1s = (X1 - center) / lengthscale.view(1, -1)
        X2s = (X2 - center) / lengthscale.view(1, -1)

    x1_norm = (X1s * X1s).sum(dim=-1, keepdim=True)
    x2_norm = (X2s * X2s).sum(dim=-1).unsqueeze(0)
    r = x1_norm + x2_norm - 2.0 * (X1s @ X2s.T)
    r = torch.clamp(r, min=0.0)

    k, _, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    return k


def value_grad_cross_blocks(
    Xc: torch.Tensor,
    x: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if x.ndim == 1:
        x = x.view(1, -1)
    d = Xc.shape[1]
    w = inv_lengthscale_sq(lengthscale, d, Xc.device, Xc.dtype)
    delta = Xc - x
    v = delta * w.view(1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    K0 = (-2.0) * kp
    k_f = k.contiguous()
    k_g = (-(K0[:, None] * v)).reshape(Xc.shape[0] * d).contiguous()
    k_xx = torch.as_tensor(outputscale, device=Xc.device, dtype=Xc.dtype)
    return k_f, k_g, k_xx


def value_grad_cross_blocks_many(
    Xc: torch.Tensor,
    Xq: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if Xq.ndim == 1:
        Xq = Xq.view(1, -1)
    M, d = Xc.shape
    Q = Xq.shape[0]
    w = inv_lengthscale_sq(lengthscale, d, Xc.device, Xc.dtype)
    delta = Xc[:, None, :] - Xq[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    K0 = (-2.0) * kp
    k_f = k.contiguous()
    k_g = (-(K0[:, :, None] * v)).permute(0, 2, 1).reshape(M * d, Q).contiguous()
    k_xx = torch.full((Q,), float(outputscale), device=Xc.device, dtype=Xc.dtype)
    return k_f, k_g, k_xx


def full_value_grad_covariance(
    X: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
) -> torch.Tensor:
    n, d = X.shape
    w = inv_lengthscale_sq(lengthscale, d, X.device, X.dtype)
    delta = X[:, None, :] - X[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    k, kp, kpp = k_kp_kpp_from_r(r, kernel, outputscale)
    K0 = (-2.0) * kp
    C2 = (-4.0) * kpp
    diagW = torch.diag(w)
    outer = v[..., :, None] * v[..., None, :]
    Kgg = K0[..., None, None] * diagW.view(1, 1, d, d) + C2[..., None, None] * outer
    Kfg = K0[..., None] * v
    Kgf = -Kfg
    blk = torch.zeros((n, n, d + 1, d + 1), device=X.device, dtype=X.dtype)
    blk[:, :, 0, 0] = k
    blk[:, :, 0, 1:] = Kfg
    blk[:, :, 1:, 0] = Kgf
    blk[:, :, 1:, 1:] = Kgg
    K = blk.permute(0, 2, 1, 3).reshape(n * (d + 1), n * (d + 1)).contiguous()
    K = 0.5 * (K + K.T)
    if sigma_f > 0.0:
        idx_f = torch.arange(0, n * (d + 1), step=d + 1, device=X.device, dtype=torch.long)
        K[idx_f, idx_f] += sigma_f * sigma_f
    if sigma_g > 0.0:
        idx_f = torch.arange(0, n * (d + 1), step=d + 1, device=X.device, dtype=torch.long)
        ar = torch.arange(1, d + 1, device=X.device, dtype=torch.long)
        idx_g = (idx_f[:, None] + ar[None, :]).reshape(-1)
        K[idx_g, idx_g] += sigma_g * sigma_g
    return K


def get_L_dense(M: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (M, _cache_key_device(device), dtype)
    cached = _L_DENSE_CACHE.get(key)
    if cached is not None:
        return cached
    dim = M * M
    L = -torch.eye(dim, device=device, dtype=dtype)
    a = torch.arange(M, device=device, dtype=torch.long)
    row = (a * M + a).repeat_interleave(M)
    col = a.repeat_interleave(M) * M + torch.arange(M, device=device).repeat(M)
    L[row, col] += 1.0
    _L_DENSE_CACHE[key] = L
    return L


def get_C_colidx(M: int, device: torch.device) -> torch.Tensor:
    key = (M, _cache_key_device(device))
    cached = _C_COLIDX_CACHE.get(key)
    if cached is not None:
        return cached
    dim = M * M
    col = torch.arange(dim, device=device, dtype=torch.long).reshape(M, M).t().reshape(-1)
    _C_COLIDX_CACHE[key] = col
    return col


def cholesky_ex_stacked_with_jitter(
    K: torch.Tensor, jitter0: float = 1e-8, jitter_max: float = 1e-3
):
    B, M, _ = K.shape
    identity = torch.eye(M, device=K.device, dtype=K.dtype).expand(B, M, M)
    jitter = 0.0
    last_info = None
    while True:
        Kj = K if jitter == 0.0 else (K + jitter * identity)
        L, info = torch.linalg.cholesky_ex(Kj)
        if bool((info == 0).all().item()):
            return L, float(jitter)
        last_info = info
        jitter = jitter0 if jitter == 0.0 else jitter * 10.0
        if jitter > jitter_max:
            raise RuntimeError(
                f"stacked cholesky_ex failed up to jitter={jitter_max}. last info={last_info}"
            )


def ldl_factor_ex_stacked_with_jitter(
    T: torch.Tensor, jitter0: float = 1e-8, jitter_max: float = 1e-3
):
    B, dim, _ = T.shape
    identity = torch.eye(dim, device=T.device, dtype=T.dtype).expand(B, dim, dim)
    jitter = 0.0
    last_info = None
    while True:
        Tj = T if jitter == 0.0 else (T + jitter * identity)
        LD, piv, info = torch.linalg.ldl_factor_ex(Tj, hermitian=True, check_errors=False)
        if bool((info == 0).all().item()):
            return LD, piv, float(jitter)
        last_info = info
        jitter = jitter0 if jitter == 0.0 else jitter * 10.0
        if jitter > jitter_max:
            raise RuntimeError(
                f"stacked ldl_factor_ex failed up to jitter={jitter_max}. last info={last_info}"
            )


def apply_pair_laplacian_stacked(V: torch.Tensor, M: int) -> torch.Tensor:
    B, dim, R = V.shape
    Vmat = V.reshape(B, M, M, R)
    row_sums = Vmat.sum(dim=2)
    Av = torch.zeros_like(Vmat)
    diag = torch.arange(M, device=V.device)
    Av[:, diag, diag, :] = row_sums
    return (Av - Vmat).reshape(B, dim, R)


def apply_pair_laplacian_transpose_stacked(V: torch.Tensor, M: int) -> torch.Tensor:
    B, dim, R = V.shape
    Vmat = V.reshape(B, M, M, R)
    diag = torch.arange(M, device=V.device)
    diag_vals = Vmat[:, diag, diag, :]
    ATv = diag_vals.unsqueeze(2).expand(B, M, M, R)
    return (ATv - Vmat).reshape(B, dim, R)


@dataclass(slots=True)
class DeroosGradientState:
    Xc: torch.Tensor
    w: torch.Tensor
    M: int
    D: int
    Lk: torch.Tensor
    Kinv: torch.Tensor
    LD: torch.Tensor
    piv: torch.Tensor


def prepare_deroos_gradient_inverse(
    Xc: torch.Tensor,
    w: torch.Tensor,
    K0_local: torch.Tensor,
    Kpp4_local: torch.Tensor,
    G_local: torch.Tensor,
) -> DeroosGradientState:
    device, dtype = Xc.device, Xc.dtype
    B, M, D = Xc.shape
    dim = M * M

    K0 = 0.5 * (K0_local + K0_local.transpose(-1, -2))
    Lk, _ = cholesky_ex_stacked_with_jitter(K0)
    identity = torch.eye(M, device=device, dtype=dtype).expand(B, M, M)
    Kinv = torch.cholesky_solve(identity, Lk)
    Kinv = 0.5 * (Kinv + Kinv.transpose(-1, -2))

    G = 0.5 * (G_local + G_local.transpose(-1, -2))
    Kpp4 = 0.5 * (Kpp4_local + Kpp4_local.transpose(-1, -2))
    inv = 1.0 / torch.clamp(Kpp4, min=Kpp4.new_tensor(1e-14))
    vals = inv.reshape(B, dim)

    row = torch.arange(dim, device=device)
    col = get_C_colidx(M, device)
    Cinv = torch.zeros((B, dim, dim), device=device, dtype=dtype)
    Cinv[:, row, col] = vals
    Cinv = 0.5 * (Cinv + Cinv.transpose(-1, -2))

    L = get_L_dense(M, device, dtype).contiguous()
    A = torch.einsum("bij,bkl->bikjl", Kinv, G).reshape(B, dim, dim).contiguous()
    AL = torch.matmul(A, L)
    UT = torch.matmul(L.t().unsqueeze(0), AL)
    T = Cinv + UT
    T = 0.5 * (T + T.transpose(-1, -2))
    LD, piv, _ = ldl_factor_ex_stacked_with_jitter(T)
    return DeroosGradientState(Xc=Xc, w=w, M=M, D=D, Lk=Lk, Kinv=Kinv, LD=LD, piv=piv)


def apply_deroos_gradient_inverse(state: DeroosGradientState, V: torch.Tensor) -> torch.Tensor:
    Xc = state.Xc
    w = state.w
    M = state.M
    D = state.D
    Bsz, MD, R = V.shape
    if MD != M * D:
        raise ValueError(f"Expected V second dimension {M * D}, got {MD}")

    dY = V.reshape(Bsz, M, D, R).permute(0, 2, 1, 3).contiguous()
    dY = dY / w.view(1, D, 1, 1)
    dY2 = dY.permute(0, 2, 1, 3).reshape(Bsz, M, D * R).contiguous()
    sol = torch.cholesky_solve(dY2, state.Lk)
    Ymat = sol.reshape(Bsz, M, D, R).permute(0, 2, 1, 3).contiguous()

    wY = Ymat * w.view(1, D, 1, 1)
    P = torch.einsum("bid,bdjr->bijr", Xc, wY).contiguous()
    pvec = P.permute(0, 2, 1, 3).reshape(Bsz, M * M, R).contiguous()
    rhs = apply_pair_laplacian_transpose_stacked(pvec, M)
    s = torch.linalg.ldl_solve(state.LD, state.piv, rhs, hermitian=True)
    tvec = apply_pair_laplacian_stacked(s, M)
    Tmat = tvec.reshape(Bsz, M, M, R).permute(0, 2, 1, 3).contiguous()
    a = torch.einsum("bijr,bjk->bikr", Tmat, state.Kinv).contiguous()
    corr = torch.einsum("bdm,bmnr->bdnr", Xc.permute(0, 2, 1), a).contiguous()
    Z = Ymat - corr
    return Z.permute(0, 2, 1, 3).reshape(Bsz, M * D, R).contiguous()


@dataclass(slots=True)
class ValueGradDeroosConditioner:
    Xc: torch.Tensor
    yc: torch.Tensor
    gc_flat: torch.Tensor
    lengthscale: torch.Tensor
    outputscale: float
    kernel: str
    sigma_f: float
    w: torch.Tensor
    Kff: torch.Tensor
    Kfg: torch.Tensor
    Kgf: torch.Tensor
    gradient_inverse: DeroosGradientState
    Dinv_Kgf: torch.Tensor
    schur_chol: torch.Tensor


def prepare_value_grad_deroos_conditioner(
    Xc: torch.Tensor,
    yc: torch.Tensor,
    gc: torch.Tensor,
    *,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
) -> ValueGradDeroosConditioner:
    if sigma_g != 0.0:
        raise ValueError("deRoos value-gradient conditioner currently assumes sigma_g=0.")
    M, d = Xc.shape
    device, dtype = Xc.device, Xc.dtype
    w = inv_lengthscale_sq(lengthscale, d, device, dtype)

    Kff = function_covariance(Xc, Xc, lengthscale, outputscale, kernel)
    Kff = 0.5 * (Kff + Kff.T)
    if sigma_f > 0.0:
        Kff = Kff + (sigma_f * sigma_f) * torch.eye(M, device=device, dtype=dtype)

    delta_cc = Xc[:, None, :] - Xc[None, :, :]
    v_cc = delta_cc * w.view(1, 1, d)
    r_cc = (delta_cc * v_cc).sum(dim=-1)
    _, kp_cc, kpp_cc = k_kp_kpp_from_r(r_cc, kernel, outputscale)
    K0_cc = (-2.0) * kp_cc
    Kpp4_cc = 4.0 * kpp_cc

    Kfg4 = K0_cc[:, :, None] * v_cc
    Kfg = Kfg4.reshape(M, M * d).contiguous()
    Kgf = (-Kfg4).permute(0, 2, 1).reshape(M * d, M).contiguous()

    G_local = (Xc * w.view(1, d)) @ Xc.T
    G_local = 0.5 * (G_local + G_local.T)
    state = prepare_deroos_gradient_inverse(
        Xc.unsqueeze(0).contiguous(),
        w,
        K0_cc.unsqueeze(0).contiguous(),
        Kpp4_cc.unsqueeze(0).contiguous(),
        G_local.unsqueeze(0).contiguous(),
    )
    Dinv_Kgf = apply_deroos_gradient_inverse(state, Kgf.unsqueeze(0)).squeeze(0)
    S = Kff - Kfg @ Dinv_Kgf
    S = 0.5 * (S + S.T)
    schur_chol = cholesky_with_jitter(S)
    return ValueGradDeroosConditioner(
        Xc=Xc,
        yc=yc.reshape(M).contiguous(),
        gc_flat=gc.reshape(M * d).contiguous(),
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        sigma_f=sigma_f,
        w=w,
        Kff=Kff,
        Kfg=Kfg,
        Kgf=Kgf,
        gradient_inverse=state,
        Dinv_Kgf=Dinv_Kgf,
        schur_chol=schur_chol,
    )


def predict_one_value_grad_deroos(
    cond: ValueGradDeroosConditioner, x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim == 1:
        x = x.view(1, -1)
    M, d = cond.Xc.shape
    k_f, k_g, k_xx = value_grad_cross_blocks(
        cond.Xc, x, cond.lengthscale, cond.outputscale, cond.kernel
    )
    Kfi = k_f.view(M, 1)
    Kgi0 = k_g.view(M * d, 1)
    Dinv_Kgi = apply_deroos_gradient_inverse(cond.gradient_inverse, Kgi0.unsqueeze(0)).squeeze(0)
    rhs_f = Kfi - cond.Kfg @ Dinv_Kgi
    x_f = torch.cholesky_solve(rhs_f, cond.schur_chol).squeeze(-1)
    x_g = Dinv_Kgi.squeeze(-1) - cond.Dinv_Kgf @ x_f
    mean = torch.dot(x_f, cond.yc) + torch.dot(x_g, cond.gc_flat)
    var = k_xx - torch.dot(k_f, x_f) - torch.dot(k_g, x_g)
    var = torch.clamp(var, min=torch.finfo(x.dtype).eps)
    return mean, var


def predict_all_value_grad_deroos(
    cond: ValueGradDeroosConditioner,
    X_eval: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    if X_eval.ndim == 1:
        X_eval = X_eval.view(1, -1)
    Kfi, Kgi0, k_xx = value_grad_cross_blocks_many(
        cond.Xc, X_eval, cond.lengthscale, cond.outputscale, cond.kernel
    )
    Dinv_Kgi = apply_deroos_gradient_inverse(cond.gradient_inverse, Kgi0.unsqueeze(0)).squeeze(0)
    rhs_f = Kfi - cond.Kfg @ Dinv_Kgi
    x_f = torch.cholesky_solve(rhs_f, cond.schur_chol)
    x_g = Dinv_Kgi - cond.Dinv_Kgf @ x_f
    mean = (x_f * cond.yc[:, None]).sum(dim=0) + (x_g * cond.gc_flat[:, None]).sum(dim=0)
    var = k_xx - (Kfi * x_f).sum(dim=0) - (Kgi0 * x_g).sum(dim=0)
    var = torch.clamp(var, min=torch.finfo(X_eval.dtype).eps)
    return mean.contiguous(), var.contiguous()


def deroos_apply_U(pair_weights: torch.Tensor, X: torch.Tensor, w: torch.Tensor) -> torch.Tensor:

    n, d = X.shape
    if pair_weights.numel() != n * n:
        raise ValueError(f"Expected {n * n} pair weights, got {pair_weights.numel()}")
    pair_mat = pair_weights.reshape(1, n * n, 1)
    lap_pair = apply_pair_laplacian_stacked(pair_mat, n).reshape(n, n)
    return (lap_pair @ X) * w.view(1, d)


def deroos_apply_Ut_Ainv(
    G: torch.Tensor,
    X: torch.Tensor,
    w: torch.Tensor,
    K0_inv: torch.Tensor,
) -> torch.Tensor:

    n, d = X.shape
    if G.shape != (n, d):
        raise ValueError(f"Expected G shape {(n, d)}, got {tuple(G.shape)}")

    Y = K0_inv @ (G / w.view(1, d))

    P = Y @ (X * w.view(1, d)).T
    return apply_pair_laplacian_transpose_stacked(P.reshape(1, n * n, 1), n).reshape(n * n)


@dataclass(slots=True)
class DeroosGradientSamplerState:
    X: torch.Tensor
    w: torch.Tensor
    K0: torch.Tensor
    K0_chol: torch.Tensor
    K0_inv: torch.Tensor
    R_eigvecs: torch.Tensor
    R_eigvals: torch.Tensor
    small_sqrt_minus_I: torch.Tensor


def _deroos_pair_C(Kpp4: torch.Tensor) -> torch.Tensor:
    n = Kpp4.shape[0]
    device, dtype = Kpp4.device, Kpp4.dtype
    dim = n * n
    row = torch.arange(dim, device=device, dtype=torch.long)
    col = get_C_colidx(n, device)
    C = torch.zeros((dim, dim), device=device, dtype=dtype)
    C[row, col] = Kpp4.reshape(-1)
    return 0.5 * (C + C.T)


def prepare_deroos_gradient_sampler(
    X: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    rank_tol: float | None = None,
) -> DeroosGradientSamplerState:

    n, d = X.shape
    device, dtype = X.device, X.dtype
    w = inv_lengthscale_sq(lengthscale, d, device, dtype)

    delta = X[:, None, :] - X[None, :, :]
    v = delta * w.view(1, 1, d)
    r = (delta * v).sum(dim=-1)
    _, kp, kpp = k_kp_kpp_from_r(r, kernel, outputscale)
    K0 = (-2.0) * kp
    K0 = 0.5 * (K0 + K0.T)
    try:
        K0_chol = torch.linalg.cholesky(K0)
    except RuntimeError:
        K0_chol = cholesky_with_jitter(K0)
    eye_n = torch.eye(n, device=device, dtype=dtype)
    K0_inv = torch.cholesky_solve(eye_n, K0_chol)
    K0_inv = 0.5 * (K0_inv + K0_inv.T)

    Kpp4 = 4.0 * kpp
    C = _deroos_pair_C(0.5 * (Kpp4 + Kpp4.T))

    G = (X * w.view(1, d)) @ X.T
    G = 0.5 * (G + G.T)
    L = get_L_dense(n, device, dtype).contiguous()
    pair_gram = torch.einsum("ij,kl->ikjl", K0_inv, G).reshape(n * n, n * n).contiguous()
    R = L.T @ pair_gram @ L
    R = 0.5 * (R + R.T)

    evals, evecs = torch.linalg.eigh(R)
    max_eval = torch.clamp(evals.max(), min=evals.new_tensor(1.0))
    if rank_tol is None:
        rank_tol = 100.0 * torch.finfo(dtype).eps * max(n * n, 1)
    keep = evals > (rank_tol * max_eval)
    if not bool(keep.any().item()):
        return DeroosGradientSamplerState(
            X=X,
            w=w,
            K0=K0,
            K0_chol=K0_chol,
            K0_inv=K0_inv,
            R_eigvecs=evecs[:, :0].contiguous(),
            R_eigvals=evals[:0].contiguous(),
            small_sqrt_minus_I=torch.zeros((0, 0), device=device, dtype=dtype),
        )

    Q = evecs[:, keep].contiguous()
    lam = evals[keep].contiguous()
    sqrt_lam = torch.sqrt(lam)
    C_range = Q.T @ C @ Q
    S = (
        torch.eye(lam.numel(), device=device, dtype=dtype)
        + (sqrt_lam[:, None] * C_range) * sqrt_lam[None, :]
    )
    S = 0.5 * (S + S.T)
    s_eval, s_evec = torch.linalg.eigh(S)

    min_allowed = (
        -1000.0
        * torch.finfo(dtype).eps
        * torch.clamp(s_eval.abs().max(), min=s_eval.new_tensor(1.0))
    )
    if bool((s_eval < min_allowed).any().item()):
        raise RuntimeError(
            f"deRoos sampler square-root matrix is not PSD. min eigenvalue={float(s_eval.min().item()):.3e}"
        )
    s_sqrt = torch.sqrt(torch.clamp(s_eval, min=1e-12))
    S_sqrt = (s_evec * s_sqrt.view(1, -1)) @ s_evec.T
    small_sqrt_minus_I = S_sqrt - torch.eye(lam.numel(), device=device, dtype=dtype)
    small_sqrt_minus_I = 0.5 * (small_sqrt_minus_I + small_sqrt_minus_I.T)

    return DeroosGradientSamplerState(
        X=X,
        w=w,
        K0=K0,
        K0_chol=K0_chol,
        K0_inv=K0_inv,
        R_eigvecs=Q,
        R_eigvals=lam,
        small_sqrt_minus_I=small_sqrt_minus_I.contiguous(),
    )


def _deroos_lowrank_sqrt_update_action(
    beta: torch.Tensor, state: DeroosGradientSamplerState
) -> torch.Tensor:

    Q = state.R_eigvecs
    lam = state.R_eigvals
    if lam.numel() == 0:
        return torch.zeros_like(beta)
    sqrt_lam = torch.sqrt(lam)
    coeff = (Q.T @ beta) / sqrt_lam
    coeff = state.small_sqrt_minus_I @ coeff
    coeff = coeff / sqrt_lam
    return Q @ coeff


def sample_gradients_deroos(
    X: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    generator: torch.Generator | None = None,
    state: DeroosGradientSamplerState | None = None,
) -> torch.Tensor:

    if state is None:
        state = prepare_deroos_gradient_sampler(X, lengthscale, outputscale, kernel)
    n, d = X.shape
    eps = torch.randn((n, d), generator=generator, device=X.device, dtype=X.dtype)

    base = (state.K0_chol @ eps) * torch.sqrt(state.w).view(1, d)
    beta = deroos_apply_Ut_Ainv(base, state.X, state.w, state.K0_inv)
    pair_update = _deroos_lowrank_sqrt_update_action(beta, state)
    return base + deroos_apply_U(pair_update, state.X, state.w)


def sample_value_grad_observations_deroos(
    X: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float,
    kernel: str,
    *,
    sigma_f: float = 0.0,
    sigma_g: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if sigma_g != 0.0:
        raise ValueError("de Roos exact sampler currently requires sigma_g=0.")
    n, d = X.shape
    state = prepare_deroos_gradient_sampler(X, lengthscale, outputscale, kernel)
    g = sample_gradients_deroos(
        X, lengthscale, outputscale, kernel, generator=generator, state=state
    )

    zeros = torch.zeros(n, device=X.device, dtype=X.dtype)
    cond = prepare_value_grad_deroos_conditioner(
        X,
        zeros,
        g,
        lengthscale=lengthscale,
        outputscale=outputscale,
        kernel=kernel,
        sigma_f=sigma_f,
        sigma_g=0.0,
    )
    x_g = (
        apply_deroos_gradient_inverse(cond.gradient_inverse, g.reshape(n * d, 1).unsqueeze(0))
        .squeeze(0)
        .squeeze(-1)
    )
    mean_f = cond.Kfg @ x_g
    eps_f = torch.randn((n,), generator=generator, device=X.device, dtype=X.dtype)
    f = mean_f + cond.schur_chol @ eps_f
    z = torch.cat([f.unsqueeze(-1), g], dim=-1).reshape(-1).contiguous()
    return f.contiguous(), g.contiguous(), z
