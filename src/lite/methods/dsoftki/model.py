import logging
from typing import Callable

import gpytorch
import gpytorch.constraints
import linear_operator
import numpy as np
import torch
import torch.nn as nn
from gpytorch.kernels import ScaleKernel
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.utils.errors import NotPSDError
from torch.utils.data import DataLoader, TensorDataset

from lite.methods.common.linear_solver.cg import linear_cg
from lite.methods.dsoftki.mll import HutchinsonPseudoLoss

logger = logging.getLogger(__name__)


class VectorToSequence(nn.Module):
    def __init__(self, input_dim, sequence_length, embedding_dim):

        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        self.proj = nn.Sequential(
            nn.Linear(input_dim, sequence_length * embedding_dim),
            nn.SiLU(),
        )

    def forward(self, x):

        B = x.size(0)
        out = self.proj(x)
        out = out.view(B, self.sequence_length, self.embedding_dim)
        return out


class DSoftKI(torch.nn.Module):
    def __init__(
        self,
        kernel: Callable,
        interp_points: torch.Tensor,
        dim: int,
        noise=1e-4,
        deriv_noise=1e-2,
        learn_noise=False,
        use_scale=False,
        learn_T=True,
        min_T=5e-5,
        device="cpu",
        fit_device="cpu",
        dtype=torch.float32,
        solver="solve",
        max_cg_iter=50,
        cg_tolerance=0.1,
        mll_approx="hutchinson",
        fit_chunk_size=1024,
        use_qr=False,
        grad_only=False,
        embed_dim=-1,
        hidden_dim=64,
        use_dot=False,
        per_interp_T=True,
    ) -> None:

        methods = ["solve", "cholesky", "cg"]
        if solver not in methods:
            raise ValueError(f"Method {solver} should be in {methods} ...")

        assert (
            isinstance(max_cg_iter, int)
            and isinstance(fit_chunk_size, int)
            and isinstance(embed_dim, int)
        ), f"{type(max_cg_iter)}{type(fit_chunk_size)}{type(embed_dim)}"
        assert (
            isinstance(noise, float)
            and isinstance(deriv_noise, float)
            and isinstance(cg_tolerance, float)
        ), f"{type(noise)}{type(deriv_noise)}{type(cg_tolerance)}"
        assert (
            isinstance(use_qr, bool)
            and isinstance(grad_only, bool)
            and isinstance(use_dot, bool)
            and isinstance(per_interp_T, bool)
        )

        devices = ["cpu"]
        if torch.cuda.is_available():
            devices += ["cuda"]
            for i in range(torch.cuda.device_count()):
                devices += [f"cuda:{i}"]
        if device not in devices:
            raise ValueError(f"Device {device} should be in {devices} ...")

        super(DSoftKI, self).__init__()

        self.device = device
        self.fit_device = fit_device
        self.dtype = dtype

        self.solve_method = solver
        self.mll_approx = mll_approx

        self.use_qr = use_qr
        self.fit_chunk_size = fit_chunk_size

        self.fit_device = fit_device

        self.noise_constraint = gpytorch.constraints.Positive()
        noise = torch.tensor([noise], dtype=self.dtype, device=self.device)
        noise = self.noise_constraint.inverse_transform(noise)
        if learn_noise:
            self.register_parameter("raw_noise", torch.nn.Parameter(noise))
        else:
            self.raw_noise = noise

        self.deriv_noise_constraint = gpytorch.constraints.Positive()
        deriv_noise = torch.tensor([deriv_noise], dtype=self.dtype, device=self.device)
        deriv_noise = self.deriv_noise_constraint.inverse_transform(deriv_noise)
        if learn_noise:
            self.register_parameter("raw_deriv_noise", torch.nn.Parameter(deriv_noise))
        else:
            self.raw_deriv_noise = deriv_noise

        self.use_scale = use_scale
        if use_scale:
            self.kernel = ScaleKernel(kernel).to(self.device)
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                self.kernel.base_kernel.register_constraint(
                    "raw_lengthscale", gpytorch.constraints.Interval(0.1, 5.0)
                )
        else:
            self.kernel = kernel.to(self.device)

        M = len(interp_points)
        self.embed_dim = embed_dim
        self.use_dot = use_dot
        if embed_dim != -1:
            self.min_T = min_T
            self.T_constraint = gpytorch.constraints.GreaterThan(min_T)
            T = torch.full((M,), 1.0, dtype=self.dtype, device=self.device)
            T = self.T_constraint.inverse_transform(T)
            self.register_parameter("raw_T", torch.nn.Parameter(T))
            self.learn_T = learn_T

            self.hidden_dim = hidden_dim

            self.embedding_query = nn.Sequential(
                nn.Linear(in_features=dim, out_features=self.hidden_dim),
                nn.Tanh(),
                nn.Linear(in_features=self.hidden_dim, out_features=embed_dim),
                nn.Tanh(),
            ).to(device=self.device)

            if use_dot:
                self.embedding_key = nn.Sequential(
                    nn.Linear(in_features=embed_dim, out_features=self.hidden_dim),
                    nn.Tanh(),
                    nn.Linear(in_features=self.hidden_dim, out_features=embed_dim),
                    nn.Tanh(),
                ).to(device=self.device)
            else:
                self.embedding_key = None

            self.register_parameter("interp_points", torch.nn.Parameter(interp_points))

            self._interp = self._embed_interp
        else:
            self.embedding_query = None
            self.register_parameter("interp_points", torch.nn.Parameter(interp_points))
            D = interp_points.shape[-1]
            self.T_constraint = gpytorch.constraints.GreaterThan(5e-5)
            T = 1
            self.learn_T = learn_T
            if per_interp_T:
                T = torch.full(
                    (
                        M,
                        D,
                    ),
                    T,
                    dtype=self.dtype,
                    device=self.device,
                )
            else:
                T = torch.full((D,), T, dtype=self.dtype, device=self.device)
            T = self.T_constraint.inverse_transform(T)
            if learn_T:
                self.register_parameter("raw_T", torch.nn.Parameter(T))
            else:
                self.raw_T = T
            self._interp = self._basic_interp

        self.U_zz = torch.zeros((M, M), dtype=self.dtype, device=self.device)
        self.K_zz_alpha = torch.zeros(M, dtype=self.dtype, device=self.device)
        if True:
            self.alpha = torch.zeros((M, 1), dtype=self.dtype, device=fit_device)
        else:
            self.alpha = torch.zeros((M, 1), dtype=self.dtype, device=self.device)

        self.store_K_hat_xz = True
        self.store_K_hat_xz_val = True

        self.grad_only = grad_only

        self.recompute_fit_artifact = False
        self.fit_buffer = None
        self.fit_b = None
        self.Q = None
        self.R = None

        self.max_cg_iter = max_cg_iter
        self.cg_tol = cg_tolerance
        self.x0 = None

    @property
    def noise(self):
        return self.noise_constraint.transform(self.raw_noise)

    @property
    def deriv_noise(self):
        return self.deriv_noise_constraint.transform(self.raw_deriv_noise)

    @property
    def T(self):
        return self.T_constraint.transform(self.raw_T)

    def get_lengthscale(self) -> float:
        if self.use_scale:
            kernel = self.kernel.base_kernel
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                return self.kernel.base_kernel.lengthscale.cpu()
            else:
                return 1.0
        else:
            kernel = self.kernel
            if hasattr(kernel, "has_lengthscale") and kernel.has_lengthscale:
                return self.kernel.lengthscale.cpu()
            else:
                return 1.0

    def get_outputscale(self) -> float:
        if self.use_scale:
            return self.kernel.outputscale.cpu()
        else:
            return 1.0

    def _mk_cov(self, z: torch.Tensor) -> torch.Tensor:
        return self.kernel(z, z).evaluate()

    def _basic_interp(self, x: torch.Tensor, val=True, grad=True) -> torch.Tensor:

        assert self.interp_points.dtype == torch.float32
        z = self.interp_points
        B = x.shape[0]
        M, D = z.shape

        x_expanded = x.unsqueeze(1).expand(-1, z.shape[0], -1)

        diff = x_expanded / self.T - z

        distances = torch.linalg.vector_norm(diff, ord=2, dim=-1)

        neg_distances = -distances

        W_xz = torch.softmax(neg_distances, dim=-1)

        if val and not grad:
            return W_xz

        dist_deriv = (diff / (distances + 1e-6).unsqueeze(-1)) / self.T

        acc = []
        for j in range(len(z)):
            w_ij = W_xz[:, j]

            delta_jk = torch.zeros(len(z), dtype=self.dtype, device=self.device)
            delta_jk[j] = 1.0

            w_ik = W_xz

            W_xz_deriv = (w_ij.unsqueeze(-1) * (delta_jk.unsqueeze(0) - w_ik)).unsqueeze(-1)

            W_xz_deriv = -W_xz_deriv * dist_deriv

            W_xz_deriv = torch.sum(W_xz_deriv, dim=1)

            acc += [W_xz_deriv.reshape(-1)]

        W_xz_deriv = torch.stack(acc, dim=1)
        assert W_xz_deriv.shape[0] == B * D and W_xz_deriv.shape[1] == M

        if self.grad_only:
            return W_xz_deriv
        else:
            assert W_xz.shape[0] == B and W_xz.shape[1] == M
            W_xz_full = torch.cat([W_xz, W_xz_deriv], dim=0)
            assert W_xz_full.shape[0] == B * (D + 1) and W_xz_full.shape[1] == M
            return W_xz_full

    def _embed_interp(self, x: torch.Tensor, val=True, grad=True):

        B, d = x.shape
        z = self.interp_points
        M, d_p = z.shape

        x = x.clone().detach().requires_grad_(True)

        if self.embed_dim == -1:

            def wrapped(x_in):
                f_x_ = x_in
                diffs_ = f_x_[:, None, :] / self.T - z[None, :, :]
                norms_ = torch.linalg.vector_norm(diffs_, ord=2, dim=-1)
                W_xz = torch.softmax(-norms_, dim=-1)
                return W_xz
        elif self.use_dot:

            def wrapped(x_in):
                f_x_ = self.embedding_query(x_in)
                z_ = self.embedding_key(z)
                dot_products = torch.matmul(f_x_, z_.T)
                W_xz = torch.softmax(-dot_products / self.T, dim=-1)
                return W_xz
        else:

            def wrapped(x_in):
                f_x_ = self.embedding_query(x_in)
                diffs_ = f_x_[:, None, :] - z[None, :, :]
                norms_ = torch.linalg.vector_norm(diffs_, ord=2, dim=-1)
                W_xz = torch.softmax(-norms_ / self.T, dim=-1)
                return W_xz

        w = wrapped(x)

        if val and not grad:
            return w

        _, vjp_fn = torch.func.vjp(wrapped, x)
        eye = torch.eye(M, device=x.device).unsqueeze(0).expand(B, -1, -1)

        dw = torch.stack([vjp_fn(eye[:, k])[0] for k in range(M)], dim=1)
        dw = dw.permute(0, 2, 1).reshape(-1, M)

        if not val and grad:
            return dw
        else:
            W_xz_full = torch.cat([w, dw], dim=0)
            return W_xz_full

    def _solve_system(
        self,
        kxx: linear_operator.operators.LinearOperator,
        full_rhs: torch.Tensor,
        forwards_matmul: Callable = None,
        return_pinv: bool = False,
    ) -> torch.Tensor:
        use_pinv = False
        with torch.no_grad():
            try:
                if self.solve_method == "solve":
                    solve = torch.linalg.solve(kxx, full_rhs)
                elif self.solve_method == "cholesky":
                    L = torch.linalg.cholesky(kxx)
                    solve = torch.cholesky_solve(full_rhs, L)
                elif self.solve_method == "cg":
                    solve = linear_cg(
                        forwards_matmul,
                        full_rhs,
                        max_iter=self.max_cg_iter,
                        tolerance=self.cg_tol,
                    )
                else:
                    raise ValueError(f"Unknown method: {self.solve_method}")
            except (torch.linalg.LinAlgError, NotPSDError) as e:
                logger.warning("Linear solve failed; using pseudoinverse: %s", e)
                solve = torch.linalg.pinv(kxx.evaluate()) @ full_rhs
                use_pinv = True

        solve = torch.nan_to_num(solve)
        return (solve, use_pinv) if return_pinv else solve

    def mll(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:

        K_zz = self._mk_cov(self.interp_points)
        W_xz = self._interp(X)

        if torch.isnan(W_xz).any():
            raise ValueError("NaN detected in W_xz ...")

        try:
            L = psd_safe_cholesky(K_zz)
            if torch.isnan(L).any():
                logger.warning("Non-finite Cholesky factor; retrying in float64")
                L = psd_safe_cholesky(K_zz.double()).float()
                if torch.isnan(L).any():
                    raise torch.linalg.LinAlgError("Non-finite Cholesky factor")
        except (torch.linalg.LinAlgError, NotPSDError):
            logger.warning("Reinitializing interpolation points after Cholesky failure")
            M = K_zz.shape[0]
            D = X.shape[1]
            self.interp_points.data = torch.randn((M, D), device=self.device, dtype=self.dtype)
            K_zz = self._mk_cov(self.interp_points)
            W_xz = self._interp(X)
            L = psd_safe_cholesky(K_zz)

        def exact():

            mean = torch.zeros(len(X) * (X.shape[1] + 1), dtype=self.dtype, device=self.device)

            LK = (W_xz @ L).to(device=self.device)
            noise = torch.cat(
                [
                    torch.ones(len(X), dtype=self.dtype, device=self.device) * self.noise,
                    torch.ones(LK.shape[0] - len(X), dtype=self.dtype, device=self.device)
                    * self.deriv_noise,
                ]
            )

            normal_dist = torch.distributions.lowrank_multivariate_normal.LowRankMultivariateNormal(
                mean, LK, noise, validate_args=None
            )

            ys = y[:, 0]
            d_ys = y[:, 1:]
            y_p = torch.cat([ys, d_ys.reshape(-1)])
            return normal_dist.log_prob(y_p)

        def hutch():

            LK = (W_xz @ L).to(device=self.device)

            hutchinson_mll = HutchinsonPseudoLoss(self, dim=LK.shape[0], num_trace_samples=10)
            if self.grad_only:
                mean = torch.zeros(len(X) * (X.shape[-1]), dtype=self.dtype, device=self.device)
                noise = (
                    torch.ones(LK.shape[0], dtype=self.dtype, device=self.device) * self.deriv_noise
                )
                y_p = y.reshape(-1)
            else:
                mean = torch.zeros(len(X) * (1 + X.shape[-1]), dtype=self.dtype, device=self.device)
                noise = torch.cat(
                    [
                        torch.ones(len(X), dtype=self.dtype, device=self.device) * self.noise,
                        torch.ones(LK.shape[0] - len(X), dtype=self.dtype, device=self.device)
                        * self.deriv_noise,
                    ]
                )
                ys = y[:, 0]
                d_ys = y[:, 1:]
                y_p = torch.cat([ys, d_ys.reshape(-1)])
            return hutchinson_mll(mean, LK, noise, y_p)

        if self.mll_approx == "exact":
            return exact()
        elif self.mll_approx == "hutchinson_fallback":
            try:
                return exact()
            except (torch.linalg.LinAlgError, NotPSDError) as e:
                logger.warning("Exact likelihood failed; using Hutchinson estimate: %s", e)
                return hutch()
        elif self.mll_approx == "hutchinson":
            return hutch()
        else:
            raise ValueError(f"Unknown MLL approximation method: {self.mll_approx}")

    def _qr_solve_fit(
        self, M: int, N: int, D: int, X: torch.Tensor, y: torch.Tensor, K_zz: torch.Tensor
    ) -> bool:
        device = self.fit_device

        D_p = D if self.grad_only else D + 1
        self.D_p = D_p

        if self.fit_buffer is None:
            self.fit_buffer = torch.zeros((N * D_p + M, M), dtype=self.dtype, device=device)
            self.fit_b = torch.zeros(N * D_p, dtype=self.dtype, device=device)
            self.fit_y = torch.zeros(N * D_p, dtype=self.dtype, device=device)
            if self.store_K_hat_xz:
                self.hat_K_xz = torch.zeros((N * D_p, M), dtype=self.dtype, device=device)
            if self.store_K_hat_xz_val:
                self.hat_K_xz_val = torch.zeros((N, M), dtype=self.dtype, device=device)

        with torch.no_grad():
            fit_chunk_size = self.fit_chunk_size
            batches = int(np.floor(N / fit_chunk_size)) + int(N % fit_chunk_size > 0)
            for i in range(batches):
                start_i = i * fit_chunk_size
                end_i = min((i + 1) * fit_chunk_size, N)
                cs = end_i - start_i

                start = start_i * D_p
                end = end_i * D_p

                X_batch = X[start_i:end_i, :]
                W_xz = self._interp(X_batch)
                self.fit_buffer[start:end, :] = (W_xz @ K_zz).to(device)
                if self.store_K_hat_xz:
                    self.hat_K_xz[start:end, :] = self.fit_buffer[start:end, :]
                if self.store_K_hat_xz_val:
                    self.hat_K_xz_val[start_i:end_i, :] = self.fit_buffer[start : start + cs, :]

                self.fit_buffer[start : start + cs, :] /= torch.sqrt(self.noise).to(device)
                self.fit_buffer[start + cs : end, :] /= torch.sqrt(self.deriv_noise).to(device)

                if self.grad_only:
                    d_ys = y[start_i:end_i, 1:]
                    self.fit_y[start:end] = d_ys.reshape(-1).to(device) / torch.sqrt(
                        self.deriv_noise
                    ).to(device)
                else:
                    ys = y[start_i:end_i, 0]
                    d_ys = y[start_i:end_i, 1:]
                    self.fit_y[start:end] = torch.cat(
                        [
                            ys.to(device) / torch.sqrt(self.noise).to(device),
                            d_ys.reshape(-1).to(device) / torch.sqrt(self.deriv_noise).to(device),
                        ]
                    )

        with torch.no_grad():
            psd_safe_cholesky(K_zz, out=self.U_zz, upper=True, max_tries=10)
            if torch.isnan(self.U_zz).any():
                logger.warning("Non-finite Cholesky factor; retrying in float64")
                tmp = psd_safe_cholesky(K_zz.double(), upper=True, max_tries=10)
                self.U_zz[:, :] = tmp.to(self.device)
                if torch.isnan(self.U_zz).any():
                    logger.warning("Reinitializing interpolation points after Cholesky failure")
                    self.interp_points.data = torch.randn(
                        (M, D), device=self.device, dtype=self.dtype
                    )
            self.fit_buffer[N * D_p :, :] = self.U_zz.to(device)

            if self.Q is None:
                self.Q = torch.zeros((N * D_p + M, M), dtype=self.dtype, device=device)
                self.R = torch.zeros((M, M), dtype=self.dtype, device=device)

            torch.linalg.qr(self.fit_buffer, out=(self.Q, self.R))

            torch.linalg.solve_triangular(
                self.R,
                (self.Q.T[:, 0 : N * D_p] @ self.fit_y).unsqueeze(1),
                upper=True,
                out=self.alpha,
            ).squeeze(1)

            torch.matmul(K_zz, self.alpha.to(self.device).squeeze(-1), out=self.K_zz_alpha)

        self.recompute_fit_artifact = True

        return False

    def fit(self, X: torch.Tensor, y: torch.Tensor) -> bool:

        N = len(X)
        M = len(self.interp_points)
        D = X.shape[-1]
        X = X.to(self.device, dtype=self.dtype)
        y = y.to(self.device, dtype=self.dtype)

        K_zz = self._mk_cov(self.interp_points)

        self.N = N
        self.K_zz = K_zz

        return self._qr_solve_fit(M, N, D, X, y, K_zz)

    def pred(self, x_star: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            W_star_z = self._interp(x_star)
            return torch.matmul(W_star_z, self.K_zz_alpha).squeeze(-1)

    def pred_cov(self, x_star: torch.Tensor) -> torch.Tensor:

        device = self.device

        with torch.no_grad():
            W_star_z = self._interp(x_star).to(device)
            Q_star_x = W_star_z @ self.hat_K_xz.T.to(device)

            Q_star_x /= torch.sqrt(self.deriv_noise.to(device))
            batches = int(np.floor(self.N / self.fit_chunk_size)) + int(
                self.N % self.fit_chunk_size > 0
            )
            for i in range(batches):
                start = i * self.fit_chunk_size * self.D_p
                if i == batches - 1:
                    end = self.N - i * self.fit_chunk_size
                else:
                    end = i * self.fit_chunk_size * self.D_p + self.fit_chunk_size
                Q_star_x[start:end, :] *= torch.sqrt(self.deriv_noise.to(device)) / torch.sqrt(
                    self.noise.to(device)
                )
            fit_b = Q_star_x.T

            beta = torch.linalg.solve_triangular(
                self.R.to(device),
                (self.Q.T[:, 0 : self.N * self.D_p].to(device) @ fit_b),
                upper=True,
            )
            Q_star_star = W_star_z @ self.K_zz.to(device) @ W_star_z.T

            res = (
                Q_star_star
                - Q_star_x @ Q_star_x.T
                + (Q_star_x @ self.fit_buffer[: self.N * self.D_p, :].to(device)) @ beta
            )

            del W_star_z
            del Q_star_x
            del fit_b
            del beta
            del Q_star_star

            return torch.clamp(res, min=self.noise.to(device))

    def pred_cov_val(self, x_star: torch.Tensor) -> torch.Tensor:

        device = self.device
        with torch.no_grad():
            if self.recompute_fit_artifact:
                dataloader = DataLoader(
                    TensorDataset(self.hat_K_xz), batch_size=self.fit_chunk_size * self.D_p
                )
                self._tmp_hat_K_xz = torch.cat(
                    [data[0][0 : len(data[0]) // self.D_p, :] for data in dataloader]
                )
                dataloader = DataLoader(
                    TensorDataset(self.Q), batch_size=self.fit_chunk_size * self.D_p
                )
                self._tmp_Q = torch.cat(
                    [data[0][0 : len(data[0]) // self.D_p, :] for data in dataloader]
                )
                self.recompute_fit_artifact = False
                dataloader = DataLoader(
                    TensorDataset(self.fit_buffer[: self.N * self.D_p, :]),
                    batch_size=self.fit_chunk_size * self.D_p,
                )
                self._tmp_fit_buffer = torch.cat(
                    [data[0][0 : len(data[0]) // self.D_p, :] for data in dataloader]
                )
                self.recompute_fit_artifact = False

            W_star_z = self._interp(x_star, val=True, grad=False).to(device)

            Q_star_x = W_star_z @ self._tmp_hat_K_xz.T.to(device)

            Q_star_x /= torch.sqrt(self.noise.to(device))

            beta = torch.linalg.solve_triangular(
                self.R.to(device),
                (self._tmp_Q.T[:, 0 : self.N].to(device) @ Q_star_x.T),
                upper=True,
            )
            Q_star_star = W_star_z @ self.K_zz.to(device) @ W_star_z.T

            res = (
                Q_star_star
                - Q_star_x @ Q_star_x.T
                + (Q_star_x @ self._tmp_fit_buffer.to(device)) @ beta
            )

            del W_star_z
            del Q_star_x
            del beta
            del Q_star_star

            return torch.clamp(res, min=self.noise.to(device))

    def __del__(self):
        if hasattr(self, "fit_buffer"):
            del self.fit_buffer
        if hasattr(self, "fit_b"):
            del self.fit_b
        if hasattr(self, "fit_y"):
            del self.fit_y
        if hasattr(self, "hat_K_xz"):
            del self.hat_K_xz
        if hasattr(self, "hat_K_xz_val"):
            del self.hat_K_xz_val
        if hasattr(self, "_tmp_hat_K_xz"):
            del self._tmp_hat_K_xz
        if hasattr(self, "_tmp_Q"):
            del self._tmp_Q
        if hasattr(self, "_tmp_fit_buffer"):
            del self._tmp_fit_buffer
        if hasattr(self, "raw_T"):
            del self.raw_T
        if hasattr(self, "raw_noise"):
            del self.raw_noise
        if hasattr(self, "raw_deriv_noise"):
            del self.raw_deriv_noise
        if hasattr(self, "interp_points"):
            del self.interp_points
        if hasattr(self, "kernel"):
            del self.kernel
