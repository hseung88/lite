from __future__ import annotations

import math

import torch

from lite.methods.common.deroos import k_kp_kpp_from_r
from lite.methods.common.initialization import resolve_kernel_lengthscale
from lite.methods.common.prediction import Prediction
from lite.methods.common.training import RegressionData
from lite.methods.common.utils import cholesky_with_jitter


def _function_covariance(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: float | torch.Tensor,
    kernel: str,
) -> torch.Tensor:

    if lengthscale.numel() == 1:
        X1s = X1 / lengthscale.reshape(1)
        X2s = X2 / lengthscale.reshape(1)
    else:
        ell = lengthscale.reshape(1, -1)
        X1s = X1 / ell
        X2s = X2 / ell
    r = (X1s * X1s).sum(dim=1, keepdim=True) + (X2s * X2s).sum(dim=1, keepdim=True).T
    r = r - 2.0 * (X1s @ X2s.T)
    r = r.clamp_min(0.0)
    k, _, _ = k_kp_kpp_from_r(r, kernel, outputscale)
    return k


class StandardGPModel:
    validation_callback = None
    name = "Standard GP"

    def __init__(
        self,
        *,
        kernel: str,
        outputscale: float,
        sigma_f: float,
        lengthscale,
        lengthscale_init: str,
        lengthscale_init_max_points: int,
        use_ard: bool,
        max_train: int,
        train_steps: int = 0,
        train_epochs: int = 0,
        lr: float = 3e-2,
        weight_decay: float = 0.0,
        learn_lengthscale: bool = True,
        learn_outputscale: bool = True,
        learn_sigma_f: bool = True,
        min_sigma_f: float = 1e-6,
        log_every: int = 0,
    ) -> None:
        self.kernel = kernel
        self.outputscale_init = float(outputscale)
        self.sigma_f_init = float(sigma_f)
        self.lengthscale_cfg = lengthscale
        self.lengthscale_init = lengthscale_init
        self.lengthscale_init_max_points = int(lengthscale_init_max_points)
        self.use_ard = bool(use_ard)
        self.max_train = int(max_train)
        self.train_steps = int(train_steps)
        self.train_epochs = int(train_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.learn_lengthscale = bool(learn_lengthscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.learn_sigma_f = bool(learn_sigma_f)
        self.min_sigma_f = float(min_sigma_f)
        self.log_every = int(log_every)
        self.X_train: torch.Tensor | None = None
        self.y_train: torch.Tensor | None = None
        self.lengthscale: torch.Tensor | None = None
        self.outputscale: float = self.outputscale_init
        self.sigma_f: float = self.sigma_f_init
        self.L: torch.Tensor | None = None
        self.alpha: torch.Tensor | None = None
        self.training_history: list[dict[str, float]] = []

    def fit(self, split: RegressionData) -> None:
        if split.X_train.shape[0] > self.max_train:
            raise RuntimeError(
                f"Standard GP skipped because n_train={split.X_train.shape[0]} exceeds standard_gp_max_train={self.max_train}"
            )
        init_lengthscale = resolve_kernel_lengthscale(
            split.X_train,
            lengthscale=self.lengthscale_cfg,
            lengthscale_init=self.lengthscale_init,
            lengthscale_init_max_points=self.lengthscale_init_max_points,
            use_ard=self.use_ard,
        )
        if self.train_steps > 0 or self.train_epochs > 0:
            self.lengthscale, self.outputscale, self.sigma_f = self._train_mll(
                split, init_lengthscale
            )
        else:
            self.lengthscale = init_lengthscale.detach().clone()
            self.outputscale = self.outputscale_init
            self.sigma_f = self.sigma_f_init
        self.X_train = split.X_train
        self.y_train = split.y_train
        K = _function_covariance(
            split.X_train, split.X_train, self.lengthscale, self.outputscale, self.kernel
        )
        K = 0.5 * (K + K.T)
        if self.sigma_f > 0.0:
            K = K + self.sigma_f * torch.eye(K.shape[0], device=K.device, dtype=K.dtype)
        self.L = cholesky_with_jitter(K)
        self.alpha = torch.cholesky_solve(split.y_train.unsqueeze(-1), self.L).squeeze(-1)

    def _train_mll(
        self, split: RegressionData, init_lengthscale: torch.Tensor
    ) -> tuple[torch.Tensor, float, float]:
        X = split.X_train
        y = split.y_train
        device, dtype = X.device, X.dtype
        eps = torch.finfo(dtype).eps
        log_lengthscale = torch.nn.Parameter(
            torch.log(init_lengthscale.detach().clone().clamp_min(eps))
        )
        log_outputscale = torch.nn.Parameter(
            torch.log(
                torch.tensor(self.outputscale_init, device=device, dtype=dtype).clamp_min(eps)
            )
        )
        log_sigma_f_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(
                    max(self.sigma_f_init - self.min_sigma_f, eps), device=device, dtype=dtype
                )
            )
        )
        params: list[torch.nn.Parameter] = []
        if self.learn_lengthscale:
            params.append(log_lengthscale)
        if self.learn_outputscale:
            params.append(log_outputscale)
        if self.learn_sigma_f:
            params.append(log_sigma_f_raw)
        if not params:
            return init_lengthscale.detach().clone(), self.outputscale_init, self.sigma_f_init
        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        n = int(X.shape[0])
        self.training_history.clear()
        n_steps = self.train_steps if self.train_epochs <= 0 else self.train_epochs
        identity = torch.eye(n, device=device, dtype=dtype)
        for step in range(1, n_steps + 1):
            lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()
            )
            outputscale = (
                torch.exp(log_outputscale)
                if self.learn_outputscale
                else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
            )
            sigma_f = (
                self.min_sigma_f + torch.exp(log_sigma_f_raw)
                if self.learn_sigma_f
                else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
            )
            K = _function_covariance(X, X, lengthscale, outputscale, self.kernel)
            K = 0.5 * (K + K.T) + sigma_f * identity
            L = cholesky_with_jitter(K)
            alpha = torch.cholesky_solve(y.unsqueeze(-1), L).squeeze(-1)
            loss = (
                0.5 * torch.dot(y, alpha)
                + torch.sum(torch.log(torch.diagonal(L)))
                + 0.5 * n * math.log(2.0 * math.pi)
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
            if self.log_every > 0 and (step == 1 or step % self.log_every == 0 or step == n_steps):
                cur_lengthscale = (
                    (
                        torch.exp(log_lengthscale)
                        if self.learn_lengthscale
                        else init_lengthscale.detach()
                    )
                    .detach()
                    .clone()
                )
                cur_outputscale = float(
                    (
                        torch.exp(log_outputscale)
                        if self.learn_outputscale
                        else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_f = float(
                    (
                        (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                        if self.learn_sigma_f
                        else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                metrics = {}
                if self.validation_callback is not None:
                    with torch.no_grad():
                        K_cur = _function_covariance(
                            X, X, cur_lengthscale, cur_outputscale, self.kernel
                        )
                        K_cur = 0.5 * (K_cur + K_cur.T) + cur_sigma_f * identity
                        L_cur = cholesky_with_jitter(K_cur)
                        alpha_cur = torch.cholesky_solve(y.unsqueeze(-1), L_cur).squeeze(-1)
                        y_mean = (
                            _function_covariance(
                                split.X_test, X, cur_lengthscale, cur_outputscale, self.kernel
                            )
                            @ alpha_cur
                        )
                        metrics = self.validation_callback(y_mean)
                ell0 = float(cur_lengthscale.reshape(-1)[0].cpu())
                rec = {
                    "step": float(step),
                    "loss": float(loss.detach().cpu()),
                    "lengthscale0": ell0,
                    "outputscale": cur_outputscale,
                    "sigma_f": cur_sigma_f,
                    **metrics,
                }
                self.training_history.append(rec)
                print(
                    "Standard GP train "
                    f"step={int(rec['step'])} loss={rec['loss']:.6g} "
                    f"ell0={rec['lengthscale0']:.6g} os={rec['outputscale']:.6g} sf={rec['sigma_f']:.3g}",
                    flush=True,
                )
        with torch.no_grad():
            final_lengthscale = (
                (torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale)
                .detach()
                .clone()
            )
            final_outputscale = float(
                (
                    torch.exp(log_outputscale)
                    if self.learn_outputscale
                    else torch.tensor(self.outputscale_init, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_f = float(
                (
                    (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                    if self.learn_sigma_f
                    else torch.tensor(self.sigma_f_init, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
        return final_lengthscale, final_outputscale, final_sigma_f

    def predict(self, X: torch.Tensor) -> Prediction:
        if self.X_train is None or self.L is None or self.alpha is None or self.lengthscale is None:
            raise RuntimeError("fit() must be called before predict().")
        K_xt = _function_covariance(
            X, self.X_train, self.lengthscale, self.outputscale, self.kernel
        )
        y_mean = K_xt @ self.alpha
        v = torch.cholesky_solve(K_xt.T, self.L)
        var = torch.as_tensor(self.outputscale, device=X.device, dtype=X.dtype) - torch.sum(
            K_xt * v.T, dim=1
        )
        return Prediction(
            y_mean=y_mean.contiguous(), y_var=var.clamp_min(torch.finfo(X.dtype).eps).contiguous()
        )
