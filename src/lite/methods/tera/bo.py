from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from lite.methods.common.bo import initial_lengthscale
from lite.methods.common.training import RegressionData
from lite.methods.common.utils import cholesky_with_jitter, scale_inputs
from lite.methods.tera.bo_training import (
    SequentialTERAModel,
    TERAModel,
    _alpha_from_r,
    _beta_from_r,
    _direct_joint_scalar_conditional,
    _func_covariance,
    _projected_gradient_noise_gram,
)


class DifferentiableTERABotorchModel:
    def __init__(self, model: TERAModel, *, prediction_batch_size: int | None = None) -> None:
        if prediction_batch_size is not None and prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        from botorch.models.model import Model

        class _Wrapped(Model):
            _num_outputs = 1

            def __init__(self, tera_model: TERAModel) -> None:
                super().__init__()
                self.tera_model = tera_model
                self.prediction_batch_size = prediction_batch_size

            @property
            def _data(self):
                data = self.tera_model.predictor.data
                if data is None:
                    raise RuntimeError("TERA model has no fitted predictor data.")
                return data

            @property
            def num_outputs(self) -> int:
                return 1

            def posterior(
                self,
                X: torch.Tensor,
                output_indices: list[int] | None = None,
                observation_noise: bool = False,
                **kwargs: Any,
            ):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from linear_operator.operators import DiagLinearOperator

                if output_indices not in (None, [0]):
                    raise ValueError("TERA BO wrapper exposes a single scalar output.")
                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_X = X.reshape(-1, d)
                if self.prediction_batch_size is not None and q != 1:
                    raise ValueError(
                        "tera_batched supports q=1; use leading batch dimensions for candidates"
                    )
                means, variances = self._predict_candidates(flat_X)
                mean = means.reshape(*batch_shape, q, 1)
                var = variances.reshape(*batch_shape, q, 1).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var = var + float(self.tera_model.sigma_f)
                covar = DiagLinearOperator(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(
                    mean, covar, interleaved=True
                )
                return GPyTorchPosterior(mvn)

            def conditioning_indices(
                self, x_eval: torch.Tensor, k: int | None = None
            ) -> torch.Tensor:
                data = self._data
                kk = min(int(self.tera_model.m if k is None else k), int(data.X_train.shape[0]))
                if kk <= 0:
                    return torch.empty(0, device=x_eval.device, dtype=torch.long)
                with torch.no_grad():
                    x2 = x_eval.detach().reshape(1, -1)
                    xs = scale_inputs(x2, data.lengthscale)
                    dists = torch.sum((data.X_train_scaled - xs) ** 2, dim=-1)
                    return torch.topk(dists, k=kk, largest=False).indices.contiguous()

            def posterior_fixed_indices(
                self, X: torch.Tensor, idx: torch.Tensor, observation_noise: bool = False
            ):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from linear_operator.operators import DiagLinearOperator

                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_X = X.reshape(-1, d)
                if self.prediction_batch_size is not None and q != 1:
                    raise ValueError(
                        "tera_batched supports q=1; use leading batch dimensions for candidates"
                    )
                means, variances = self._predict_candidates(flat_X, fixed_indices=idx)
                mean = means.reshape(*batch_shape, q, 1)
                var = variances.reshape(*batch_shape, q, 1).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var = var + float(self.tera_model.sigma_f)
                covar = DiagLinearOperator(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(
                    mean, covar, interleaved=True
                )
                return GPyTorchPosterior(mvn)

            def _predict_candidates(self, flat_X, fixed_indices=None):
                if self.prediction_batch_size is None:
                    means, variances = [], []
                    for x in flat_X:
                        idx = (
                            self.conditioning_indices(x) if fixed_indices is None else fixed_indices
                        )
                        mean, var = self._predict_one(x.view(1, -1), idx)
                        means.append(mean)
                        variances.append(var)
                    return torch.stack(means), torch.stack(variances)

                from .batched_prediction import BatchedTERAPredictor

                data = self._data
                predictor = BatchedTERAPredictor(
                    self.tera_model.m,
                    self.tera_model.gradient_noise_model,
                    self.prediction_batch_size,
                )
                predictor.build(data)
                means, variances = [], []
                k = max(0, min(int(self.tera_model.m), len(data.X_train)))
                for chunk in flat_X.split(self.prediction_batch_size):
                    scaled = scale_inputs(chunk, data.lengthscale)
                    with torch.no_grad():
                        if fixed_indices is not None:
                            ids = fixed_indices.to(device=chunk.device).expand(len(chunk), -1)
                        else:
                            distances = (
                                (data.X_train_scaled[None] - scaled.detach()[:, None])
                                .square()
                                .sum(-1)
                            )
                            ids = distances.topk(k, largest=False).indices
                    # Call the differentiable core, not the MD22 no_grad wrapper.
                    mean, var = predictor._predict_batch(chunk, scaled, ids, centered=False)
                    means.append(mean)
                    variances.append(var)
                return torch.cat(means), torch.cat(variances)

            def _predict_one(
                self, x_eval: torch.Tensor, idx: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                data = self._data
                device, dtype = x_eval.device, x_eval.dtype
                m = int(idx.numel())
                outputscale = torch.as_tensor(data.outputscale, device=device, dtype=dtype)
                k_xx = outputscale.reshape(())
                if m == 0:
                    return x_eval.new_zeros(()), k_xx
                Xc = data.X_train[idx]
                Xcs = data.X_train_scaled[idx]
                xs = scale_inputs(x_eval, data.lengthscale)
                Kff = _func_covariance(Xc, Xc, data.lengthscale, outputscale, data.kernel_name)
                Kff = 0.5 * (Kff + Kff.T)
                if float(data.sigma_f) > 0.0:
                    Kff = Kff + (float(data.sigma_f) ** 2) * torch.eye(
                        m, device=device, dtype=dtype
                    )
                k_fc = _func_covariance(
                    Xc, x_eval, data.lengthscale, outputscale, data.kernel_name
                ).squeeze(-1)
                delta = (Xc - x_eval).T.contiguous()
                delta_s = (Xcs - xs).T.contiguous()
                H = delta_s.T @ delta_s
                cols = H.T.contiguous()
                q_mat = cols[:, None, :] - cols[None, :, :]
                r_i = torch.diagonal(H, 0)
                alpha_i = _alpha_from_r(r_i, data.kernel_name, outputscale)
                delta_cc = Xcs[:, None, :] - Xcs[None, :, :]
                r_cc = (delta_cc * delta_cc).sum(dim=-1)
                alpha_cc = _alpha_from_r(r_cc, data.kernel_name, outputscale)
                beta_cc = _beta_from_r(r_cc, data.kernel_name, outputscale)
                bar_k = ((-alpha_i[:, None]) * cols).reshape(m * m).contiguous()
                Q = (
                    ((-alpha_cc[:, :, None]) * q_mat)
                    .permute(0, 2, 1)
                    .reshape(m * m, m)
                    .contiguous()
                )
                G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m, m)
                G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (
                    q_mat[:, :, :, None] * q_mat[:, :, None, :]
                )
                if data.sigma_g > 0.0:
                    R = _projected_gradient_noise_gram(
                        delta, data.lengthscale, self.tera_model.gradient_noise_model
                    )
                    diag_idx = torch.arange(m, device=device)
                    G0_blocks[diag_idx, diag_idx] = (
                        G0_blocks[diag_idx, diag_idx] + float(data.sigma_g) * R
                    )
                G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m * m, m * m).contiguous()
                q_obs = (data.g_train_obs[idx] @ delta).reshape(-1).contiguous()
                return _direct_joint_scalar_conditional(
                    Kff=Kff,
                    Q=Q,
                    G0=G0,
                    y_c=data.f_train_obs[idx],
                    q_obs=q_obs,
                    k_fc=k_fc,
                    bar_k=bar_k,
                    prior_var=k_xx,
                )

        self.model = _Wrapped(model).eval()


class DifferentiableTargetTERABotorchModel:
    def __init__(self, model: TERAModel, *, target_batch_size: int = 512) -> None:
        from botorch.models.model import Model

        class _Wrapped(Model):
            _num_outputs = 1

            def __init__(self, tera_model: TERAModel, batch_size: int) -> None:
                super().__init__()
                self.tera_model = tera_model
                self.target_batch_size = max(1, int(batch_size))

            @property
            def _data(self):
                data = self.tera_model.predictor.data
                if data is None:
                    raise RuntimeError("TERA model has no fitted predictor data.")
                return data

            @property
            def num_outputs(self) -> int:
                return 1

            def posterior(
                self,
                X: torch.Tensor,
                output_indices: list[int] | None = None,
                observation_noise: bool = False,
                **kwargs: Any,
            ):
                import gpytorch
                from botorch.posteriors.gpytorch import GPyTorchPosterior
                from linear_operator.operators import DiagLinearOperator

                if output_indices not in (None, [0]):
                    raise ValueError("TERA target-directed wrapper exposes a single scalar output.")
                batch_shape, q, d = X.shape[:-2], X.shape[-2], X.shape[-1]
                flat_X = X.reshape(-1, d)
                means: list[torch.Tensor] = []
                vars_: list[torch.Tensor] = []
                for chunk in flat_X.split(self.target_batch_size, dim=0):
                    mu, var = self._predict_target_batch(chunk)
                    means.append(mu)
                    vars_.append(var)
                mean_flat = torch.cat(means, dim=0)
                var_flat = torch.cat(vars_, dim=0).clamp_min(torch.finfo(X.dtype).eps)
                if observation_noise:
                    var_flat = var_flat + float(self.tera_model.sigma_f)
                mean = mean_flat.reshape(*batch_shape, q, 1)
                var = var_flat.reshape(*batch_shape, q, 1)
                covar = DiagLinearOperator(var.reshape(*batch_shape, q))
                mvn = gpytorch.distributions.MultitaskMultivariateNormal(
                    mean, covar, interleaved=True
                )
                return GPyTorchPosterior(mvn)

            def _predict_target_batch(
                self, X_eval: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                data = self._data
                device, dtype = X_eval.device, X_eval.dtype
                B, d = X_eval.shape
                m = min(int(self.tera_model.m), int(data.X_train.shape[0]))
                outputscale = torch.as_tensor(data.outputscale, device=device, dtype=dtype).reshape(
                    ()
                )
                if m <= 0:
                    return X_eval.new_zeros(B), outputscale.expand(B).clone()

                with torch.no_grad():
                    Xs = scale_inputs(X_eval.detach(), data.lengthscale)
                    dists = torch.sum(
                        (data.X_train_scaled[None, :, :] - Xs[:, None, :]) ** 2, dim=-1
                    )
                    idx = torch.topk(dists, k=m, largest=False).indices.contiguous()

                Xc = data.X_train[idx]
                Xcs = data.X_train_scaled[idx]
                yc = data.f_train_obs[idx]
                gc = data.g_train_obs[idx]
                Xs = scale_inputs(X_eval, data.lengthscale)

                delta = Xc - X_eval[:, None, :]
                delta_s = Xcs - Xs[:, None, :]
                H = torch.bmm(delta_s, delta_s.transpose(-2, -1))
                h = torch.diagonal(H, dim1=-2, dim2=-1)
                r_cc = (h[:, :, None] + h[:, None, :] - 2.0 * H).clamp_min(0.0)

                Kff = _func_covariance_from_scaled_sqdist(
                    r_cc.detach(), outputscale, data.kernel_name
                )
                diag = torch.arange(m, device=device)
                Kff[:, diag, diag] = Kff[:, diag, diag] + float(data.sigma_f) ** 2
                k_fc = _func_covariance_from_scaled_sqdist(h, outputscale, data.kernel_name)

                alpha_i = _alpha_from_r(h, data.kernel_name, outputscale)
                alpha_cc = _alpha_from_r(r_cc, data.kernel_name, outputscale)
                beta_cc = _beta_from_r(r_cc, data.kernel_name, outputscale)

                bar_k = alpha_i * h

                Q = alpha_cc * (h[:, :, None] - H)

                G0 = alpha_cc * H + beta_cc * (H - h[:, :, None]) * (h[:, None, :] - H)

                if float(data.sigma_g) > 0.0:
                    if self.tera_model.gradient_noise_model == "iid":
                        r_noise_diag = (delta * delta).sum(dim=-1)
                    elif self.tera_model.gradient_noise_model == "scaled":
                        inv_ell2 = (
                            data.lengthscale.to(device=device, dtype=dtype).square().reciprocal()
                        )
                        if inv_ell2.numel() == 1:
                            r_noise_diag = (delta * delta).sum(dim=-1) * inv_ell2.reshape(())
                        else:
                            r_noise_diag = (delta * delta * inv_ell2.view(1, 1, -1)).sum(dim=-1)
                    else:
                        raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
                    diag = torch.arange(m, device=device)
                    G0[:, diag, diag] = G0[:, diag, diag] + float(data.sigma_g) * r_noise_diag

                q_obs = -torch.einsum("bkd,bkd->bk", gc, delta)

                joint_dim = 2 * m
                K_joint = Kff.new_empty(B, joint_dim, joint_dim)
                K_joint[:, :m, :m] = Kff
                K_joint[:, :m, m:] = Q.transpose(-2, -1)
                K_joint[:, m:, :m] = Q
                K_joint[:, m:, m:] = G0
                L_joint = cholesky_with_jitter(K_joint)

                obs = torch.cat([yc, q_obs], dim=-1)
                cross = torch.cat([k_fc, bar_k], dim=-1)
                rhs = torch.stack([obs, cross], dim=-1)
                sol = torch.cholesky_solve(rhs, L_joint)

                mean = (cross * sol[..., 0]).sum(dim=-1)
                var = torch.clamp(
                    outputscale - (cross * sol[..., 1]).sum(dim=-1),
                    min=torch.finfo(dtype).eps,
                )
                return mean, var

        self.model = _Wrapped(model, target_batch_size).eval()


def _func_covariance_from_scaled_sqdist(
    r2: torch.Tensor,
    outputscale: torch.Tensor,
    kernel_name: str,
) -> torch.Tensor:

    if kernel_name == "rbf":
        return outputscale * torch.exp(-0.5 * r2)
    if kernel_name == "matern52":
        r = torch.sqrt(torch.clamp(r2, min=1e-12))
        a = math.sqrt(5.0) * r
        return outputscale * (1.0 + a + (a * a) / 3.0) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")


@dataclass
class TERABOState:
    lengthscale_cfg: float | list[float] | None = None
    outputscale: float | None = None
    sigma_f: float | None = None
    sigma_g: float | None = None
    num_calls: int = 0
    num_refits: int = 0


def _lengthscale_to_cfg(lengthscale: torch.Tensor, use_ard: bool) -> float | list[float]:
    flat = lengthscale.detach().cpu().reshape(-1)
    if use_ard:
        return [float(x) for x in flat]
    return float(flat[0])


def _should_refit_tera(state: TERABOState | None, cfg: Any) -> bool:
    if state is None or state.lengthscale_cfg is None:
        return True
    every = max(1, int(cfg.tera_refit_every))
    return (int(state.num_calls) % every) == 0


def fit_tera_surrogate(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    train_g: torch.Tensor,
    cfg: Any,
    *,
    seed: int,
    state: TERABOState | None = None,
    prediction_mode: str = "full",
    training_mode: str = "full",
):
    if prediction_mode not in {"full", "batched", "target"}:
        raise ValueError(f"Unknown TERA prediction_mode: {prediction_mode}")
    first_fit = state is None or state.lengthscale_cfg is None
    do_refit = _should_refit_tera(state, cfg)

    if first_fit:
        ell = initial_lengthscale(
            train_X.shape[-1], cfg, device=train_X.device, dtype=train_X.dtype, method="tera"
        )
        lengthscale_cfg: float | list[float] = _lengthscale_to_cfg(ell, cfg.use_ard)
        outputscale = float(cfg.outputscale_init)
        sigma_f = float(cfg.tera_sigma_f)
        sigma_g = float(cfg.tera_sigma_g)
        train_steps = int(cfg.tera_initial_train_steps if do_refit else 0)
    else:
        assert state is not None
        lengthscale_cfg = state.lengthscale_cfg
        outputscale = float(
            state.outputscale if state.outputscale is not None else cfg.outputscale_init
        )
        sigma_f = float(state.sigma_f if state.sigma_f is not None else cfg.tera_sigma_f)
        sigma_g = float(state.sigma_g if state.sigma_g is not None else cfg.tera_sigma_g)
        train_steps = int(cfg.tera_update_train_steps if do_refit else 0)

    if training_mode not in {"full", "target"}:
        raise ValueError(f"Unknown TERA training_mode: {training_mode}")

    split = RegressionData(
        X_train=train_X, y_train=train_y.reshape(-1), g_train=train_g, X_test=train_X[:0]
    )
    model_class = SequentialTERAModel if prediction_mode == "full" else TERAModel
    model = model_class(
        m=cfg.tera_m,
        kernel=cfg.kernel,
        outputscale=outputscale,
        sigma_f=sigma_f,
        sigma_g=sigma_g,
        lengthscale=lengthscale_cfg,
        lengthscale_init="one",
        lengthscale_init_max_points=0,
        use_ard=cfg.use_ard,
        seed=seed,
        train_steps=train_steps,
        train_epochs=0,
        graph_refresh_epochs=0,
        train_batch_size=cfg.tera_train_batch_size,
        lr=cfg.tera_lr,
        weight_decay=cfg.gp_weight_decay,
        learn_lengthscale=cfg.tera_learn_lengthscale and train_steps > 0,
        learn_outputscale=cfg.tera_learn_outputscale and train_steps > 0,
        learn_sigma_f=cfg.tera_learn_sigma_f and train_steps > 0,
        learn_sigma_g=cfg.tera_learn_sigma_g and train_steps > 0,
        min_sigma_f=cfg.min_noise_var,
        min_sigma_g=0.0,
        lengthscale_min=cfg.tera_lengthscale_min,
        lengthscale_max=cfg.tera_lengthscale_max,
        log_every=0,
        gradient_noise_model=cfg.tera_gradient_noise_model,
        training_mode=training_mode,
    )
    model.fit(split)

    if state is not None:
        state.lengthscale_cfg = _lengthscale_to_cfg(model.lengthscale, cfg.use_ard)
        state.outputscale = float(model.outputscale)
        state.sigma_f = float(model.sigma_f)
        state.sigma_g = float(model.sigma_g)
        state.num_calls += 1
        if train_steps > 0:
            state.num_refits += 1

    if prediction_mode == "target":
        return DifferentiableTargetTERABotorchModel(
            model, target_batch_size=int(getattr(cfg, "tera_target_batch_size", 512))
        ).model
    if prediction_mode == "batched":
        return DifferentiableTERABotorchModel(
            model, prediction_batch_size=cfg.tera_prediction_batch_size
        ).model
    return DifferentiableTERABotorchModel(model).model
