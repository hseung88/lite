from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from lite.methods.common.training import RegressionData
from lite.methods.lite.model import LITEModel
from lite.methods.lite.posterior import predict_marginals


@dataclass
class LITEBOState:
    lengthscale_cfg: float | list[float] | None = None
    outputscale: float | None = None
    sigma_f: float | None = None
    sigma_g: float | None = None
    num_calls: int = 0
    num_refits: int = 0


class DifferentiableLITEBotorchModel:
    def __init__(self, model):
        from botorch.models.model import Model
        from botorch.posteriors.gpytorch import GPyTorchPosterior
        from gpytorch.distributions import MultivariateNormal
        from linear_operator.operators import DiagLinearOperator

        class _Wrapped(Model):
            _num_outputs = 1

            def __init__(self, lite_model):
                super().__init__()
                self.lite_model = lite_model

            @property
            def num_outputs(self):
                return 1

            @property
            def batch_shape(self):
                return torch.Size()

            def posterior(
                self,
                X,
                output_indices=None,
                observation_noise=False,
                posterior_transform=None,
                **kwargs,
            ):
                if output_indices not in (None, [0]):
                    raise ValueError("LITE predicts one scalar function value")
                if X.shape[-2] != 1:
                    raise ValueError(
                        "LITE BO supports q=1; batch candidate evaluation uses leading batch dimensions"
                    )
                data = self.lite_model.predictor.data
                if data is None:
                    raise RuntimeError("LITE must be fitted before prediction")
                mean, var = predict_marginals(
                    X.reshape(-1, X.shape[-1]),
                    data.X_train,
                    data.f_train_obs,
                    data.g_train_obs,
                    lengthscale=data.lengthscale,
                    outputscale=data.outputscale,
                    noise_y_var=self.lite_model.sigma_f,
                    noise_g_var=self.lite_model.sigma_g,
                    kernel=data.kernel_name,
                    gradient_noise_model=self.lite_model.gradient_noise_model,
                    m=self.lite_model.m,
                    prediction_batch_size=self.lite_model.prediction_batch_size,
                    normalize_directions=self.lite_model.normalize_directions,
                    train_scaled=data.X_train_scaled,
                )
                shape = X.shape[:-1]
                mean, var = mean.reshape(shape), var.reshape(shape)
                if torch.is_tensor(observation_noise):
                    var = var + observation_noise.squeeze(-1)
                elif observation_noise:
                    var = var + self.lite_model.sigma_f
                posterior = GPyTorchPosterior(MultivariateNormal(mean, DiagLinearOperator(var)))
                return (
                    posterior_transform(posterior) if posterior_transform is not None else posterior
                )

        self.model = _Wrapped(model).eval()


def fit_lite_surrogate(
    train_X, train_y, train_g, cfg, *, seed, state=None, normalize_directions=True
):
    first_fit = state is None or state.lengthscale_cfg is None
    do_refit = first_fit or state.num_calls % max(1, cfg.lite_refit_every) == 0
    if first_fit:
        if cfg.lite_lengthscale_init == "d_scaled":
            ell = cfg.lite_base_lengthscale * math.sqrt(train_X.shape[-1])
        elif cfg.lite_lengthscale_init == "base":
            ell = cfg.lite_base_lengthscale
        else:
            raise ValueError("lite_lengthscale_init must be 'base' or 'd_scaled'")
        lengthscale = [ell] * train_X.shape[-1] if cfg.use_ard else ell
        outputscale, sigma_f, sigma_g = cfg.outputscale_init, cfg.lite_sigma_f, cfg.lite_sigma_g
        steps = cfg.lite_initial_train_steps
    else:
        lengthscale = state.lengthscale_cfg
        outputscale, sigma_f, sigma_g = state.outputscale, state.sigma_f, state.sigma_g
        steps = cfg.lite_update_train_steps if do_refit else 0
    model = LITEModel(
        m=cfg.lite_m,
        kernel=cfg.kernel,
        outputscale=outputscale,
        sigma_f=sigma_f,
        sigma_g=sigma_g,
        lengthscale=lengthscale,
        lengthscale_init="one",
        lengthscale_init_max_points=0,
        use_ard=cfg.use_ard,
        seed=seed,
        train_steps=steps,
        train_epochs=0,
        train_batch_size=cfg.lite_train_batch_size,
        prediction_batch_size=cfg.lite_prediction_batch_size,
        normalize_directions=normalize_directions,
        lr=cfg.lite_lr,
        weight_decay=cfg.gp_weight_decay,
        learn_lengthscale=cfg.lite_learn_lengthscale,
        learn_outputscale=cfg.lite_learn_outputscale,
        learn_sigma_f=cfg.lite_learn_sigma_f,
        learn_sigma_g=cfg.lite_learn_sigma_g,
        min_sigma_f=cfg.min_noise_var,
        min_sigma_g=0.0,
        gradient_noise_model=cfg.lite_gradient_noise_model,
        lengthscale_min=cfg.lite_lengthscale_min,
        lengthscale_max=cfg.lite_lengthscale_max,
    )
    split = RegressionData(
        X_train=train_X, y_train=train_y.reshape(-1), g_train=train_g, X_test=train_X[:0]
    )
    model.fit(split)
    if state is not None:
        ell = model.lengthscale.detach().cpu().reshape(-1)
        state.lengthscale_cfg = ell.tolist() if cfg.use_ard else float(ell[0])
        state.outputscale, state.sigma_f, state.sigma_g = (
            model.outputscale,
            model.sigma_f,
            model.sigma_g,
        )
        state.num_calls += 1
        state.num_refits += int(steps > 0)
    return DifferentiableLITEBotorchModel(model).model
