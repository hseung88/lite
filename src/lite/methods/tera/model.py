from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch

from lite.methods.common.base import NeighborhoodPredictor
from lite.methods.common.batching import resolve_batch_size
from lite.methods.common.data import SimulatedDataset
from lite.methods.common.deroos import function_covariance
from lite.methods.common.initialization import resolve_kernel_lengthscale
from lite.methods.common.ordering import maximin_ordering, predecessor_neighbors
from lite.methods.common.prediction import Prediction
from lite.methods.common.training import RegressionData
from lite.methods.common.utils import cholesky_with_jitter, scale_inputs


@dataclass(slots=True)
class VecchiaTrainingState:
    X: torch.Tensor
    y: torch.Tensor
    g: torch.Tensor
    neighbors: torch.Tensor
    sample_positions: torch.Tensor


def _direct_joint_scalar_conditional(
    *,
    Kff: torch.Tensor,
    Q: torch.Tensor,
    G0: torch.Tensor,
    y_c: torch.Tensor,
    q_obs: torch.Tensor,
    k_fc: torch.Tensor,
    bar_k: torch.Tensor,
    prior_var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:

    top = torch.cat([Kff, Q.T], dim=1)
    bottom = torch.cat([Q, G0], dim=1)
    K_joint = torch.cat([top, bottom], dim=0)
    K_joint = 0.5 * (K_joint + K_joint.T)

    obs = torch.cat([y_c, q_obs], dim=0)
    cross = torch.cat([k_fc, bar_k], dim=0)
    rhs = torch.stack([obs, cross], dim=1)
    try:
        L_joint = cholesky_with_jitter(K_joint)
        sol = torch.cholesky_solve(rhs, L_joint)
    except torch.linalg.LinAlgError:
        sol = torch.linalg.pinv(K_joint) @ rhs

    mean = torch.dot(cross, sol[:, 0])
    var = torch.clamp(
        prior_var - torch.dot(cross, sol[:, 1]),
        min=torch.finfo(K_joint.dtype).eps,
    )
    return mean, var


class TERAPredictor(NeighborhoodPredictor):
    def __init__(self, m: int, gradient_noise_model: str) -> None:
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.m = int(m)
        self.gradient_noise_model = gradient_noise_model
        self.data: SimulatedDataset | None = None

    def build(self, data: SimulatedDataset) -> None:
        self.data = data

    def _predict_one(
        self, *, x_eval: torch.Tensor, x_eval_scaled: torch.Tensor, idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.data is not None
        data = self.data
        device = x_eval.device
        dtype = x_eval.dtype

        m_local = int(idx.numel())
        k_xx = (
            function_covariance(
                x_eval, x_eval, data.lengthscale, data.outputscale, data.kernel_name
            )
            .reshape(())
            .to(dtype=dtype)
        )
        if m_local == 0:
            return x_eval.new_zeros(()), k_xx

        Xc = data.X_train[idx]
        Xc_scaled = data.X_train_scaled[idx]
        Kff = function_covariance(Xc, Xc, data.lengthscale, data.outputscale, data.kernel_name)
        Kff = 0.5 * (Kff + Kff.T)
        if data.sigma_f > 0.0:
            Kff = Kff + (data.sigma_f**2) * torch.eye(m_local, device=device, dtype=dtype)

        k_fc = function_covariance(
            Xc, x_eval, data.lengthscale, data.outputscale, data.kernel_name
        ).squeeze(-1)

        delta = (Xc - x_eval).T.contiguous()
        delta_scaled = (Xc_scaled - x_eval_scaled).T.contiguous()
        H = delta_scaled.T @ delta_scaled
        cols = H.T.contiguous()
        q = cols[:, None, :] - cols[None, :, :]

        r_i = torch.diagonal(H, 0)
        alpha_i = _alpha_from_r(
            r_i, data.kernel_name, torch.as_tensor(data.outputscale, device=device, dtype=dtype)
        )

        delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
        r_cc = (delta_cc * delta_cc).sum(dim=-1)
        outputscale = torch.as_tensor(data.outputscale, device=device, dtype=dtype)
        alpha_cc = _alpha_from_r(r_cc, data.kernel_name, outputscale)
        beta_cc = _beta_from_r(r_cc, data.kernel_name, outputscale)

        bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * m_local).contiguous()
        Q = (
            ((-alpha_cc[:, :, None]) * q)
            .permute(0, 2, 1)
            .reshape(m_local * m_local, m_local)
            .contiguous()
        )

        G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m_local, m_local)
        G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
        if data.sigma_g > 0.0:
            R = _projected_gradient_noise_gram(delta, data.lengthscale, self.gradient_noise_model)
            diag_idx = torch.arange(m_local, device=device)
            G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + (data.sigma_g) * R
        G0 = (
            G0_blocks.permute(0, 2, 1, 3).reshape(m_local * m_local, m_local * m_local).contiguous()
        )

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


class TERAModel:
    name = "TERA"

    def __init__(
        self,
        *,
        m: int,
        kernel: str,
        outputscale: float,
        sigma_f: float,
        sigma_g: float,
        lengthscale,
        lengthscale_init: str,
        lengthscale_init_max_points: int,
        use_ard: bool,
        seed: int = 0,
        train_steps: int = 0,
        train_epochs: int = 0,
        graph_refresh_epochs: int = 0,
        train_batch_size: int | None = None,
        batch_size: int | None = None,
        prediction_batch_size: int = 1,
        lr: float = 5e-2,
        weight_decay: float = 0.0,
        learn_lengthscale: bool = True,
        learn_outputscale: bool = True,
        learn_sigma_f: bool = True,
        learn_sigma_g: bool = False,
        min_sigma_f: float = 1e-6,
        min_sigma_g: float = 0.0,
        lengthscale_min: float | None = None,
        lengthscale_max: float | None = None,
        log_every: int = 0,
        gradient_noise_model: str = "iid",
        training_mode: Literal["full", "target"] = "full",
        validation_callback=None,
    ) -> None:
        if prediction_batch_size < 1:
            raise ValueError("prediction_batch_size must be positive")
        self.prediction_batch_size = prediction_batch_size
        self.m = int(m)
        self.kernel = kernel
        self.outputscale_init = float(outputscale)
        self.sigma_f_init = float(sigma_f)
        self.sigma_g_init = float(sigma_g)
        self.lengthscale_cfg = lengthscale
        self.lengthscale_init = lengthscale_init
        self.lengthscale_init_max_points = int(lengthscale_init_max_points)
        self.use_ard = bool(use_ard)
        self.seed = int(seed)
        self.train_steps = int(train_steps)
        self.train_epochs = int(train_epochs)
        self.graph_refresh_epochs = int(graph_refresh_epochs)
        self.train_batch_size = resolve_batch_size(
            train_batch_size, batch_size, default=64, name="train_batch_size"
        )
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.learn_lengthscale = bool(learn_lengthscale)
        self.learn_outputscale = bool(learn_outputscale)
        self.learn_sigma_f = bool(learn_sigma_f)
        self.learn_sigma_g = bool(learn_sigma_g)
        self.min_sigma_f = float(min_sigma_f)
        self.min_sigma_g = float(min_sigma_g)
        self.lengthscale_min = None if lengthscale_min is None else float(lengthscale_min)
        self.lengthscale_max = None if lengthscale_max is None else float(lengthscale_max)
        if self.lengthscale_min is not None and self.lengthscale_min <= 0.0:
            raise ValueError("lengthscale_min must be positive when provided.")
        if self.lengthscale_max is not None and self.lengthscale_max <= 0.0:
            raise ValueError("lengthscale_max must be positive when provided.")
        if (
            self.lengthscale_min is not None
            and self.lengthscale_max is not None
            and self.lengthscale_min > self.lengthscale_max
        ):
            raise ValueError("lengthscale_min must be no larger than lengthscale_max.")
        self.log_every = int(log_every)
        if gradient_noise_model not in {"iid", "scaled"}:
            raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
        self.gradient_noise_model = gradient_noise_model
        if training_mode not in {"full", "target"}:
            raise ValueError("training_mode must be either 'full' or 'target'.")
        self.training_mode = training_mode
        self.validation_callback = validation_callback

        self.predictor = self._make_predictor()
        self.lengthscale: torch.Tensor | None = None
        self.outputscale: float = self.outputscale_init
        self.sigma_f: float = self.sigma_f_init
        self.sigma_g: float = self.sigma_g_init
        self.training_history: list[dict[str, float]] = []

    def _make_predictor(self):
        if self.prediction_batch_size > 1:
            from .batched_prediction import BatchedTERAPredictor

            return BatchedTERAPredictor(
                self.m, self.gradient_noise_model, self.prediction_batch_size
            )
        return TERAPredictor(m=self.m, gradient_noise_model=self.gradient_noise_model)

    def _batch_nll(self, **kwargs):
        return _batch_nll(training_mode=self.training_mode, **kwargs)

    def _likelihood_lengthscale(self, lengthscale):
        return self._apply_lengthscale_bounds(lengthscale)

    def _apply_lengthscale_bounds(self, lengthscale: torch.Tensor) -> torch.Tensor:
        if self.lengthscale_min is not None:
            lengthscale = lengthscale.clamp_min(float(self.lengthscale_min))
        if self.lengthscale_max is not None:
            lengthscale = lengthscale.clamp_max(float(self.lengthscale_max))
        return lengthscale

    def _project_log_lengthscale_(self, log_lengthscale: torch.Tensor) -> None:
        lower = math.log(float(self.lengthscale_min)) if self.lengthscale_min is not None else None
        upper = math.log(float(self.lengthscale_max)) if self.lengthscale_max is not None else None
        if lower is None and upper is None:
            return
        with torch.no_grad():
            log_lengthscale.clamp_(min=lower, max=upper)

    def fit(self, split: RegressionData) -> None:
        init_lengthscale = resolve_kernel_lengthscale(
            split.X_train,
            lengthscale=self.lengthscale_cfg,
            lengthscale_init=self.lengthscale_init,
            lengthscale_init_max_points=self.lengthscale_init_max_points,
            use_ard=self.use_ard,
        )
        init_lengthscale = self._apply_lengthscale_bounds(init_lengthscale)
        self.training_history.clear()

        if self.train_steps > 0 or self.train_epochs > 0:
            if self._use_outer_graph_refresh():
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = (
                    self._train_likelihood_with_graph_refresh(
                        split,
                        init_lengthscale,
                    )
                )
            else:
                state = build_training_state(split, init_lengthscale, self.m)
                self.lengthscale, self.outputscale, self.sigma_f, self.sigma_g = (
                    self._train_likelihood(state, split, init_lengthscale)
                )
        else:
            self.lengthscale = init_lengthscale.detach().clone()
            self.outputscale = self.outputscale_init
            self.sigma_f = self.sigma_f_init
            self.sigma_g = self.sigma_g_init

        X_train_scaled = scale_inputs(split.X_train, self.lengthscale)
        dummy_eval = split.X_test[:0]
        data = SimulatedDataset(
            X_train=split.X_train,
            X_train_scaled=X_train_scaled,
            X_eval=dummy_eval,
            X_eval_scaled=dummy_eval,
            lengthscale=self.lengthscale,
            outputscale=self.outputscale,
            sigma_f=math.sqrt(max(float(self.sigma_f), 0.0)),
            sigma_g=self.sigma_g,
            kernel_name=self.kernel,
            f_train_obs=split.y_train,
            g_train_obs=split.g_train,
            z_train_obs=torch.cat([split.y_train.unsqueeze(-1), split.g_train], dim=1).reshape(-1),
            sampling_backend="training",
        )
        self.predictor.build(data)

    def _use_outer_graph_refresh(self) -> bool:
        return (
            self.train_epochs > 0
            and self.graph_refresh_epochs > 0
            and self.use_ard
            and self.learn_lengthscale
        )

    def _train_likelihood_with_graph_refresh(
        self,
        split: RegressionData,
        init_lengthscale: torch.Tensor,
    ) -> tuple[torch.Tensor, float, float, float]:
        remaining_epochs = int(self.train_epochs)
        refresh_epochs = max(1, int(self.graph_refresh_epochs))
        current_lengthscale = init_lengthscale.detach().clone()
        current_outputscale = self.outputscale_init
        current_sigma_f = self.sigma_f_init
        current_sigma_g = self.sigma_g_init
        step_offset = 0

        while remaining_epochs > 0:
            stage_epochs = min(refresh_epochs, remaining_epochs)
            state = build_training_state(split, current_lengthscale, self.m)
            current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g = (
                self._train_likelihood(
                    state,
                    split,
                    current_lengthscale,
                    start_outputscale=current_outputscale,
                    start_sigma_f=current_sigma_f,
                    start_sigma_g=current_sigma_g,
                    train_epochs_override=stage_epochs,
                    train_steps_override=0,
                    step_offset=step_offset,
                )
            )
            step_offset += stage_epochs * math.ceil(
                int(state.sample_positions.numel())
                / max(1, min(self.train_batch_size, int(state.sample_positions.numel())))
            )
            remaining_epochs -= stage_epochs
        return current_lengthscale, current_outputscale, current_sigma_f, current_sigma_g

    def predict(self, X: torch.Tensor) -> Prediction:
        pred = self.predictor.predict_f_marginals(X)
        return Prediction(y_mean=pred.mean, y_var=pred.var)

    def _train_likelihood(
        self,
        state: VecchiaTrainingState,
        split: RegressionData,
        init_lengthscale: torch.Tensor,
        *,
        start_outputscale: float | None = None,
        start_sigma_f: float | None = None,
        start_sigma_g: float | None = None,
        train_epochs_override: int | None = None,
        train_steps_override: int | None = None,
        step_offset: int = 0,
    ) -> tuple[torch.Tensor, float, float, float]:
        device, dtype = state.X.device, state.X.dtype
        eps = torch.finfo(dtype).eps

        init_lengthscale = self._apply_lengthscale_bounds(init_lengthscale)
        log_lengthscale = torch.nn.Parameter(
            torch.log(init_lengthscale.detach().clone().clamp_min(eps))
        )
        self._project_log_lengthscale_(log_lengthscale)
        outputscale0 = (
            self.outputscale_init if start_outputscale is None else float(start_outputscale)
        )
        sigma_f0 = self.sigma_f_init if start_sigma_f is None else float(start_sigma_f)
        sigma_g0 = self.sigma_g_init if start_sigma_g is None else float(start_sigma_g)
        log_outputscale = torch.nn.Parameter(
            torch.log(torch.tensor(outputscale0, device=device, dtype=dtype).clamp_min(eps))
        )
        log_sigma_f_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(max(sigma_f0 - self.min_sigma_f, eps), device=device, dtype=dtype)
            )
        )
        log_sigma_g_raw = torch.nn.Parameter(
            torch.log(
                torch.tensor(max(sigma_g0 - self.min_sigma_g, eps), device=device, dtype=dtype)
            )
        )

        params: list[torch.nn.Parameter] = []
        if self.learn_lengthscale:
            params.append(log_lengthscale)
        if self.learn_outputscale:
            params.append(log_outputscale)
        if self.learn_sigma_f:
            params.append(log_sigma_f_raw)
        if self.learn_sigma_g:
            params.append(log_sigma_g_raw)
        if not params:
            return init_lengthscale.detach().clone(), outputscale0, sigma_f0, sigma_g0

        opt = torch.optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self.seed)
        sample_positions = state.sample_positions.cpu()
        n_sample = int(sample_positions.numel())
        train_batch_size = max(1, min(self.train_batch_size, n_sample))
        steps_per_epoch = math.ceil(n_sample / train_batch_size)
        effective_epochs = (
            self.train_epochs if train_epochs_override is None else int(train_epochs_override)
        )
        effective_steps = (
            self.train_steps if train_steps_override is None else int(train_steps_override)
        )
        total_steps = (
            effective_epochs * steps_per_epoch if effective_epochs > 0 else effective_steps
        )
        perm = torch.empty(0, dtype=torch.long)
        cursor = n_sample

        for step in range(1, total_steps + 1):
            if effective_epochs > 0:
                if perm.numel() == 0 or cursor >= n_sample:
                    perm = torch.randperm(n_sample, generator=gen)
                    cursor = 0
                draw_cpu = perm[cursor : min(cursor + train_batch_size, n_sample)]
                cursor += int(draw_cpu.numel())
                idx = sample_positions[draw_cpu]
            else:
                draw = torch.randint(0, n_sample, (train_batch_size,), generator=gen, device="cpu")
                idx = sample_positions[draw]
            lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale.detach()
            )
            lengthscale = self._likelihood_lengthscale(lengthscale)
            outputscale = (
                torch.exp(log_outputscale)
                if self.learn_outputscale
                else torch.tensor(outputscale0, device=device, dtype=dtype)
            )
            sigma_f = (
                self.min_sigma_f + torch.exp(log_sigma_f_raw)
                if self.learn_sigma_f
                else torch.tensor(sigma_f0, device=device, dtype=dtype)
            )
            sigma_g = (
                self.min_sigma_g + torch.exp(log_sigma_g_raw)
                if self.learn_sigma_g
                else torch.tensor(sigma_g0, device=device, dtype=dtype)
            )

            loss = self._batch_nll(
                state=state,
                target_positions=idx,
                lengthscale=lengthscale,
                outputscale=outputscale,
                sigma_f=sigma_f,
                sigma_g=sigma_g,
                kernel=self.kernel,
                gradient_noise_model=self.gradient_noise_model,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=100.0)
            opt.step()
            if self.learn_lengthscale:
                self._project_log_lengthscale_(log_lengthscale)

            if self.log_every > 0 and (
                step == 1 or step % self.log_every == 0 or step == total_steps
            ):
                cur_lengthscale = (
                    torch.exp(log_lengthscale)
                    if self.learn_lengthscale
                    else init_lengthscale.detach()
                )
                cur_lengthscale = self._apply_lengthscale_bounds(cur_lengthscale).detach().clone()
                cur_outputscale = float(
                    (
                        torch.exp(log_outputscale)
                        if self.learn_outputscale
                        else torch.tensor(outputscale0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_f = float(
                    (
                        (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                        if self.learn_sigma_f
                        else torch.tensor(sigma_f0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                cur_sigma_g = float(
                    (
                        (self.min_sigma_g + torch.exp(log_sigma_g_raw))
                        if self.learn_sigma_g
                        else torch.tensor(sigma_g0, device=device, dtype=dtype)
                    )
                    .detach()
                    .cpu()
                )
                metrics = {}
                if self.validation_callback is not None:
                    with torch.no_grad():
                        predictor = self._make_predictor()
                        X_train_scaled = scale_inputs(split.X_train, cur_lengthscale)
                        dummy_eval = split.X_test[:0]
                        data = SimulatedDataset(
                            X_train=split.X_train,
                            X_train_scaled=X_train_scaled,
                            X_eval=dummy_eval,
                            X_eval_scaled=dummy_eval,
                            lengthscale=cur_lengthscale,
                            outputscale=cur_outputscale,
                            sigma_f=math.sqrt(max(float(cur_sigma_f), 0.0)),
                            sigma_g=cur_sigma_g,
                            kernel_name=self.kernel,
                            f_train_obs=split.y_train,
                            g_train_obs=split.g_train,
                            z_train_obs=torch.cat(
                                [split.y_train.unsqueeze(-1), split.g_train], dim=1
                            ).reshape(-1),
                            sampling_backend="training",
                        )
                        predictor.build(data)
                        pred = predictor.predict_f_marginals(split.X_test)
                        metrics = self.validation_callback(pred.mean)
                rec = {
                    "step": float(step_offset + step),
                    "loss": float(loss.detach().cpu()),
                    "lengthscale": float(cur_lengthscale.reshape(-1)[0].cpu()),
                    "outputscale": cur_outputscale,
                    "sigma_f": cur_sigma_f,
                    "sigma_g": cur_sigma_g,
                    **metrics,
                }
                self.training_history.append(rec)
                print(
                    f"{self.name} train "
                    f"step={int(rec['step'])} loss={rec['loss']:.6g} "
                    f"ell={rec['lengthscale']:.6g} os={rec['outputscale']:.6g} "
                    f"sf={rec['sigma_f']:.3g} sg={rec['sigma_g']:.3g}",
                    flush=True,
                )

        with torch.no_grad():
            final_lengthscale = (
                torch.exp(log_lengthscale) if self.learn_lengthscale else init_lengthscale
            )
            final_lengthscale = self._apply_lengthscale_bounds(final_lengthscale).detach().clone()
            final_outputscale = float(
                (
                    torch.exp(log_outputscale)
                    if self.learn_outputscale
                    else torch.tensor(outputscale0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_f = float(
                (
                    (self.min_sigma_f + torch.exp(log_sigma_f_raw))
                    if self.learn_sigma_f
                    else torch.tensor(sigma_f0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
            final_sigma_g = float(
                (
                    (self.min_sigma_g + torch.exp(log_sigma_g_raw))
                    if self.learn_sigma_g
                    else torch.tensor(sigma_g0, device=device, dtype=dtype)
                )
                .detach()
                .cpu()
            )
        return final_lengthscale, final_outputscale, final_sigma_f, final_sigma_g


def build_training_state(
    split: RegressionData, init_lengthscale: torch.Tensor, m: int
) -> VecchiaTrainingState:
    X_scaled = scale_inputs(split.X_train, init_lengthscale)
    order = maximin_ordering(X_scaled)
    X = split.X_train[order].contiguous()
    y = split.y_train[order].contiguous()
    g = split.g_train[order].contiguous()
    Xs = X_scaled[order].contiguous()

    return VecchiaTrainingState(
        X=X,
        y=y,
        g=g,
        neighbors=predecessor_neighbors(Xs, m),
        sample_positions=torch.arange(len(X)),
    )


def _batch_nll(
    **kwargs,
) -> torch.Tensor:

    from .batched_training import batch_nll

    return batch_nll(**kwargs)


def _batch_nll_sequential(
    *,
    state: VecchiaTrainingState,
    target_positions: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    sigma_f: torch.Tensor,
    sigma_g: torch.Tensor,
    kernel: str,
    gradient_noise_model: str,
    training_mode: Literal["full", "target"] = "full",
    gram_distances: bool = False,
) -> torch.Tensor:
    terms: list[torch.Tensor] = []
    for pos_t in target_positions:
        pos = int(pos_t.detach().item())
        mean, var = _local_observed_y_factor(
            X=state.X,
            y=state.y,
            g=state.g,
            target_pos=pos,
            cond_idx=state.neighbors[pos, : min(pos, state.neighbors.shape[1])],
            lengthscale=lengthscale,
            outputscale=outputscale,
            sigma_f=sigma_f,
            sigma_g=sigma_g,
            kernel=kernel,
            gradient_noise_model=gradient_noise_model,
            training_mode=training_mode,
            gram_distances=gram_distances,
        )
        resid = state.y[pos] - mean
        terms.append(0.5 * (torch.log(var) + resid * resid / var + math.log(2.0 * math.pi)))
    return torch.stack(terms).mean()


def _local_observed_y_factor(
    *,
    X: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    target_pos: int,
    cond_idx: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    sigma_f: torch.Tensor,
    sigma_g: torch.Tensor,
    kernel: str,
    gradient_noise_model: str,
    training_mode: Literal["full", "target"] = "full",
    gram_distances: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    device, dtype = X.device, X.dtype
    x = X[target_pos : target_pos + 1]
    y_target_var = outputscale + sigma_f
    m_local = int(cond_idx.numel())
    if m_local == 0:
        return X.new_zeros(()), y_target_var.clamp_min(torch.finfo(dtype).eps)

    Xc = X[cond_idx]
    yc = y[cond_idx]
    gc = g[cond_idx]
    x_scaled = scale_inputs(x, lengthscale)
    Xc_scaled = scale_inputs(Xc, lengthscale)

    Kff = _func_covariance(Xc, Xc, lengthscale, outputscale, kernel)
    Kff = 0.5 * (Kff + Kff.T)
    Kff = Kff + sigma_f * torch.eye(m_local, device=device, dtype=dtype)

    k_fc = _func_covariance(Xc, x, lengthscale, outputscale, kernel).squeeze(-1)

    delta = (Xc - x).T.contiguous()
    delta_scaled = (Xc_scaled - x_scaled).T.contiguous()
    H = delta_scaled.T @ delta_scaled

    r_i = torch.diagonal(H, 0)
    alpha_i = _alpha_from_r(r_i, kernel, outputscale)

    if gram_distances:
        r_cc = (r_i[:, None] + r_i[None, :] - 2.0 * H).clamp_min(0.0)
    else:
        delta_cc = Xc_scaled[:, None, :] - Xc_scaled[None, :, :]
        r_cc = delta_cc.square().sum(dim=-1)
    alpha_cc = _alpha_from_r(r_cc, kernel, outputscale)
    beta_cc = _beta_from_r(r_cc, kernel, outputscale)

    if training_mode == "target":
        h_diag = r_i
        row = torch.arange(m_local, device=device)

        bar_k = alpha_i * h_diag

        Q = alpha_cc * (h_diag[:, None] - H)

        G0 = alpha_cc * H + beta_cc * (H - h_diag[:, None]) * (h_diag[None, :] - H)
        if bool((sigma_g > 0).detach().cpu().item()):
            delta_sq = delta.square()
            if gradient_noise_model == "iid":
                r_noise_diag = delta_sq.sum(dim=0)
            elif gradient_noise_model == "scaled":
                inv_ell2 = lengthscale.square().reciprocal()
                if inv_ell2.numel() == 1:
                    r_noise_diag = inv_ell2.reshape(()) * delta_sq.sum(dim=0)
                else:
                    r_noise_diag = (delta_sq * inv_ell2.view(-1, 1)).sum(dim=0)
            else:
                raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
            G0[row, row] = G0[row, row] + sigma_g * r_noise_diag

        q_obs = -(gc * delta.T).sum(dim=1).contiguous()

        top = torch.cat([Kff, Q.T], dim=1)
        bottom = torch.cat([Q, G0], dim=1)
        K_joint = torch.cat([top, bottom], dim=0)
        K_joint = 0.5 * (K_joint + K_joint.T)
        L_joint = cholesky_with_jitter(K_joint)

        obs = torch.cat([yc, q_obs], dim=0)
        cross = torch.cat([k_fc, bar_k], dim=0)
        rhs = torch.stack([obs, cross], dim=1)
        sol = torch.cholesky_solve(rhs, L_joint)

        mean = torch.dot(cross, sol[:, 0])
        var = torch.clamp(
            y_target_var - torch.dot(cross, sol[:, 1]),
            min=torch.finfo(dtype).eps,
        )
        return mean, var

    if training_mode != "full":
        raise ValueError(f"Unknown TERA training_mode: {training_mode}")

    cols = H.T.contiguous()
    q = cols[:, None, :] - cols[None, :, :]

    bar_k = ((-alpha_i[:, None]) * cols).reshape(m_local * m_local).contiguous()
    Q = (
        ((-alpha_cc[:, :, None]) * q)
        .permute(0, 2, 1)
        .reshape(m_local * m_local, m_local)
        .contiguous()
    )

    G0_blocks = alpha_cc[:, :, None, None] * H.view(1, 1, m_local, m_local)
    G0_blocks = G0_blocks + beta_cc[:, :, None, None] * (q[:, :, :, None] * q[:, :, None, :])
    if bool((sigma_g > 0).detach().cpu().item()):
        R = _projected_gradient_noise_gram(delta, lengthscale, gradient_noise_model)
        diag_idx = torch.arange(m_local, device=device)
        G0_blocks[diag_idx, diag_idx] = G0_blocks[diag_idx, diag_idx] + sigma_g * R
    G0 = G0_blocks.permute(0, 2, 1, 3).reshape(m_local * m_local, m_local * m_local).contiguous()

    q_obs = (gc @ delta).reshape(-1).contiguous()
    return _direct_joint_scalar_conditional(
        Kff=Kff,
        Q=Q,
        G0=G0,
        y_c=yc,
        q_obs=q_obs,
        k_fc=k_fc,
        bar_k=bar_k,
        prior_var=y_target_var,
    )


def _projected_gradient_noise_gram(
    delta: torch.Tensor, lengthscale: torch.Tensor, gradient_noise_model: str
) -> torch.Tensor:
    if gradient_noise_model == "iid":
        R = delta.T @ delta
    elif gradient_noise_model == "scaled":
        inv_ell2 = lengthscale.square().reciprocal()
        if inv_ell2.numel() == 1:
            delta_noise = delta * inv_ell2.reshape(1, 1)
        else:
            delta_noise = delta * inv_ell2.view(-1, 1)
        R = delta.T @ delta_noise
    else:
        raise ValueError("gradient_noise_model must be either 'iid' or 'scaled'.")
    return 0.5 * (R + R.T)


def _func_covariance(
    X1: torch.Tensor,
    X2: torch.Tensor,
    lengthscale: torch.Tensor,
    outputscale: torch.Tensor,
    kernel_name: str,
) -> torch.Tensor:
    if lengthscale.numel() == 1:
        delta_scaled = (X1[:, None, :] - X2[None, :, :]) / lengthscale.reshape(1, 1, 1)
    else:
        delta_scaled = (X1[:, None, :] - X2[None, :, :]) / lengthscale.view(1, 1, -1)

    if kernel_name == "rbf":
        r = (delta_scaled * delta_scaled).sum(dim=-1)
        return outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        r2 = (delta_scaled * delta_scaled).sum(dim=-1)
        r = torch.sqrt(torch.clamp(r2, min=1e-12))
        a = math.sqrt(5.0) * r
        return outputscale * (1.0 + a + (a * a) / 3.0) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def _alpha_from_r(r: torch.Tensor, kernel_name: str, outputscale: torch.Tensor) -> torch.Tensor:
    if kernel_name == "rbf":
        return outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return (5.0 / 3.0) * outputscale * (1.0 + a) * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")


def _beta_from_r(r: torch.Tensor, kernel_name: str, outputscale: torch.Tensor) -> torch.Tensor:
    if kernel_name == "rbf":
        return -outputscale * torch.exp(-0.5 * r)
    if kernel_name == "matern52":
        a = math.sqrt(5.0) * torch.sqrt(torch.clamp(r, min=1e-12))
        return -(25.0 / 3.0) * outputscale * torch.exp(-a)
    raise ValueError(f"Unknown kernel name: {kernel_name}")
