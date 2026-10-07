from __future__ import annotations

import contextlib
import os
from types import SimpleNamespace

import torch
from torch.utils.data import Dataset

from lite.experiments.md22.config import resolve_dataset_config
from lite.experiments.md22.data import baseline_targets
from lite.methods.common.prediction import Prediction


class GradientDataset(Dataset):
    def __init__(self, X, y, g, *, dtype):

        self.X = X.detach().to(device="cpu", dtype=dtype)
        self.y = y.detach().to(device="cpu", dtype=dtype).reshape(-1)
        self.g = g.detach().to(device="cpu", dtype=dtype)
        self.dim = self.X.shape[1]
        self.scale, self.shift = 1.0, 0.0

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], {"energy": self.y[idx], "neg_force": self.g[idx]}


def collate_gradients(batch):
    return (
        torch.stack([x for x, _ in batch]),
        torch.stack([torch.cat([y["energy"].reshape(1), y["neg_force"]]) for _, y in batch]),
    )


class DerivativeBaselineModel:
    def __init__(self, *, name, config, seed):
        self.name, self.cfg, self.seed = name, config, seed
        self.training_history = []
        self.model = None
        self.likelihood = None

    def fit(self, split):
        from lite.experiments.md22.models.baseline_config import build_config

        prefix = "dsoftki" if self.name == "dsoftki" else "ddsvgp"
        cfg = resolve_dataset_config(self.cfg, split.d)
        self.kernel = "rbf" if self.name == "ddsvgp" else cfg.kernel
        self.device = split.X_train.device
        self.dtype = getattr(torch, getattr(cfg, prefix + "_dtype"))
        if self.name == "dsoftki" and self.dtype != torch.float32:
            raise ValueError("DSoftKI currently requires float32.")
        self.prediction_batch_size = getattr(cfg, prefix + "_prediction_batch_size")
        self.num_directions = min(cfg.ddsvgp_num_directions, split.d)
        if self.num_directions < 1 or self.prediction_batch_size < 1:
            raise ValueError("Directions and batch sizes must be positive.")
        y_train, g_train, y_test, g_test, target_scale = baseline_targets(
            split, joint_scale=cfg.baseline_joint_scale
        )
        self.target_scale = float(target_scale)
        self.scale = self.target_scale / float(split.scaler.energy_std)
        train = GradientDataset(split.X_train, y_train, g_train, dtype=self.dtype)
        test = (
            GradientDataset(split.X_test, y_test, g_test, dtype=self.dtype)
            if cfg.log_training_curves
            else None
        )
        noise = getattr(cfg, prefix + "_noise")
        noise = cfg.sigma_f if noise is None else noise
        lengthscale = cfg.lengthscale
        if lengthscale is None or isinstance(lengthscale, list):
            lengthscale = 1.0
        args = SimpleNamespace(
            method=self.name,
            num_workers=cfg.baseline_num_workers,
            kernel=cfg.kernel,
            lengthscale=lengthscale,
            use_ard=cfg.dsoftki_use_ard,
            num_inducing=min(getattr(cfg, prefix + "_num_inducing"), len(train)),
            noise=noise,
            deriv_noise=cfg.dsoftki_deriv_noise
            if cfg.dsoftki_deriv_noise is not None
            else noise * split.d,
            learn_noise=cfg.dsoftki_learn_noise,
            solver="cg",
            cg_tolerance=cfg.dsoftki_cg_tolerance,
            mll_approx="hutchinson_fallback",
            fit_chunk_size=cfg.dsoftki_fit_chunk_size,
            use_qr=True,
            dtype=getattr(cfg, prefix + "_dtype"),
            device=str(self.device),
            fit_device=str(self.device),
            seed=self.seed,
            batch_size=getattr(cfg, prefix + "_batch_size"),
            epochs=getattr(cfg, prefix + "_train_epochs"),
            lr=getattr(cfg, prefix + "_lr"),
            embed_lr=cfg.dsoftki_lr,
            weight_decay=0.0,
            curve_log_every=cfg.curve_log_every,
            log_training_curves=cfg.log_training_curves,
            num_directions=self.num_directions,
            mll_type=cfg.ddsvgp_mll_type,
            gamma=0.1,
        )
        if args.epochs < 1 or args.num_inducing < 1:
            raise ValueError("Baseline epochs and inducing point count must be positive.")
        metadata = SimpleNamespace(dataset=split.name)
        model_cfg = build_config(args, metadata)
        self.resolved_config = model_cfg
        old_dtype = torch.get_default_dtype()
        try:
            with (
                open(os.devnull, "w") as stream,
                contextlib.nullcontext() if cfg.verbose else contextlib.redirect_stdout(stream),
            ):
                if self.name == "dsoftki":
                    from lite.methods.dsoftki.train import train_gp

                    self.model = train_gp(model_cfg, train, test, collate_fn=collate_gradients)
                else:
                    from lite.methods.ddsvgp.train import train_gp

                    self.model, self.likelihood = train_gp(
                        model_cfg, train, test, collate_fn=collate_gradients
                    )
        finally:
            torch.set_default_dtype(old_dtype)
        self.training_history = []
        for rec in getattr(self.model, "training_history", []):
            rmse = float(rec["normalized_energy_rmse_per_atom"]) * self.scale / split.n_atoms
            self.training_history.append(
                dict(
                    step=rec["step"],
                    normalized_energy_rmse_per_atom=rmse,
                    raw_energy_rmse_per_atom=rmse * float(split.scaler.energy_std),
                )
            )
        self.model.eval()
        if self.likelihood is not None:
            self.likelihood.eval()

    @torch.no_grad()
    def predict(self, X):
        if self.model is None:
            raise RuntimeError("fit() must precede predict().")
        means, variances = [], []
        if self.name == "ddsvgp":
            directions = torch.zeros(
                self.num_directions, X.shape[1], device=self.device, dtype=self.dtype
            )
            directions.diagonal().fill_(1)
        for chunk in X.split(self.prediction_batch_size):
            xb = chunk.to(device=self.device, dtype=self.dtype)
            if self.name == "dsoftki":
                mean = self.model.pred(xb)[: len(xb)]
            else:
                out = self.model(xb, derivative_directions=directions.repeat(len(xb), 1))
                mean = out.mean[:: self.num_directions + 1]
                variances.append(out.variance[:: self.num_directions + 1].to(X) * self.scale**2)
            means.append(mean.to(X) * self.scale)
        if not means:
            return Prediction(X.new_empty(0), X.new_empty(0) if self.name == "ddsvgp" else None)
        return Prediction(torch.cat(means), torch.cat(variances) if variances else None)
