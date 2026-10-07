from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import KFold, train_test_split


class LassoDNAProblem:
    name = "lassodna"
    kind = "lassodna"
    optimum_value = float("nan")
    active_dim = 180

    def __init__(
        self,
        *,
        device,
        dtype,
        data_path=None,
        data_home=None,
        split_seed=42,
        n_splits=5,
        test_size=0.15,
        tol=1e-8,
        max_iter=1000,
        backend="celer",
        torch_max_iter=20000,
        X=None,
        y=None,
    ):
        if backend not in {"celer", "torch"}:
            raise ValueError("lassodna_backend must be 'celer' or 'torch'.")
        self.backend = backend
        if X is None:
            if data_path:
                path = Path(data_path)
                if path.suffix == ".npz":
                    with np.load(path, allow_pickle=False) as f:
                        X, y = f["X"], f["y"]
                else:
                    from sklearn.datasets import load_svmlight_file

                    X, y = load_svmlight_file(str(path), n_features=180)
            else:
                import os
                import tempfile
                from urllib.request import urlopen

                from sklearn.datasets import load_svmlight_file

                cache = Path(data_home or "data/lassodna").expanduser()
                cache.mkdir(parents=True, exist_ok=True)
                path = cache / "dna.scale"
                if not path.exists():
                    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/dna.scale"

                    with tempfile.NamedTemporaryFile(dir=cache, delete=False) as f:
                        tmp = Path(f.name)
                        try:
                            with urlopen(url, timeout=60) as response:
                                f.write(response.read())
                            f.close()
                            load_svmlight_file(str(tmp), n_features=180)
                            os.replace(tmp, path)
                        finally:
                            tmp.unlink(missing_ok=True)
                X, y = load_svmlight_file(str(path), n_features=180)
        X = (
            sparse.csc_matrix(X, dtype=np.float64)
            if sparse.issparse(X)
            else np.asarray(X, dtype=np.float64, order="F")
        )
        y = np.asarray(y, dtype=np.float64).reshape(-1)
        if X.shape[0] != len(y) or X.shape[1] != 180:
            raise ValueError("LassoDNA requires X with 180 features and matching y.")
        if not np.isfinite(y).all() or not np.isfinite(X.data if sparse.issparse(X) else X).all():
            raise ValueError("LassoDNA data must be finite.")
        if tol <= 0 or max_iter < 1:
            raise ValueError("Lasso solver tolerance and iteration limit must be positive.")
        self.dim = 180
        self.lower = torch.zeros(self.dim, device=device, dtype=dtype)
        self.upper = torch.ones(self.dim, device=device, dtype=dtype)
        self.tol, self.max_iter = float(tol), int(max_iter)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=split_seed
        )
        self.alpha_max = float(np.max(np.abs(self.X_train.T @ self.y_train)) / len(self.y_train))
        if self.alpha_max <= 0:
            raise ValueError("Degenerate Lasso data give zero alpha_max.")
        self.log_alpha_max = np.log(self.alpha_max)
        self.log_alpha_min = np.log(self.alpha_max / 1e5)
        self.log_range = self.log_alpha_max - self.log_alpha_min
        self.folds = []
        self.kf = KFold(n_splits=n_splits, shuffle=True, random_state=split_seed)
        for train, val in self.kf.split(self.X_train):
            xt, xv = self.X_train[train], self.X_train[val]
            yt, yv = self.y_train[train], self.y_train[val]
            gram = xt.T @ xt / len(train)
            if sparse.issparse(gram):
                gram = gram.toarray()
            self.folds.append((xt, yt, xv, yv, np.asarray(gram)))
        if backend == "torch":
            from lite.experiments.bo.lassodna_torch import TorchLassoCV

            self.torch_solver = TorchLassoCV(
                self.folds, device=device, alpha_max=self.alpha_max, max_iter=torch_max_iter
            )

    def value_and_grad(self, unit_x, *, with_grad=True):
        if self.backend == "torch":
            value, gradient = self._torch_value_and_grad(unit_x, with_grad=with_grad)
            return float(value), None if gradient is None else gradient.cpu().numpy()

        from celer import Lasso

        x = np.asarray(unit_x, dtype=np.float64)
        if x.shape != (self.dim,) or not np.isfinite(x).all() or np.any((x < 0) | (x > 1)):
            raise ValueError("LassoDNA input must be a finite 180-vector in [0,1].")
        penalties = np.exp(self.log_alpha_min + self.log_range * x)
        values = []
        gradient = np.zeros(self.dim)
        for xt, yt, xv, yv, gram in self.folds:
            cross = np.asarray(xt.T @ yt).reshape(-1) / len(yt)
            factor = None
            for refinement in range(5):
                estimator = Lasso(
                    alpha=1.0,
                    weights=penalties,
                    fit_intercept=False,
                    warm_start=False,
                    tol=max(self.tol * 0.01**refinement, 1e-14),
                    max_iter=self.max_iter,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    estimator.fit(xt, yt)
                raw = estimator.coef_
                active = np.flatnonzero(raw != 0)
                beta = np.zeros_like(raw)
                if len(active):
                    try:
                        factor = cho_factor(
                            gram[np.ix_(active, active)], lower=True, check_finite=False
                        )
                        beta[active] = cho_solve(
                            factor,
                            cross[active] - penalties[active] * np.sign(raw[active]),
                            check_finite=False,
                        )
                    except np.linalg.LinAlgError as exc:
                        raise RuntimeError(
                            "Lasso active Hessian is singular; a unique active-set solution is unavailable."
                        ) from exc
                inactive = raw == 0
                dual = cross - gram @ beta
                slack = 1e-10 * max(1.0, self.alpha_max)
                if np.array_equal(np.sign(beta[active]), np.sign(raw[active])) and np.all(
                    np.abs(dual[inactive]) <= penalties[inactive] + slack
                ):
                    break
            else:
                raise RuntimeError(
                    "Could not certify the weighted Lasso solution; tighten tolerance or increase max_iter."
                )
            residual = np.asarray(xv @ beta).reshape(-1) - yv
            values.append(float(residual @ residual / len(yv)))
            if with_grad:
                active = np.flatnonzero(beta != 0)
                if len(active):
                    outer = np.asarray(xv[:, active].T @ residual).reshape(-1) * 2 / len(yv)
                    adjoint = cho_solve(factor, outer, check_finite=False)
                    gradient[active] -= (
                        penalties[active] * np.sign(beta[active]) * adjoint * self.log_range
                    )
        return float(np.mean(values)), gradient / len(self.folds) if with_grad else None

    @torch.no_grad()
    def _torch_value_and_grad(self, unit_x, *, with_grad):
        x = torch.as_tensor(unit_x, device=self.lower.device, dtype=torch.float64)
        if x.shape != (self.dim,) or not bool(
            torch.isfinite(x).all() & ((x >= 0) & (x <= 1)).all()
        ):
            raise ValueError("LassoDNA input must be a finite 180-vector in [0,1].")
        penalties = torch.exp(self.log_alpha_min + self.log_range * x)
        return self.torch_solver.evaluate(penalties, log_range=self.log_range, with_grad=with_grad)

    def observe(self, X_unit, *, noise_std=0.0, with_grad=True, generator=None):
        if noise_std != 0:
            raise ValueError("LassoDNA experiments use the deterministic CV objective.")
        shape = X_unit.shape[:-1]
        if self.backend == "torch":
            values, grads = [], []
            for x in X_unit.detach().reshape(-1, self.dim):
                value, grad = self._torch_value_and_grad(x, with_grad=with_grad)
                values.append(-value)
                grads.append(-grad if with_grad else torch.zeros_like(x))
            return (
                torch.stack(values).to(X_unit).reshape(shape),
                torch.stack(grads).to(X_unit).reshape(*shape, self.dim),
            )
        points = X_unit.detach().cpu().double().reshape(-1, self.dim).numpy()
        values, grads = [], []
        for x in points:
            value, grad = self.value_and_grad(x, with_grad=with_grad)

            values.append(-value)
            grads.append(-grad if with_grad else np.zeros(self.dim))
        return (
            X_unit.new_tensor(values).reshape(shape),
            X_unit.new_tensor(np.asarray(grads)).reshape(*shape, self.dim),
        )

    def objective(self, X_unit):
        return self.observe(X_unit, with_grad=False)[0]
