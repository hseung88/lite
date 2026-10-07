"""Device-resident weighted Lasso and active-set implicit hypergradients.

FISTA identifies candidate supports; only a polished, KKT-certified solution
is returned. No ridge, finite differences, or CPU solver fallback is used.
"""

from __future__ import annotations

import math

import numpy as np
import torch


class TorchLassoCV:
    def __init__(self, folds, *, device, alpha_max, max_iter):
        if max_iter < 1:
            raise ValueError("lassodna_torch_max_iter must be positive.")
        self.max_iter = int(max_iter)
        self.slack = 1e-10 * max(1.0, alpha_max)

        def tensor(x):
            if hasattr(x, "toarray"):
                x = x.toarray()
            return torch.as_tensor(np.asarray(x), device=device, dtype=torch.float64)

        self.gram = torch.stack([tensor(g) for _, _, _, _, g in folds])
        self.cross = torch.stack([tensor(xt.T @ yt / len(yt)) for xt, yt, _, _, _ in folds])
        self.val = [(tensor(xv), tensor(yv)) for _, _, xv, yv, _ in folds]
        # Each fold uses its own Lipschitz constant. The small safety margin
        # prevents a rounded eigenvalue from making the step too large.
        self.lipschitz = torch.linalg.eigvalsh(self.gram)[:, -1:] * (1 + 1e-12)
        if not bool((self.lipschitz > 0).all()):
            raise ValueError("Degenerate Lasso training fold.")
        self.eye = torch.eye(self.gram.shape[-1], device=device, dtype=torch.float64)

    @torch.no_grad()
    def evaluate(self, penalties, *, log_range, with_grad):
        penalties = penalties.to(device=self.gram.device, dtype=torch.float64)
        beta = torch.zeros_like(self.cross)
        extrapolated = beta.clone()
        done = torch.zeros(len(self.gram), device=beta.device, dtype=torch.bool)
        certified = beta.clone()
        factors = self.eye.expand_as(self.gram).clone()
        momentum = 1.0
        for iteration in range(1, self.max_iter + 1):
            grad = (self.gram @ extrapolated.unsqueeze(-1)).squeeze(-1) - self.cross
            proposal = extrapolated - grad / self.lipschitz
            next_beta = proposal.sign() * (proposal.abs() - penalties / self.lipschitz).clamp_min(0)
            next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = next_beta + ((momentum - 1) / next_momentum) * (next_beta - beta)
            beta = torch.where(done[:, None], certified, next_beta)
            extrapolated = torch.where(done[:, None], certified, extrapolated)
            momentum = next_momentum

            if iteration % 50 and iteration != self.max_iter:
                continue

            active = beta != 0
            # Inactive coordinates are padded by identity, not regularized.
            # Thus the active block is exactly the unmodified Lasso Hessian.
            matrix = self.gram * (active[:, :, None] & active[:, None, :])
            matrix = matrix + torch.diag_embed((~active).to(beta.dtype))
            chol, info = torch.linalg.cholesky_ex(matrix, check_errors=False)
            safe_chol = torch.where((info == 0)[:, None, None], chol, self.eye)
            rhs = (self.cross - penalties * beta.sign()) * active
            polished = torch.cholesky_solve(rhs.unsqueeze(-1), safe_chol).squeeze(-1)
            dual = self.cross - (self.gram @ polished.unsqueeze(-1)).squeeze(-1)
            signs_ok = (polished.sign() == beta.sign()).all(-1)
            stationary = torch.where(active, (dual - penalties * beta.sign()).abs(), 0)
            inactive_violation = torch.where(~active, (dual.abs() - penalties).clamp_min(0), 0)
            valid = (
                (info == 0)
                & signs_ok
                & (stationary.amax(-1) <= self.slack)
                & (inactive_violation.amax(-1) <= self.slack)
                & torch.isfinite(polished).all(-1)
            )
            newly_done = valid & ~done
            certified = torch.where(newly_done[:, None], polished, certified)
            factors = torch.where(newly_done[:, None, None], safe_chol, factors)
            done |= valid
            if bool(done.all()):
                break
        else:
            raise RuntimeError(
                "Torch weighted Lasso did not obtain a nonsingular, KKT-certified solution "
                f"in {self.max_iter} FISTA iterations. Increase --lassodna-torch-max-iter "
                "or use --lassodna-backend celer; a nonunique active solution may also be the cause."
            )

        losses, outer = [], []
        for b, (xv, yv) in zip(certified, self.val):
            residual = xv @ b - yv
            losses.append(residual.square().mean())
            if with_grad:
                outer.append(2 * (xv.T @ residual) / len(yv))
        value = torch.stack(losses).mean()
        if not with_grad:
            return value, None
        active = certified != 0
        rhs = torch.stack(outer) * active
        adjoint = torch.cholesky_solve(rhs.unsqueeze(-1), factors).squeeze(-1)
        gradient = (-log_range * penalties * certified.sign() * adjoint).mean(0)
        return value, gradient
