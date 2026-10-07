import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from lite.experiments.symmetry.config import SymmetryConfig, load_config
from lite.methods.common.covariance import condition_scalar
from lite.methods.common.data import SimulatedDataset
from lite.methods.common.ordering import maximin_ordering, predecessor_neighbors
from lite.methods.common.random import RandomStream, seed_for
from lite.methods.common.utils import tau_from_target_median_correlation
from lite.methods.lite.directional import directional_blocks, raw_lite_directions
from lite.methods.lite.posterior import _radial
from lite.methods.tera.simulation import TERAPredictor


def compound_symmetry_error(H):
    """Frobenius projection onto {a I + b 11^T}, including diagonal errors."""
    m = H.shape[0]
    if H.shape != (m, m) or m < 2 or not torch.isfinite(H).all():
        raise ValueError("H must be a finite square Gram matrix of order >= 2.")
    norm = torch.linalg.matrix_norm(H)
    if norm <= 0:
        raise ValueError("Relative CS error is undefined for a zero Gram matrix.")
    diagonal = H.diagonal().sum()
    b = (H.sum() - diagonal) / (m * (m - 1))
    a = diagonal / m - b
    fitted = a * torch.eye(m, dtype=H.dtype, device=H.device) + b
    return torch.linalg.matrix_norm(H - fitted) / norm, a, b


def covariance_data(X, cfg):
    # Inputs and observed gradients are both in scaled coordinates. Zero
    # observations are placeholders: posterior variances do not depend on data.
    y = X.new_zeros(len(X))
    g = torch.zeros_like(X)
    return SimulatedDataset(
        X_train=X,
        X_train_scaled=X,
        X_eval=X[:0],
        X_eval_scaled=X[:0],
        lengthscale=X.new_ones(1),
        outputscale=cfg.outputscale,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
        kernel_name=cfg.kernel,
        f_train_obs=y,
        g_train_obs=g,
        z_train_obs=torch.cat([y[:, None], g], dim=1).flatten(),
        sampling_backend="covariance_only",
    )


@torch.no_grad()
def factor_metrics(target, neighbors, cfg, *, reference=None, indices=None):
    """Paired metrics for a single set; target is noisy y_i, not latent f_i."""
    m = len(neighbors)
    delta = neighbors - target
    H = delta @ delta.T
    cs_error, a, b = compound_symmetry_error(H)
    if reference is None:
        reference = TERAPredictor(m, rank_rtol=cfg.rank_rtol)
        reference.build(covariance_data(neighbors, cfg))
        indices = torch.arange(m, device=neighbors.device)
    _, latent_full = reference.predict_local(target, indices)

    u, v = raw_lite_directions(
        target,
        neighbors,
        lengthscale=1.0,
        outputscale=cfg.outputscale,
        kernel=cfg.kernel,
        sigma_f=cfg.sigma_f,
    )
    directions = torch.stack([u, v], dim=1).flatten(0, 1)
    sites = torch.arange(m, device=neighbors.device).repeat_interleave(2)
    K, cross = directional_blocks(
        target,
        neighbors,
        directions,
        sites,
        lengthscale=1.0,
        outputscale=cfg.outputscale,
        kernel=cfg.kernel,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
    )
    _, latent_lite, rank = condition_scalar(
        K,
        cross,
        cross.new_zeros(len(cross)),
        cfg.outputscale,
        rtol=cfg.rank_rtol,
    )
    Kff = K[:m, :m]
    _, latent_function_only, _ = condition_scalar(
        Kff,
        cross[:m],
        cross.new_zeros(m),
        cfg.outputscale,
        rtol=cfg.rank_rtol,
    )
    # The manuscript conditions a noisy function observation y_i. Its own noise
    # is independent of all conditioning observations, hence adds to both sides.
    full = latent_full + cfg.sigma_f**2
    lite = latent_lite + cfg.sigma_f**2
    function_only = latent_function_only + cfg.sigma_f**2
    delta_var = lite - full
    tol = 100 * cfg.rank_rtol * max(1.0, cfg.outputscale)
    values = torch.stack([full, lite, function_only, cs_error])
    if not torch.isfinite(values).all() or min(float(full), float(lite)) <= 0:
        raise RuntimeError("Nonfinite metrics or nonpositive predictive variance.")
    if float(delta_var) < -tol or float(lite - function_only) > tol:
        raise RuntimeError("Conditional variances violate full <= LITE <= function-only.")
    # Only roundoff-sized negative gaps are clipped; save the signed raw gap.
    kl = 0.5 * torch.log1p(delta_var.clamp_min(0) / full)
    correlations, _, _ = _radial(H.diagonal(), cfg.outputscale, cfg.kernel)
    return dict(
        cs_error=float(cs_error),
        cs_a=float(a),
        cs_b=float(b),
        variance_full=float(full),
        variance_lite=float(lite),
        variance_function_only=float(function_only),
        variance_gap_raw=float(delta_var),
        expected_kl=float(kl),
        full_gradient_gain=float(function_only - full),
        lite_gradient_gain=float(function_only - lite),
        lite_effective_rank=rank,
        median_target_correlation=float((correlations / cfg.outputscale).median()),
    )


def target_positions(cfg, repeat, *, stream=RandomStream.DERIVATIVE_SUBSET):
    # Same ordering positions across d, separate random stream from the inputs.
    seed = seed_for(cfg.seed, stream, repeat)
    gen = torch.Generator().manual_seed(seed)
    positions = torch.randperm(cfg.n_train - cfg.m, generator=gen)[: cfg.n_targets] + cfg.m
    return positions.sort().values, seed


def ordered_design(cfg, d, seed):
    gen = torch.Generator(device=cfg.device).manual_seed(seed)
    X = torch.rand((cfg.n_train, d), generator=gen, device=cfg.device, dtype=torch.float64)
    # A scalar lengthscale leaves the maximin order and neighbor identities
    # invariant, so choose the graph before calibrating that scalar.
    order = maximin_ordering(X)
    X = X[order]
    return X, order, predecessor_neighbors(X, cfg.m)


def calibrate_lengthscale(cfg, d):
    # Independent probe, excluded from all plotted repeats; one ell per d.
    seed = seed_for(cfg.seed, RandomStream.GP_EVAL_INPUTS, d)
    X, _, neighbors = ordered_design(cfg, d, seed)
    positions, position_seed = target_positions(cfg, 0, stream=RandomStream.INITIALIZATION)
    positions = positions.to(X.device)
    delta = X[neighbors[positions]] - X[positions, None]
    median_distance = delta.square().sum(-1).sqrt().median()
    tau = tau_from_target_median_correlation(cfg.kernel, cfg.target_median_correlation)
    ell = float(median_distance / tau)
    if not np.isfinite(ell) or ell <= 0:
        raise RuntimeError("Invalid calibrated lengthscale.")
    return ell, seed, position_seed


@torch.no_grad()
def run(cfg: SymmetryConfig, outdir):
    cfg.validate()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    sets = out / "conditioning_sets"
    sets.mkdir(exist_ok=True)
    metadata = dict(
        **asdict(cfg),
        design="iid_uniform_unit_cube",
        ordering="maximin",
        neighbors="nearest_m_predecessors",
        prediction_target="noisy_y_i",
        gradient_noise_model="isotropic_scaled_coordinates",
        sampling="covariance_only_no_observation_draws",
        reference="TERA orthonormal QR reduction, exact full-gradient local posterior",
        lengthscale_policy="one independent calibration probe per d; fixed across repeats",
        kl_orientation="expected KL(full || LITE)",
        uncertainty="SEM across independent repeat means, not across local factors",
        torch_version=str(torch.__version__),
        calibration={},
    )
    (out / "config_resolved.yaml").write_text(yaml.safe_dump(metadata))
    all_rows, summaries = [], []
    for d in cfg.d_values:
        print(f"d={d}: calibrating on an independent maximin + KNN probe", flush=True)
        ell, probe_seed, position_seed = calibrate_lengthscale(cfg, d)
        metadata["calibration"][d] = dict(
            lengthscale=ell,
            input_seed=probe_seed,
            target_seed=position_seed,
        )
        (out / "config_resolved.yaml").write_text(yaml.safe_dump(metadata))
        for repeat in range(cfg.repeats):
            input_seed = seed_for(cfg.seed, RandomStream.GP_TRAIN_INPUTS, repeat, d)
            X, order, neighbors = ordered_design(cfg, d, input_seed)
            X = X / ell
            positions, target_seed = target_positions(cfg, repeat)
            positions = positions.to(X.device)
            selected_neighbors = neighbors[positions]
            np.savez_compressed(
                sets / f"d{d}_repeat{repeat}.npz",
                order=order.cpu().numpy(),
                target_positions=positions.cpu().numpy(),
                neighbor_positions=selected_neighbors.cpu().numpy(),
                target_original_ids=order[positions].cpu().numpy(),
                neighbor_original_ids=order[selected_neighbors].cpu().numpy(),
            )
            reference = TERAPredictor(cfg.m, rank_rtol=cfg.rank_rtol)
            reference.build(covariance_data(X, cfg))
            print(
                f"  repeat {repeat + 1}/{cfg.repeats}: {cfg.n_targets} factors; ell={ell:.6g}",
                flush=True,
            )
            start = len(all_rows)
            for position, idx in zip(positions, selected_neighbors):
                metrics = factor_metrics(X[position], X[idx], cfg, reference=reference, indices=idx)
                all_rows.append(
                    dict(
                        d=d,
                        m=cfg.m,
                        repeat=repeat,
                        seed=cfg.seed,
                        input_seed=input_seed,
                        target_seed=target_seed,
                        target_position=int(position),
                        target_original_id=int(order[position]),
                        lengthscale=ell,
                        kernel=cfg.kernel,
                        outputscale=cfg.outputscale,
                        sigma_f=cfg.sigma_f,
                        sigma_g=cfg.sigma_g,
                        status="ok",
                        **metrics,
                    )
                )
            part = pd.DataFrame(all_rows[start:])
            summaries.append(
                dict(
                    d=d,
                    m=cfg.m,
                    repeat=repeat,
                    n_targets=len(part),
                    lengthscale=ell,
                    seed=cfg.seed,
                    input_seed=input_seed,
                    target_seed=target_seed,
                    kernel=cfg.kernel,
                    outputscale=cfg.outputscale,
                    sigma_f=cfg.sigma_f,
                    sigma_g=cfg.sigma_g,
                    mean_cs_error=float(part.cs_error.mean()),
                    mean_expected_kl=float(part.expected_kl.mean()),
                    mean_full_gradient_gain=float(part.full_gradient_gain.mean()),
                    mean_lite_gradient_gain=float(part.lite_gradient_gain.mean()),
                    median_target_correlation=float(part.median_target_correlation.median()),
                    status="ok",
                )
            )
            pd.DataFrame(all_rows).to_csv(out / "target_results.csv", index=False)
            pd.DataFrame(summaries).to_csv(out / "results.csv", index=False)
            print(
                f"    CS error={summaries[-1]['mean_cs_error']:.6g}; "
                f"expected KL={summaries[-1]['mean_expected_kl']:.6g}",
                flush=True,
            )
    return pd.DataFrame(summaries)


def main():
    p = argparse.ArgumentParser(
        description="Compound symmetry and expected KL on training Vecchia factors."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--d", "--d-values", dest="d_values", nargs="+", type=int)
    for name in ("m", "n-train", "n-targets", "repeats", "seed"):
        p.add_argument(f"--{name}", type=int)
    for name in ("sigma-f", "sigma-g", "outputscale", "target-median-correlation", "rank-rtol"):
        p.add_argument(f"--{name}", type=float)
    p.add_argument("--device")
    p.add_argument("--num-threads", type=int, default=1, help="CPU intra-op threads (default: 1).")
    args = p.parse_args()
    cfg = load_config(args.config)
    for name in asdict(cfg):
        value = getattr(args, name, None)
        if value is not None:
            setattr(cfg, name, value)
    if args.num_threads < 1:
        p.error("--num-threads must be positive")
    torch.set_num_threads(args.num_threads)
    run(cfg, args.outdir)


if __name__ == "__main__":
    main()
