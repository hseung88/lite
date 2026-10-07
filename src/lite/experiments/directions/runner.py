import argparse
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
from time import perf_counter

import pandas as pd
import torch
import yaml

from lite.experiments.gp_sim.config import load_config
from lite.experiments.gp_sim.metrics import gaussian_kl_1d
from lite.experiments.gp_sim.simulation import simulate_dataset
from lite.methods.common.random import RandomStream, seed_for
from lite.methods.lite.directional import (
    directional_posterior,
    raw_lite_directions,
)
from lite.methods.tera.simulation import TERAPredictor

METHODS = ("Target", "Target + random", "LITE")
DENSE_SAMPLING_MAX_OBS_DIM = 6000


def sampling_backend(cfg, d, mode="auto"):
    if mode not in {"auto", "dense", "deroos"}:
        raise ValueError(f"Unknown sampling mode: {mode}")
    if mode != "auto":
        return mode
    limit = min(DENSE_SAMPLING_MAX_OBS_DIM, cfg.dense_sampling_max_obs_dim)
    return "dense" if cfg.n_train * (d + 1) <= limit else "deroos"


def clock(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter()


def select_directions(u, v, random, method):
    m = len(u)
    sites = torch.arange(m, device=u.device)
    if method == "Target":
        return u, sites
    if method == "Conditional":
        return u - v, sites
    if method == "Target + random":
        return torch.stack([u, random], 1).flatten(0, 1), sites.repeat_interleave(2)
    if method == "LITE":
        return torch.stack([u, v], 1).flatten(0, 1), sites.repeat_interleave(2)
    raise ValueError(method)


@torch.no_grad()
def run(cfg, outdir, *, rtol=1e-12, sampling="auto", sampling_device=None):
    if not 0 < rtol < 1:
        raise ValueError("Rank tolerance must lie between zero and one.")
    if cfg.kernel != "matern52" or cfg.use_ard:
        raise ValueError("This ablation requires isotropic Matérn-5/2.")
    if any(not math.isfinite(s) or s < 0 for s in (cfg.sigma_f, cfg.sigma_g)):
        raise ValueError("sigma_f and sigma_g must be finite, nonnegative standard deviations.")
    generation_device = cfg.device if sampling_device is None else sampling_device
    if not cfg.m_values or min(cfg.m_values) < 1 or max(cfg.m_values) > cfg.n_train:
        raise ValueError("Require 1 <= each m <= n_train; m is never silently truncated.")
    if min(cfg.d_values) < 1 or cfg.repeats < 1 or cfg.n_eval < 1:
        raise ValueError("Dimensions, repeats and target count must be positive.")
    if cfg.dtype != "float64":
        raise ValueError("Use float64 for this posterior comparison.")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_resolved.yaml").write_text(
        yaml.safe_dump(
            dict(
                asdict(cfg),
                reference_backend="tera",
                reference_scope="function values and full gradients at the same m conditioning inputs",
                sampling=sampling,
                sampling_device=str(generation_device),
                gradient_noise_model="isotropic_original_coordinates",
                prediction_target="latent_function",
                sampling_backends={d: sampling_backend(cfg, d, sampling) for d in cfg.d_values},
                dense_sampling_switch_dim=DENSE_SAMPLING_MAX_OBS_DIM,
                rank_rtol=rtol,
                direction_methods=list(METHODS),
                random_distribution="D_i xi_ij with xi_ij~N(0,I_m), independent per target/neighbor; no QR/SVD or normalization",
                lengthscale_policy="calibrate once on repeat 0 per dimension; fix across m/methods/repeats",
            )
        )
    )
    all_rows = []
    summary = []
    timings = []
    for d in cfg.d_values:
        ell = None
        for repeat in range(cfg.repeats):
            backend = sampling_backend(cfg, d, sampling)
            base_cfg = replace(cfg, sampling=backend, device=generation_device)
            sim_cfg = (
                base_cfg
                if ell is None
                else replace(base_cfg, target_median_correlation=None, lengthscale=ell)
            )
            print(
                f"d={d} repeat={repeat + 1}/{cfg.repeats}: generating GP observations "
                f"(sampling={backend}, n_train={cfg.n_train})",
                flush=True,
            )
            started = clock(cfg.device)
            data = simulate_dataset(sim_cfg, d, repeat=repeat)
            if torch.device(generation_device).type == "cuda":
                torch.cuda.synchronize(generation_device)
            # Transfer only inputs and sampled observations, never the dense covariance.
            data = replace(
                data,
                **{
                    field.name: getattr(data, field.name).to(cfg.device)
                    for field in fields(data)
                    if isinstance(getattr(data, field.name), torch.Tensor)
                },
            )
            simulation_seconds = clock(cfg.device) - started
            ell = float(data.lengthscale.item())

            ids = torch.argsort(torch.cdist(data.X_eval_scaled, data.X_train_scaled), dim=1)[
                :, : max(cfg.m_values)
            ]
            predictor = TERAPredictor(1, rank_rtol=rtol)
            predictor.build(data)
            print(
                f"  observations ready in {simulation_seconds:.2f}s; ell={ell:.6g}",
                flush=True,
            )
            for m in cfg.m_values:
                print(f"  m={m} reference=TERA: computing {cfg.n_eval} targets", flush=True)
                started = clock(cfg.device)
                start = len(all_rows)
                for i, target in enumerate(data.X_eval):
                    idx = ids[i, :m]
                    X = data.X_train[idx]
                    y = data.f_train_obs[idx]
                    g = data.g_train_obs[idx]
                    ref_mean, ref_var = predictor.predict_local(target, idx)
                    u, v = raw_lite_directions(
                        target,
                        X,
                        lengthscale=data.lengthscale,
                        outputscale=cfg.outputscale,
                        kernel=cfg.kernel,
                        sigma_f=cfg.sigma_f,
                    )

                    random_seed = seed_for(cfg.seed, RandomStream.DIRECTIONS, repeat, d, i)
                    gen = torch.Generator(device=X.device).manual_seed(random_seed)
                    coeff = torch.randn(
                        (max(cfg.m_values), max(cfg.m_values)),
                        device=X.device,
                        dtype=X.dtype,
                        generator=gen,
                    )[:m, :m]

                    random = coeff @ (X - target)
                    for method in METHODS:
                        directions, sites = select_directions(u, v, random, method)
                        mean, var, rank = directional_posterior(
                            target,
                            X,
                            y,
                            g,
                            directions,
                            sites,
                            lengthscale=data.lengthscale,
                            outputscale=cfg.outputscale,
                            kernel=cfg.kernel,
                            rtol=rtol,
                            sigma_f=cfg.sigma_f,
                            sigma_g=cfg.sigma_g,
                        )
                        kl = gaussian_kl_1d(ref_mean, ref_var, mean, var)
                        if not torch.isfinite(kl) or kl < -1e-9:
                            raise RuntimeError("Invalid marginal KL.")
                        all_rows.append(
                            dict(
                                d=d,
                                m=m,
                                repeat=repeat,
                                seed=cfg.seed,
                                target=i,
                                method=method,
                                marginal_kl=max(0.0, float(kl)),
                                mean=float(mean),
                                variance=float(var),
                                reference_mean=float(ref_mean),
                                reference_variance=float(ref_var),
                                lengthscale=ell,
                                reference_backend="tera",
                                sampling_backend=data.sampling_backend,
                                noise_model="isotropic_original_coordinates",
                                sigma_f=cfg.sigma_f,
                                sigma_g=cfg.sigma_g,
                                random_seed=random_seed,
                                effective_rank=rank,
                                observation_dim=m + len(sites),
                                status="ok",
                            )
                        )
                prediction_seconds = clock(cfg.device) - started
                timings.append(
                    dict(
                        d=d,
                        m=m,
                        repeat=repeat,
                        simulation_seconds=simulation_seconds,
                        prediction_seconds=prediction_seconds,
                        reference_backend="tera",
                        sampling_backend=data.sampling_backend,
                    )
                )
                part = pd.DataFrame(all_rows[start:])
                for method in METHODS:
                    summary.append(
                        dict(
                            d=d,
                            m=m,
                            repeat=repeat,
                            method=method,
                            sigma_f=cfg.sigma_f,
                            sigma_g=cfg.sigma_g,
                            mean_marginal_kl=float(
                                part.loc[part.method == method, "marginal_kl"].mean()
                            ),
                            n_eval=cfg.n_eval,
                            lengthscale=ell,
                            reference_backend="tera",
                            sampling_backend=data.sampling_backend,
                            status="ok",
                        )
                    )
                pd.DataFrame(all_rows).to_csv(out / "target_results.csv", index=False)
                pd.DataFrame(summary).to_csv(out / "results.csv", index=False)
                pd.DataFrame(timings).to_csv(out / "timings.csv", index=False)
                print(f"  m={m} complete in {prediction_seconds:.2f}s", flush=True)
    return pd.DataFrame(summary)


def main():
    p = argparse.ArgumentParser(
        description="Paired direction-selection experiment using the existing GP simulation."
    )
    p.add_argument("--config", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--d", "--dimension", "--d-values", dest="d_values", type=int, nargs="+")
    p.add_argument("--m", "--m-values", dest="m_values", type=int, nargs="+")
    p.add_argument("--repeats", type=int)
    p.add_argument("--n-train", type=int)
    p.add_argument("--n-eval", type=int)
    p.add_argument("--target-median-correlation", type=float)
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--sampling", choices=["auto", "dense", "deroos"], default="auto")
    p.add_argument(
        "--sampling-device",
        default=None,
        help="Device for GP observation generation; defaults to --device.",
    )
    p.add_argument("--sigma-f", type=float, help="Function observation noise standard deviation.")
    p.add_argument(
        "--sigma-g",
        type=float,
        help="Gradient noise standard deviation per original-coordinate component.",
    )
    p.add_argument("--dense-sampling-max-obs-dim", type=int)
    p.add_argument("--rank-rtol", type=float, default=1e-12)
    a = p.parse_args()
    cfg = load_config(a.config)
    for field in (
        "d_values",
        "m_values",
        "repeats",
        "n_train",
        "n_eval",
        "target_median_correlation",
        "seed",
        "device",
        "sigma_f",
        "sigma_g",
        "dense_sampling_max_obs_dim",
    ):
        value = getattr(a, field)
        if value is not None:
            setattr(cfg, field, value)
    run(cfg, a.outdir, rtol=a.rank_rtol, sampling=a.sampling, sampling_device=a.sampling_device)


if __name__ == "__main__":
    main()
