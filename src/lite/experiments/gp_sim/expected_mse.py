from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import torch
import yaml

from lite.experiments.gp_sim.config import (
    METHOD_LITE,
    METHOD_TERA_BATCHED,
    METHOD_VECCHIA_GP,
    ExperimentConfig,
)
from lite.experiments.gp_sim.simulation import _make_design
from lite.methods.common.data import SimulatedDataset
from lite.methods.common.random import RandomStream, seed_for
from lite.methods.common.utils import (
    calibrate_isotropic_lengthscale_from_inputs,
    dtype_from_name,
    resolve_lengthscale,
    scale_inputs,
)
from lite.methods.lite.posterior import nearest_neighbors
from lite.methods.lite.simulation import LITEPredictor
from lite.methods.tera.simulation_batched import BatchedTERASimulationPredictor
from lite.methods.vecchia.simulation_batched import BatchedVecchiaSimulationPredictor


def prediction_batch_size(cfg, method):
    if method == METHOD_VECCHIA_GP:
        return cfg.vecchia_prediction_batch_size or cfg.lite_prediction_batch_size
    if method == METHOD_LITE:
        return cfg.lite_prediction_batch_size
    if method == METHOD_TERA_BATCHED:
        return cfg.tera_prediction_batch_size
    raise ValueError(f"Unsupported expected-MSE method: {method}")


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def design_signature(cfg, d, repeat):
    return dict(
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        d=d,
        repeat=repeat,
        seed=cfg.seed,
        design=cfg.design,
        dtype=cfg.dtype,
        kernel=cfg.kernel,
        use_ard=cfg.use_ard,
        lengthscale=cfg.lengthscale,
        target_median_correlation=cfg.target_median_correlation,
        lengthscale_probe_size=cfg.lengthscale_probe_size,
        max_m=max(cfg.m_values or [cfg.m]),
        knn_query_batch_size=cfg.knn_query_batch_size,
        knn_train_chunk_size=cfg.knn_train_chunk_size,
        version=1,
    )


@torch.no_grad()
def prepare_design(cfg, d, repeat, cache_path):
    device, dtype = torch.device(cfg.device), dtype_from_name(cfg.dtype)
    begin = time.perf_counter()
    X = _make_design(
        cfg.n_train,
        d,
        design=cfg.design,
        seed=seed_for(cfg.seed, RandomStream.GP_TRAIN_INPUTS, repeat, d),
        device=device,
        dtype=dtype,
    )
    Q = _make_design(
        cfg.n_eval,
        d,
        design=cfg.design,
        seed=seed_for(cfg.seed, RandomStream.GP_EVAL_INPUTS, repeat, d),
        device=device,
        dtype=dtype,
    )
    if cfg.target_median_correlation is not None:
        if cfg.use_ard:
            raise ValueError("Automatic median-correlation calibration requires use_ard: false")
        # Never call pdist on the full 100,000-input design.
        probe_n = min(cfg.lengthscale_probe_size, len(X))
        ell = calibrate_isotropic_lengthscale_from_inputs(
            X[:probe_n], cfg.kernel, cfg.target_median_correlation
        )
    else:
        probe_n = 0
        ell = resolve_lengthscale(cfg.lengthscale, d, cfg.use_ard, device=device, dtype=dtype)
    if not torch.isfinite(ell).all() or not (ell > 0).all():
        raise ValueError("Lengthscales must be finite and positive")
    scaled_X, scaled_Q = scale_inputs(X, ell), scale_inputs(Q, ell)
    _sync(device)
    knn_begin = time.perf_counter()
    maximum_m = max(cfg.m_values or [cfg.m])
    ids = torch.cat(
        [
            nearest_neighbors(
                scaled_X,
                scaled_Q[start : start + cfg.knn_query_batch_size],
                maximum_m,
                chunk_size=cfg.knn_train_chunk_size,
            )
            for start in range(0, len(Q), cfg.knn_query_batch_size)
        ]
    )
    _sync(device)
    knn_time = time.perf_counter() - knn_begin
    metadata = dict(
        design_id=hashlib.sha256(
            json.dumps(design_signature(cfg, d, repeat), sort_keys=True).encode()
        ).hexdigest()[:16],
        lengthscale_values=ell.cpu().tolist(),
        lengthscale_probe_count=probe_n,
        neighbor_max_m=maximum_m,
        knn_time_sec=knn_time,
        design_prepare_time_sec=time.perf_counter() - begin,
        input_ordering="none",
        observations="zeros_for_timing_only",
    )
    input_signature = {
        key: value
        for key, value in design_signature(cfg, d, repeat).items()
        if key not in {"max_m", "knn_query_batch_size", "knn_train_chunk_size"}
    }
    metadata["input_design_id"] = hashlib.sha256(
        json.dumps(input_signature, sort_keys=True).encode()
    ).hexdigest()[:16]
    payload = dict(
        X_train=X.cpu(),
        X_eval=Q.cpu(),
        lengthscale=ell.cpu(),
        neighborhoods=ids.cpu(),
        signature=design_signature(cfg, d, repeat),
        metadata=metadata,
    )
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp = cache_path.with_suffix(".tmp")
    torch.save(payload, temp)
    temp.replace(cache_path)
    _write_json(cache_path.with_suffix(".json"), metadata)
    return metadata


def make_timing_data(payload, cfg, *, include_gradients=True):
    device = torch.device(cfg.device)
    X, Q, ell = (
        payload[key].to(device).contiguous() for key in ["X_train", "X_eval", "lengthscale"]
    )
    return SimulatedDataset(
        X_train=X,
        X_train_scaled=scale_inputs(X, ell),
        X_eval=Q,
        X_eval_scaled=scale_inputs(Q, ell),
        lengthscale=ell,
        outputscale=cfg.outputscale,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
        kernel_name=cfg.kernel,
        f_train_obs=X.new_zeros(len(X)),
        g_train_obs=torch.zeros_like(X) if include_gradients else X.new_empty((len(X), 0)),
        z_train_obs=X.new_empty(0),
        sampling_backend="none_expected_mse",
    )


@torch.no_grad()
def measure_method(cfg, cache_path, method, m):
    device = torch.device(cfg.device)
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    data = (
        make_timing_data(payload, cfg, include_gradients=False)
        if method == METHOD_VECCHIA_GP
        else make_timing_data(payload, cfg)
    )
    # Only this m's IDs are resident during the measured method execution.
    ids = payload["neighborhoods"][:, :m].contiguous().to(device)
    neighborhoods = list(ids.unbind(0))
    metadata = payload["metadata"]
    del payload
    batch_size = prediction_batch_size(cfg, method)
    if method == METHOD_LITE:
        predictor = LITEPredictor(m, batch_size, normalize_directions=True)
    elif method == METHOD_VECCHIA_GP:
        predictor = BatchedVecchiaSimulationPredictor(m, prediction_batch_size=batch_size)
    elif method == METHOD_TERA_BATCHED:
        predictor = BatchedTERASimulationPredictor(m, prediction_batch_size=batch_size)
    else:
        raise ValueError("Only Vecchia GP, LITE and batched TERA are supported")
    _sync(device)
    begin = time.perf_counter()
    predictor.build(data)
    _sync(device)
    build_time = time.perf_counter() - begin
    warm_n = min(batch_size, len(data.X_eval))
    for _ in range(cfg.warmup_batches):
        warm = predictor.predict_f_marginals(
            data.X_eval[:warm_n], neighborhoods=neighborhoods[:warm_n]
        )
        del warm
    _sync(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    times, peaks = [], []
    baseline = (
        torch.cuda.memory_allocated(device) / 1024**3 if device.type == "cuda" else float("nan")
    )
    for _ in range(cfg.measurement_repeats):
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        _sync(device)
        begin = time.perf_counter()
        pred = predictor.predict_f_marginals(data.X_eval, neighborhoods=neighborhoods)
        _sync(device)
        times.append(time.perf_counter() - begin)
        peaks.append(
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else float("nan")
        )
        del pred
    # Replay outside timing/peak measurement to evaluate the actual linear estimator.
    begin = time.perf_counter()
    pred = predictor.predict_f_marginals(
        data.X_eval, neighborhoods=neighborhoods, return_expected_mse=True
    )
    _sync(device)
    evaluation_time = time.perf_counter() - begin
    risks = pred.expected_mse
    tolerance = 1024 * torch.finfo(risks.dtype).eps * max(1.0, cfg.outputscale)
    if not torch.isfinite(risks).all() or (risks < -tolerance).any():
        raise ArithmeticError("Nonfinite or significantly negative expected MSE")
    mse = float(risks.clamp_min(0).mean())
    return dict(
        metadata,
        expected_mse=mse,
        root_expected_mse=math.sqrt(mse),
        posterior_variance_mean=float(pred.var.mean()),
        maxabs_risk_variance_gap=float((risks - pred.var).abs().max()),
        minimum_raw_expected_mse=float(risks.min()),
        build_time_sec=build_time,
        predict_time_sec=statistics.median(times),
        wall_time_sec=statistics.median(times),
        peak_mem_gib=max(peaks),
        baseline_mem_gib=baseline,
        incremental_peak_mem_gib=max(peaks) - baseline,
        measurement_times_sec=json.dumps(times),
        evaluation_time_sec=evaluation_time,
        prediction_batch_size=batch_size,
        normalized_directions=method == METHOD_LITE,
    )


def _worker(job):
    cfg = ExperimentConfig(**job["config"])
    try:
        if job["operation"] == "prepare":
            result = prepare_design(cfg, job["d"], job["repeat"], job["cache"])
        else:
            result = measure_method(cfg, job["cache"], job["method"], job["m"])
        result["status"] = "ok"
    except Exception as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or (
            isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()
        )
        result = dict(status="oom" if is_oom else "failed", error=f"{type(exc).__name__}: {exc}")
    _write_json(job["result"], result)


def _launch(job, path, verbose):
    _write_json(path, job)
    env = dict(os.environ)
    source = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = source + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-m", "lite.experiments.gp_sim.expected_mse", "--job", str(path)],
        env=env,
        capture_output=True,
        text=True,
    )
    result_path = Path(job["result"])
    if proc.returncode != 0 or not result_path.exists():
        return dict(
            status="failed", error=f"Worker exited {proc.returncode}: {proc.stderr[-4000:]}"
        )
    result = json.loads(result_path.read_text())
    if verbose and result["status"] != "ok":
        print(result.get("error", "Unknown worker failure"), flush=True)
    return result


def _base_row(cfg, d, repeat, m, method):
    return dict(
        experiment_name=cfg.experiment_name,
        evaluation_mode="expected_mse",
        method=method,
        repeat=repeat,
        seed=cfg.seed,
        n_train=cfg.n_train,
        n_eval=cfg.n_eval,
        d=d,
        m=m,
        kernel=cfg.kernel,
        use_ard=cfg.use_ard,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
        outputscale=cfg.outputscale,
        target_median_correlation=cfg.target_median_correlation,
        dtype=cfg.dtype,
        device=cfg.device,
        timing_scope="conditional_mean_and_variance_prediction",
        memory_unit="GiB",
        memory_measurement="torch_cuda_peak_allocated",
        measurement_repeats=cfg.measurement_repeats,
        warmup_batches=cfg.warmup_batches,
        target="latent_function",
        learned_parameters=False,
        expected_mse=float("nan"),
        root_expected_mse=float("nan"),
        wall_time_sec=float("nan"),
        predict_time_sec=float("nan"),
        peak_mem_gib=float("nan"),
        baseline_mem_gib=float("nan"),
        prediction_batch_size=prediction_batch_size(cfg, method),
        tera_statistic_dimension_ok=(d >= m if method == METHOD_TERA_BATCHED else True),
    )


def run_expected_mse(cfg, outdir, *, verbose=False, cache_dir=None, keep_cache=False):
    if cfg.evaluation_mode != "expected_mse":
        raise ValueError("Expected-MSE runner requires evaluation_mode: expected_mse")
    outdir = Path(outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_resolved.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    jobs = outdir / "jobs"
    jobs.mkdir(exist_ok=True)
    temporary = None
    if cache_dir is None:
        if keep_cache:
            cache_dir = outdir / "cache"
        else:
            temporary = tempfile.TemporaryDirectory(prefix="gp_expected_mse_")
            cache_dir = temporary.name
    cache_dir = Path(cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    m_values = cfg.m_values or [cfg.m]
    total = (
        len(cfg.n_train_values or [cfg.n_train])
        * len(cfg.d_values)
        * cfg.repeats
        * len(m_values)
        * len(cfg.methods)
    )
    try:
        for n in cfg.n_train_values or [cfg.n_train]:
            local = replace(cfg, n_train=n)
            for d in cfg.d_values:
                for repeat in range(cfg.repeats):
                    signature = design_signature(local, d, repeat)
                    design_id = hashlib.sha256(
                        json.dumps(signature, sort_keys=True).encode()
                    ).hexdigest()[:16]
                    cache = cache_dir / f"design_{design_id}.pt"
                    prep_path = jobs / f"prepare_{design_id}.json"
                    if not cache.exists() or not cache.with_suffix(".json").exists():
                        result = _launch(
                            dict(
                                operation="prepare",
                                config=asdict(local),
                                d=d,
                                repeat=repeat,
                                cache=str(cache),
                                result=str(prep_path.with_suffix(".result.json")),
                            ),
                            prep_path,
                            verbose,
                        )
                        if result["status"] != "ok":
                            raise RuntimeError(f"Input preparation failed: {result.get('error')}")
                    metadata = json.loads(cache.with_suffix(".json").read_text())
                    for m in m_values:
                        for method in cfg.methods:
                            name = {
                                METHOD_VECCHIA_GP: "vecchia",
                                METHOD_LITE: "lite",
                                METHOD_TERA_BATCHED: "tera_batched",
                            }[method]
                            tag = f"{design_id}_{name}_m{m}"
                            path = jobs / f"{tag}.json"
                            if verbose:
                                print(
                                    f"[{len(rows) + 1}/{total}] d={d} n={n} repeat={repeat} {name} m={m}",
                                    flush=True,
                                )
                            result = _launch(
                                dict(
                                    operation="measure",
                                    config=asdict(local),
                                    d=d,
                                    repeat=repeat,
                                    cache=str(cache),
                                    m=m,
                                    method=method,
                                    result=str(path.with_suffix(".result.json")),
                                ),
                                path,
                                verbose,
                            )
                            row = dict(_base_row(local, d, repeat, m, method), **metadata)
                            row.update(result)
                            rows.append(row)
                            table = pd.DataFrame(rows)
                            temp = outdir / "results.csv.tmp"
                            table.to_csv(temp, index=False)
                            temp.replace(outdir / "results.csv")
                            if verbose:
                                print(
                                    f"  {row['status']}: root expected MSE={row['root_expected_mse']:.6g} "
                                    f"time={row['wall_time_sec']:.4g}s memory={row['peak_mem_gib']:.4g} GiB",
                                    flush=True,
                                )
    finally:
        if temporary is not None:
            temporary.cleanup()
    return rows


def main():
    parser = argparse.ArgumentParser(description="Internal isolated expected-MSE worker")
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    _worker(json.loads(Path(args.job).read_text()))


if __name__ == "__main__":
    main()
