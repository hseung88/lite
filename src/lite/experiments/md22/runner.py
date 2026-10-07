from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import torch

from lite.experiments.md22.config import (
    MD22Config,
    load_config,
    resolve_dataset_config,
    resolve_method_config,
)
from lite.experiments.md22.data import add_observation_noise, load_md22_raw, make_split
from lite.experiments.md22.metrics import normalized_energy_rmse_per_atom, raw_energy_rmse_per_atom
from lite.experiments.md22.models import (
    BatchedTERAModel,
    DDSVGPModel,
    DSoftKIModel,
    LITEModel,
    StandardGPModel,
    TERAModel,
)
from lite.experiments.options import add_shared_arguments, resolve_cli
from lite.methods.common.random import RandomStream, seed_for
from lite.methods.common.utils import dtype_from_name
from lite.methods.names import normalize_md22_method_name
from lite.methods.vecchia.model import VecchiaGPModel


@dataclass(slots=True)
class MD22ResultRow:
    experiment_name: str
    dataset: str
    seed: int
    method: str
    n_train: int
    n_test: int
    d: int
    n_atoms: int
    split_id: str
    preprocessing_version: str
    x_scale: float
    m: int
    prediction_batch_size: int
    kernel: str
    normalized_energy_rmse_per_atom: float
    raw_energy_rmse_per_atom: float
    fit_time_sec: float
    predict_time_sec: float
    wall_time_sec: float
    peak_mem_gb: float
    status: str
    fit_peak_mem_gb: float = math.nan
    predict_peak_mem_gb: float = math.nan
    normalized_rmse_scale: str = "train_energy_std"
    model_target_scale: float = math.nan
    model_dtype: str = ""
    device: str = ""
    observation_noise_fraction: float = 0.0
    energy_noise_fraction: float = 0.0
    force_noise_fraction: float = 0.0
    observation_noise_id: str = "clean"
    energy_noise_std: float = 0.0
    force_noise_std: float = 0.0
    added_function_noise_var: float = 0.0
    added_gradient_noise_var: float = 0.0


def observation_settings(split) -> dict:
    return {
        name: getattr(split, name)
        for name in (
            "observation_noise_fraction",
            "energy_noise_fraction",
            "force_noise_fraction",
            "observation_noise_id",
            "energy_noise_std",
            "force_noise_std",
            "added_function_noise_var",
            "added_gradient_noise_var",
        )
    }


def config_with_observation_noise(cfg, split):
    if (
        not cfg.initialize_noise_from_observations
        or split.energy_noise_fraction == split.force_noise_fraction == 0
    ):
        return cfg
    return replace(
        cfg,
        sigma_f=cfg.sigma_f + split.added_function_noise_var,
        sigma_g=cfg.sigma_g + split.added_gradient_noise_var,
    )


def _clear_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)


def _peak_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    torch.cuda.synchronize(device)
    return float(torch.cuda.max_memory_allocated(device)) / (1024.0**3)


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. Use --device cpu explicitly."
            )
        index = torch.cuda.current_device() if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} does not exist.")
        device = torch.device("cuda", index)
    return device


def method_settings(cfg: MD22Config, split, method: str, seed: int, device) -> dict:
    """Record effective options for each dataset/seed/method before fitting."""
    method = normalize_md22_method_name(method)
    cfg = resolve_method_config(cfg, method)
    prefix = "tera" if method == "tera_batched" else method
    settings = {
        key[len(prefix) + 1 :]: value
        for key, value in asdict(cfg).items()
        if key.startswith(prefix + "_")
    }
    settings.update(
        dataset=split.name,
        method=method,
        seed=seed,
        split_id=split.split_id,
        preprocessing_version=split.preprocessing_version,
        device=str(device),
        d=split.d,
        n_train=len(split.X_train),
        n_test=len(split.X_test),
        kernel=cfg.kernel,
        dtype=getattr(cfg, prefix + "_dtype", cfg.dtype),
        use_ard=cfg.use_ard,
        lengthscale=cfg.lengthscale,
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
        x_scale=cfg.x_scale,
        energy_mean=float(split.scaler.energy_mean),
        energy_std=float(split.scaler.energy_std),
        normalized_rmse_scale="train_energy_std",
        log_training_curves=cfg.log_training_curves,
        initialize_noise_from_observations=cfg.initialize_noise_from_observations,
        **observation_settings(split),
    )
    if method == "standard_gp":
        settings["lr"] = _standard_gp_lr_for_dataset(cfg, split.name)
    else:
        settings["training_seed"] = seed_for(seed, RandomStream.TRAINING)
    if method in {"tera", "tera_batched", "lite", "vecchia"}:
        settings.update(m=cfg.m, outputscale=cfg.outputscale)
    if method == "vecchia":
        settings.update(factor_evaluation="batched", uses_gradients=False)
        settings.pop("sigma_g")
    if method in {"tera", "tera_batched"}:
        settings["factor_evaluation"] = "sequential" if method == "tera" else "batched"
        settings["prediction_batch_size"] = (
            1 if method == "tera" else cfg.tera_prediction_batch_size
        )
    if method in {"lite"}:
        settings["normalize_directions"] = True
    if method in {"dsoftki", "ddsvgp"}:
        settings.update(
            num_inducing=min(getattr(cfg, prefix + "_num_inducing"), len(split.X_train)),
            initialization="kmeans",
            kernel="rbf" if method == "ddsvgp" else cfg.kernel,
            use_ard=False if method == "ddsvgp" else cfg.dsoftki_use_ard,
            lengthscale=cfg.lengthscale if isinstance(cfg.lengthscale, (int, float)) else 1.0,
            target_normalization="joint_value_derivative_train_scale"
            if cfg.baseline_joint_scale
            else "train_energy_std",
            normalization_dtype="float64",
            num_workers=cfg.baseline_num_workers,
            optimizer="Adam",
            weight_decay=0.0,
        )
        # These baselines use their own likelihood noise fields, recorded above.
        settings.pop("sigma_f")
        settings.pop("sigma_g")
        if method == "ddsvgp":
            settings.update(
                num_directions=min(cfg.ddsvgp_num_directions, split.d),
                internal_solve_dtype="float64",
                likelihood_input="latent",
            )
        else:
            settings.update(
                solver="cg",
                mll_approx="hutchinson_fallback",
                use_qr=True,
                fit_device=str(device),
                float64_numerical_fallback=True,
            )
    return settings


def _standard_gp_lr_for_dataset(cfg: MD22Config, dataset_name: str) -> float:
    if cfg.standard_gp_lr is not None:
        return float(cfg.standard_gp_lr)
    return float(cfg.standard_gp_lr_by_dataset.get(dataset_name, cfg.standard_gp_lr_default))


def _make_model(method: str, cfg: MD22Config, seed: int, *, dataset_name: str):
    method = normalize_md22_method_name(method)
    cfg = resolve_method_config(cfg, method)
    seed = seed_for(seed, RandomStream.TRAINING)
    if method == "standard_gp":
        model = StandardGPModel(
            kernel=cfg.kernel,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            lengthscale=cfg.lengthscale,
            lengthscale_init=cfg.lengthscale_init,
            lengthscale_init_max_points=cfg.lengthscale_init_max_points,
            use_ard=cfg.use_ard,
            max_train=cfg.standard_gp_max_train,
            train_steps=cfg.standard_gp_train_steps,
            train_epochs=cfg.standard_gp_train_epochs,
            lr=_standard_gp_lr_for_dataset(cfg, dataset_name),
            weight_decay=cfg.standard_gp_weight_decay,
            learn_lengthscale=cfg.standard_gp_learn_lengthscale,
            learn_outputscale=cfg.standard_gp_learn_outputscale,
            learn_sigma_f=cfg.standard_gp_learn_sigma_f,
            min_sigma_f=cfg.standard_gp_min_sigma_f,
            log_every=cfg.standard_gp_log_every,
        )
    elif method in {"tera", "tera_batched", "lite", "vecchia"}:
        prefix = method if method in {"lite", "vecchia"} else "tera"
        training_fields = (
            "train_steps",
            "train_epochs",
            "graph_refresh_epochs",
            "train_batch_size",
            "lr",
            "weight_decay",
            "learn_lengthscale",
            "learn_outputscale",
            "learn_sigma_f",
            "learn_sigma_g",
            "min_sigma_f",
            "min_sigma_g",
            "log_every",
            "gradient_noise_model",
        )
        options = {
            name: getattr(cfg, f"{prefix}_{name}")
            for name in training_fields
            if hasattr(cfg, f"{prefix}_{name}")
        }
        options["prediction_batch_size"] = (
            1 if method == "tera" else getattr(cfg, f"{prefix}_prediction_batch_size")
        )
        model_class = {
            "tera": TERAModel,
            "tera_batched": BatchedTERAModel,
            "lite": LITEModel,
            "vecchia": VecchiaGPModel,
        }[method]
        if method == "lite":
            options["normalize_directions"] = True
        model = model_class(
            m=cfg.m,
            kernel=cfg.kernel,
            outputscale=cfg.outputscale,
            sigma_f=cfg.sigma_f,
            sigma_g=cfg.sigma_g,
            lengthscale=cfg.lengthscale,
            lengthscale_init=cfg.lengthscale_init,
            lengthscale_init_max_points=cfg.lengthscale_init_max_points,
            use_ard=cfg.use_ard,
            seed=seed,
            **options,
        )
    elif method == "dsoftki":
        model = DSoftKIModel(config=cfg, seed=seed)
    elif method == "ddsvgp":
        model = DDSVGPModel(config=cfg, seed=seed)
    else:
        raise ValueError(f"Unknown MD22 method: {method}")
    model.name = method
    model.validation_enabled = cfg.log_training_curves
    return model


def _run_one_method(
    cfg: MD22Config,
    split,
    method: str,
    seed: int,
    device: torch.device,
    history: list | None = None,
    on_fit=None,
) -> MD22ResultRow:
    cfg = resolve_method_config(cfg, method)
    model = _make_model(method, cfg, seed, dataset_name=split.name)
    try:
        _clear_cuda(device)
        t0 = time.perf_counter()
        model.fit(split)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        fit_time = time.perf_counter() - t0
        fit_peak = _peak_gb(device)
        if on_fit is not None:
            on_fit(model, cfg, seed, split)
        if history is not None:
            for rec in getattr(model, "training_history", []):
                history.append(
                    dict(
                        rec,
                        dataset=split.name,
                        method=method,
                        seed=seed,
                        m=cfg.m,
                        split_id=split.split_id,
                        status="ok",
                        **observation_settings(split),
                    )
                )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        t1 = time.perf_counter()
        with torch.no_grad():
            pred = model.predict(split.X_test)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            predict_time = time.perf_counter() - t1
            predict_peak = _peak_gb(device)
            if not bool(torch.isfinite(pred.y_mean).all()):
                raise FloatingPointError("Non-finite predictive mean.")
            if pred.y_var is not None and not bool(torch.isfinite(pred.y_var).all()):
                raise FloatingPointError("Non-finite predictive variance.")
            norm_rmse = normalized_energy_rmse_per_atom(pred.y_mean, split.y_test, split.n_atoms)
            raw_rmse = raw_energy_rmse_per_atom(
                pred.y_mean,
                split.E_test,
                energy_mean=split.scaler.energy_mean,
                energy_std=split.scaler.energy_std,
                n_atoms=split.n_atoms,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        peak = max(fit_peak, predict_peak)
        return MD22ResultRow(
            experiment_name=cfg.experiment_name,
            dataset=split.name,
            seed=seed,
            method=method,
            n_train=int(split.X_train.shape[0]),
            n_test=int(split.X_test.shape[0]),
            d=split.d,
            n_atoms=split.n_atoms,
            split_id=split.split_id,
            preprocessing_version=split.preprocessing_version,
            x_scale=float(split.scaler.x_scale),
            m=cfg.m,
            prediction_batch_size=getattr(
                model,
                "prediction_batch_size",
                cfg.lite_prediction_batch_size if method in {"lite"} else 1,
            ),
            kernel=getattr(model, "kernel", cfg.kernel),
            normalized_energy_rmse_per_atom=norm_rmse,
            raw_energy_rmse_per_atom=raw_rmse,
            fit_time_sec=fit_time,
            predict_time_sec=predict_time,
            wall_time_sec=fit_time + predict_time,
            peak_mem_gb=peak,
            fit_peak_mem_gb=fit_peak,
            predict_peak_mem_gb=predict_peak,
            model_target_scale=getattr(model, "target_scale", float(split.scaler.energy_std)),
            model_dtype=getattr(
                cfg, {"dsoftki": "dsoftki_dtype", "ddsvgp": "ddsvgp_dtype"}.get(method, "dtype")
            ),
            device=str(device),
            status="ok",
            **observation_settings(split),
        )
    except Exception as exc:
        return _failed_row(cfg, split, method, seed, str(exc))


def _failed_row(cfg: MD22Config, split, method: str, seed: int, status: str) -> MD22ResultRow:
    cfg = resolve_method_config(cfg, method)
    return MD22ResultRow(
        experiment_name=cfg.experiment_name,
        dataset=split.name,
        seed=seed,
        method=method,
        n_train=int(split.X_train.shape[0]),
        n_test=int(split.X_test.shape[0]),
        d=split.d,
        n_atoms=split.n_atoms,
        split_id=split.split_id,
        preprocessing_version=split.preprocessing_version,
        x_scale=float(split.scaler.x_scale),
        m=cfg.m,
        prediction_batch_size=(
            cfg.tera_prediction_batch_size
            if method == "tera_batched"
            else cfg.lite_prediction_batch_size
            if method == "lite"
            else 1
        ),
        kernel=cfg.kernel,
        normalized_energy_rmse_per_atom=math.nan,
        raw_energy_rmse_per_atom=math.nan,
        fit_time_sec=math.nan,
        predict_time_sec=math.nan,
        wall_time_sec=math.nan,
        peak_mem_gb=math.nan,
        status=f"failed: {status}",
        **observation_settings(split),
    )


def run(cfg: MD22Config, outdir: str | Path, *, on_fit=None) -> list[MD22ResultRow]:
    cfg = replace(cfg, methods=list(dict.fromkeys(map(normalize_md22_method_name, cfg.methods))))
    device = resolve_device(cfg.device)
    dtype = dtype_from_name(cfg.dtype)
    rows: list[MD22ResultRow] = []
    history: list[dict] = []
    import yaml

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "config_resolved.yaml").write_text(yaml.safe_dump(asdict(cfg), sort_keys=False))
    (outdir / "method_configs.jsonl").write_text("")
    dataset_configs = {}

    for dataset in cfg.datasets:
        raw = load_md22_raw(cfg.data_dir, dataset, device=device, dtype=dtype)
        dataset_cfg = resolve_dataset_config(cfg, raw.d)
        dataset_configs[dataset] = asdict(dataset_cfg)
        (outdir / "dataset_configs_resolved.yaml").write_text(
            yaml.safe_dump(dataset_configs, sort_keys=False)
        )
        for seed in cfg.seeds:
            split = make_split(
                raw,
                seed=seed,
                train_frac=cfg.train_frac,
                test_frac=cfg.test_frac,
                n_train=cfg.n_train,
                n_test=cfg.n_test,
                x_scale=cfg.x_scale,
                preprocessing_version=cfg.preprocessing_version,
            )
            split = add_observation_noise(
                split,
                cfg.observation_noise_fraction,
                energy_fraction=cfg.energy_noise_fraction,
                force_fraction=cfg.force_noise_fraction,
                seed=seed if cfg.observation_noise_seed is None else cfg.observation_noise_seed,
            )
            run_cfg = config_with_observation_noise(dataset_cfg, split)
            for method in cfg.methods:
                method_cfg = resolve_method_config(run_cfg, method)
                with (outdir / "method_configs.jsonl").open("a") as log:
                    log.write(
                        json.dumps(method_settings(method_cfg, split, method, seed, device)) + "\n"
                    )
                if cfg.verbose:
                    print(f"  running {method} ...", flush=True)
                row = _run_one_method(
                    method_cfg,
                    split,
                    method,
                    seed,
                    device,
                    history=history if cfg.log_training_curves else None,
                    on_fit=on_fit,
                )
                rows.append(row)
                if history:
                    pd.DataFrame(history).to_csv(outdir / "training_curves.csv", index=False)
                if cfg.verbose:
                    print(
                        f"  {method}: normalized_rmse={row.normalized_energy_rmse_per_atom} raw_rmse={row.raw_energy_rmse_per_atom} time={row.wall_time_sec} mem={row.peak_mem_gb} status={row.status}"
                    )
                pd.DataFrame([asdict(r) for r in rows]).to_csv(
                    outdir / cfg.out_csv_name, index=False
                )
    return rows


def _parse_seeds(value: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integer seeds, e.g. 1,2,3.") from exc
    if any(seed < 0 or seed >= 2**64 for seed in seeds):
        raise argparse.ArgumentTypeError("Seeds must be between 0 and 2**64 - 1.")
    return seeds


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run MD22 GP regression with observed gradients.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--outdir", type=str, required=True)
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated: standard_gp,dsoftki,ddsvgp,tera,tera_batched,lite,vecchia.",
    )
    p.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset override.")
    p.add_argument(
        "--seeds", type=_parse_seeds, default=None, help="Comma-separated seed override."
    )
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--n-test", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--kernel", type=str, default=None, choices=["rbf", "matern52"])
    p.add_argument("--m", type=int, default=None)
    p.add_argument(
        "--sigma-f",
        "--sigma_f",
        dest="sigma_f",
        type=float,
        default=None,
        help="Function/value noise variance.",
    )
    p.add_argument(
        "--sigma-g",
        "--sigma_g",
        dest="sigma_g",
        type=float,
        default=None,
        help="Gradient noise variance.",
    )
    p.add_argument(
        "--lengthscale", type=float, default=None, help="Explicit isotropic lengthscale override."
    )
    p.add_argument("--lengthscale-init", type=str, default=None, choices=["median", "one"])
    p.add_argument(
        "--learn-lengthscale", dest="learn_lengthscale", action="store_true", default=None
    )
    p.add_argument("--no-learn-lengthscale", dest="learn_lengthscale", action="store_false")
    p.add_argument("--use-ard", dest="use_ard", action="store_true", default=None)
    p.add_argument("--no-use-ard", dest="use_ard", action="store_false")
    p.add_argument(
        "--standard-gp-max-train",
        "--exact-max-train",
        type=int,
        default=None,
        help="Maximum n_train allowed for dense Standard GP before skipping.",
    )
    p.add_argument("--standard-gp-train-epochs", "--exact-train-epochs", type=int, default=None)
    p.add_argument("--standard-gp-train-steps", "--exact-train-steps", type=int, default=None)
    p.add_argument(
        "--standard-gp-lr",
        "--exact-lr",
        "--exact_lr",
        "--standard_gp_lr",
        dest="standard_gp_lr",
        type=float,
        default=None,
    )
    p.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=None)
    for method in ("tera", "lite"):
        p.add_argument(
            f"--{method}-train-batch-size",
            f"--{method}-batch-size",
            type=int,
            help="Conditional likelihood factors per parameter update.",
        )
        p.add_argument(f"--{method}-gradient-noise-model", choices=("iid", "scaled"))
    p.add_argument(
        "--lite-prediction-batch-size",
        type=int,
        help="Prediction targets processed together at fitted parameters.",
    )
    p.add_argument("--log-training-curves", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--curve-log-every", type=int, default=None)
    for prefix in ("dsoftki", "ddsvgp"):
        for field in ("train-epochs", "num-inducing", "batch-size", "prediction-batch-size"):
            p.add_argument(f"--{prefix}-{field}", type=int, default=None)
        p.add_argument(f"--{prefix}-lr", type=float, default=None)
    from lite.experiments.cli import add_config_arguments

    add_config_arguments(
        p,
        MD22Config,
        prefixes=("standard_gp_", "lite_", "tera_", "vecchia_", "dsoftki_", "ddsvgp_", "baseline_"),
        names=(
            "data_dir",
            "dtype",
            "outputscale",
            "observation_noise_fraction",
            "energy_noise_fraction",
            "force_noise_fraction",
            "observation_noise_seed",
            "initialize_noise_from_observations",
        ),
    )
    add_shared_arguments(p, MD22Config, "md22")
    return p


def _parse_args(argv=None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def config_from_args(args) -> MD22Config:
    """Resolve config and overrides identically for main and ablation runs."""
    cfg = load_config(args.config)
    if args.methods is not None:
        cfg.methods = [
            normalize_md22_method_name(m.strip()) for m in args.methods.split(",") if m.strip()
        ]
    if args.datasets is not None:
        cfg.datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    if args.seeds is not None:
        cfg.seeds = args.seeds
    if not cfg.methods or not cfg.datasets or not cfg.seeds:
        raise ValueError("At least one method, dataset, and seed is required.")
    cfg = resolve_cli(cfg, args, "md22")
    for prefix in ("standard_gp", "tera", "lite", "vecchia"):
        if (
            cfg.log_training_curves
            and getattr(args, f"{prefix}_log_every", None) is None
            and getattr(args, "log_every", None) is None
        ):
            setattr(cfg, f"{prefix}_log_every", cfg.curve_log_every)
    return cfg


def main() -> None:
    args = _parse_args()
    cfg = config_from_args(args)
    rows = run(cfg, args.outdir)
    print(f"Wrote {len(rows)} rows to {Path(args.outdir) / cfg.out_csv_name}")


if __name__ == "__main__":
    main()
