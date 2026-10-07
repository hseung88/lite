from __future__ import annotations

import argparse
from dataclasses import replace

from lite.experiments.gp_sim.config import (
    METHOD_ORDER,
    ExperimentConfig,
    validate_methods,
)


def _parse_int_list(spec: str) -> list[int]:
    spec = spec.strip()
    if spec == "":
        raise ValueError("Expected a non-empty integer list specification.")
    if ":" in spec:
        parts = spec.split(":")
        if len(parts) != 3:
            raise ValueError(f"Invalid sweep spec: {spec}")
        start, stop, step = (int(p) for p in parts)
        if step == 0:
            raise ValueError("Step cannot be zero.")
        return list(range(start, stop + (1 if step > 0 else -1), step))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def _parse_m(spec: str) -> int | str:
    txt = spec.strip().lower()
    if txt == "adaptive":
        return "adaptive"
    return int(spec)


def _parse_float_or_list(spec: str) -> float | list[float] | None:
    txt = spec.strip().lower()
    if txt in {"none", "null"}:
        return None
    vals = [float(x) for x in spec.split(",") if x.strip() != ""]
    if len(vals) == 1:
        return vals[0]
    return vals


def _parse_methods(spec: str) -> list[str]:
    methods = [x.strip() for x in spec.split(",") if x.strip() != ""]
    if not methods:
        raise ValueError("Expected at least one method name.")
    return methods


def _apply_overrides(cfg: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    updates: dict[str, object] = {}
    for field in [
        "evaluation_mode",
        "knn_query_batch_size",
        "knn_train_chunk_size",
        "lengthscale_probe_size",
        "measurement_repeats",
        "warmup_batches",
        "lite_prediction_batch_size",
        "tera_prediction_batch_size",
        "vecchia_prediction_batch_size",
        "sample_eval",
        "sampling_device",
        "sampling_chunk_size",
        "experiment_name",
        "kernel",
        "outputscale",
        "sigma_f",
        "sigma_g",
        "n_train",
        "n_eval",
        "m",
        "repeats",
        "seed",
        "device",
        "dtype",
        "target_median_correlation",
        "design",
        "sampling",
        "dense_sampling_max_obs_dim",
        "deroos_sampling_max_n2",
        "standard_dgp_dense_max_d",
        "standard_dgp_dense_max_obs_dim",
        "standard_dgp_deroos_max_n",
        "vecchia_dense_max_local_dim",
    ]:
        val = getattr(args, field, None)
        if val is not None:
            if field == "m" and isinstance(val, str):
                val = _parse_m(val)
            updates[field] = val

    if args.use_ard is not None:
        updates["use_ard"] = args.use_ard
    if args.lengthscale is not None:
        updates["lengthscale"] = _parse_float_or_list(args.lengthscale)
        if args.target_median_correlation is None:
            updates["target_median_correlation"] = None
    if args.d_values is not None:
        updates["d_values"] = _parse_int_list(args.d_values)
    if args.n_train_values is not None:
        updates["n_train_values"] = _parse_int_list(args.n_train_values)
    if args.m_values is not None:
        updates["m_values"] = _parse_int_list(args.m_values)
    if args.methods is not None:
        updates["methods"] = validate_methods(_parse_methods(args.methods))

    return replace(cfg, **updates) if updates else cfg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run GP-simulated marginal predictive fidelity experiments."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="Output directory. Defaults to ./outputs/<experiment_name>.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress during the sweep.")
    parser.add_argument("--evaluation-mode", choices=["standard", "expected_mse"], default=None)
    parser.add_argument("--knn-query-batch-size", type=int, default=None)
    parser.add_argument("--knn-train-chunk-size", type=int, default=None)
    parser.add_argument("--lengthscale-probe-size", type=int, default=None)
    parser.add_argument("--measurement-repeats", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=None)

    parser.add_argument("--lite-prediction-batch-size", type=int, default=None)
    parser.add_argument("--tera-prediction-batch-size", type=int, default=None)
    parser.add_argument(
        "--vecchia-prediction-batch-size",
        type=int,
        default=None,
        help="Expected-MSE Vecchia batch size; defaults to the LITE batch size.",
    )
    parser.add_argument(
        "--sample-eval",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Jointly sample latent test values for RMSE (dense sampling only).",
    )
    parser.add_argument(
        "--sampling-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Optional device used only for GP generation, outside method timing.",
    )
    parser.add_argument("--sampling-chunk-size", type=int, default=None)
    parser.add_argument("--experiment-name", dest="experiment_name", type=str, default=None)
    parser.add_argument("--kernel", choices=["rbf", "matern52"], default=None)
    parser.add_argument(
        "--use-ard", dest="use_ard", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--lengthscale", type=str, default=None, help="Single float, comma list, or 'none'."
    )
    parser.add_argument(
        "--target-median-correlation", dest="target_median_correlation", type=float, default=None
    )
    parser.add_argument("--outputscale", type=float, default=None)
    parser.add_argument(
        "--sigma-f",
        dest="sigma_f",
        type=float,
        default=None,
        help="Function observation noise standard deviation.",
    )
    parser.add_argument(
        "--sigma-g",
        dest="sigma_g",
        type=float,
        default=None,
        help="IID coordinate gradient noise standard deviation.",
    )
    parser.add_argument("--n-train", dest="n_train", type=int, default=None)
    parser.add_argument("--n-train-values", dest="n_train_values", type=str, default=None)
    parser.add_argument("--n-eval", dest="n_eval", type=int, default=None)
    parser.add_argument("--m", type=str, default=None)
    parser.add_argument("--m-values", dest="m_values", type=str, default=None)
    parser.add_argument("--d-values", dest="d_values", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--dtype", choices=["float32", "float64"], default=None)
    parser.add_argument("--design", choices=["sobol", "random"], default=None)
    parser.add_argument("--sampling", choices=["dense", "deroos", "zeros"], default=None)
    parser.add_argument(
        "--dense-sampling-max-obs-dim", dest="dense_sampling_max_obs_dim", type=int, default=None
    )
    parser.add_argument(
        "--deroos-sampling-max-n2", dest="deroos_sampling_max_n2", type=int, default=None
    )
    parser.add_argument(
        "--standard-dgp-dense-max-d",
        "--exact-dense-max-d",
        dest="standard_dgp_dense_max_d",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--standard-dgp-dense-max-obs-dim",
        "--exact-dense-max-obs-dim",
        dest="standard_dgp_dense_max_obs_dim",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--standard-dgp-deroos-max-n",
        "--exact-deroos-max-n",
        dest="standard_dgp_deroos_max_n",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--vecchia-dense-max-local-dim", dest="vecchia_dense_max_local_dim", type=int, default=None
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated subset of: "
        + ", ".join(METHOD_ORDER)
        + ". Quote this argument when method names contain spaces.",
    )
    return parser
