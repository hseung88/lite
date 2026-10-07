"""Argument registration and resolution shared by BO runs and sweeps."""

import argparse
from dataclasses import fields

from lite.experiments.bo.config import BOConfig, _validate_methods, load_config
from lite.experiments.cli import add_config_arguments
from lite.experiments.options import add_shared_arguments, resolve_cli


def build_parser():
    parser = argparse.ArgumentParser(description="Run gradient-based BO experiments.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", required=True)
    for name in ("methods", "benchmarks", "seeds"):
        parser.add_argument(f"--{name}", help="Comma-separated values; otherwise use the config.")
    parser.add_argument(
        "--query-batch-size",
        "--batch-size",
        dest="batch_size",
        type=int,
        help="Objective evaluations per BO iteration (independent of training batches).",
    )
    for method in ("lite", "tera"):
        parser.add_argument(
            f"--{method}-train-batch-size",
            f"--{method}-batch-size",
            type=int,
            help="Conditional likelihood factors per parameter update.",
        )
    parser.add_argument("--dtype", choices=("float32", "float64"))
    add_config_arguments(parser, BOConfig, names=tuple(f.name for f in fields(BOConfig)))
    add_shared_arguments(parser, BOConfig, "bo")
    parser.add_argument("--append-results", action="store_true")
    parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    return parser


def config_from_args(args):
    cfg = load_config(args.config)
    for name in ("methods", "benchmarks", "seeds"):
        value = getattr(args, name, None)
        if value is not None:
            entries = [x.strip() for x in value.split(",") if x.strip()]
            if name == "methods":
                entries = _validate_methods(entries)
            elif name == "seeds":
                entries = [int(x) for x in entries]
            setattr(cfg, name, list(dict.fromkeys(entries)))
    if not cfg.methods or not cfg.benchmarks or not cfg.seeds:
        raise ValueError("At least one method, benchmark, and seed is required.")
    cfg = resolve_cli(cfg, args, "bo")
    if cfg.n_init < 1 or cfg.batch_size < 1:
        raise ValueError("n_init and query_batch_size must be positive.")
    if cfg.budget < cfg.n_init:
        raise ValueError("The evaluation budget must be at least n_init.")
    return cfg
