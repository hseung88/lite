from __future__ import annotations

import argparse
from pathlib import Path

from lite.experiments.gp_sim.cli import _apply_overrides
from lite.experiments.gp_sim.cli import build_parser as build_standard_parser
from lite.experiments.gp_sim.config import load_config
from lite.experiments.gp_sim.expected_mse import run_expected_mse


def build_parser() -> argparse.ArgumentParser:
    parser = build_standard_parser()
    parser.description = (
        "Run GP-simulated fidelity experiments with subprocess-isolated "
        "method timing and CUDA memory measurement."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for per-setting data/reference caches. Defaults to a temporary directory.",
    )
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="Keep per-setting data/reference cache files after the run.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    outdir = Path(args.outdir) if args.outdir is not None else Path("outputs") / cfg.experiment_name
    if cfg.evaluation_mode != "expected_mse":
        parser.error("This supplementary package supports evaluation_mode: expected_mse.")
    run_expected_mse(
        cfg, outdir, verbose=args.verbose, cache_dir=args.cache_dir, keep_cache=args.keep_cache
    )
    print(f"Saved expected-MSE results to {outdir}")


if __name__ == "__main__":
    main()
