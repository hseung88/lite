from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd

from lite.experiments.md22.ablation import _positive_csv
from lite.experiments.md22.runner import build_parser as main_parser
from lite.experiments.md22.runner import config_from_args


def largest_fitting_batch(evaluate, cap, *, start=32):
    if (cap is not None and cap < 1) or start < 1:
        raise ValueError("Batch bounds must be positive.")

    def fits(batch):
        result = evaluate(batch)
        if result["status"] not in {"ok", "oom"}:
            raise RuntimeError(
                result.get("error", "Batch probe failed for a reason other than OOM.")
            )
        return result["status"] == "ok"

    first = min(start, cap) if cap is not None else start
    low = 0
    if fits(first):
        low = first
        while cap is None or low < cap:
            candidate = min(2 * low, cap) if cap is not None else 2 * low
            if not fits(candidate):
                high = candidate
                break
            low = candidate
        else:
            return low
    else:
        high = first
    while high - low > 1:
        candidate = (low + high) // 2
        if fits(candidate):
            low = candidate
        else:
            high = candidate
    return low


def _worker(
    config_path,
    cache_path,
    job_dir,
    *,
    mode,
    m,
    method="lite",
    batch_size=32,
    warmup_steps=2,
    measurements=5,
    profile_dir=None,
    allow_repeat_targets=False,
):
    job_dir.mkdir(parents=True, exist_ok=True)
    result_path = job_dir / "result.json"
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable,
        "-u",
        "-m",
        "lite.experiments.md22.step_worker",
        "--mode",
        mode,
        "--config",
        str(config_path.resolve()),
        "--cache",
        str(cache_path.resolve()),
        "--result",
        str(result_path.resolve()),
        "--m",
        str(m),
        "--method",
        method,
        "--batch-size",
        str(batch_size),
        "--warmup-steps",
        str(warmup_steps),
        "--measurements",
        str(measurements),
    ]
    if profile_dir is not None:
        command += ["--profile-dir", str(profile_dir.resolve())]
    if allow_repeat_targets:
        command.append("--allow-repeat-targets")
    env = os.environ.copy()
    src = str(Path(__file__).resolve().parents[3])
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [src, env.get("PYTHONPATH", "")]))
    with (job_dir / "run.log").open("w") as log:
        process = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT, env=env, check=False
        )
    if process.returncode != 0 or not result_path.exists():
        return dict(
            status="error",
            error=f"Worker exited with {process.returncode}; see {job_dir / 'run.log'}",
        )
    result = json.loads(result_path.read_text())
    result["log_path"] = str(job_dir / "run.log")
    return result


def build_parser():
    parser = main_parser()
    parser.description = "One forward/backward training step: fixed and largest fitting batches."
    parser.add_argument(
        "--m-values", type=_positive_csv, default="10,20,30,40,50,60,80,100,150,200"
    )
    parser.add_argument("--batch-modes", choices=["both", "fixed", "max"], default="both")
    parser.add_argument("--fixed-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-cap", type=int)
    parser.add_argument(
        "--allow-repeat-targets",
        action="store_true",
        help="For GPU capacity tests, allow batches larger than the eligible target pool.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=5)
    parser.add_argument("--profile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--profile-m", type=int, default=30)
    parser.add_argument("--profile-batch-size", type=int, default=32)
    parser.add_argument("--profile-steps", type=int, default=3)
    return parser


def run_step_scaling(cfg, args):
    if len(cfg.datasets) != 1:
        raise ValueError("Select one MD22 dataset for this experiment.")
    if not cfg.methods or any(
        method not in {"vecchia", "lite", "tera_batched"} for method in cfg.methods
    ):
        raise ValueError(
            "Use --methods vecchia,tera_batched,lite. Sequential tera is deliberately rejected."
        )
    if not cfg.seeds:
        raise ValueError("At least one seed is required.")
    if (
        min(
            args.fixed_batch_size,
            args.measurements,
            args.profile_batch_size,
            args.profile_steps,
            args.profile_m,
        )
        < 1
        or args.warmup_steps < 0
    ):
        raise ValueError("Invalid batch, repeat, profiler, or warmup settings.")
    if args.max_batch_cap is not None and args.max_batch_cap < 1:
        raise ValueError("--max-batch-cap must be positive.")
    if (
        cfg.device == "cpu"
        and args.batch_modes in {"both", "max"}
        and args.allow_repeat_targets
        and args.max_batch_cap is None
    ):
        raise ValueError(
            "Unbounded GPU-capacity search requires CUDA; set --max-batch-cap for CPU checks."
        )
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    settings = dict(
        config=asdict(cfg),
        benchmark_options={
            k: v
            for k, v in vars(args).items()
            if k
            in {
                "m_values",
                "fixed_batch_size",
                "max_batch_cap",
                "warmup_steps",
                "measurements",
                "profile",
                "profile_m",
                "profile_batch_size",
                "profile_steps",
                "batch_modes",
                "allow_repeat_targets",
            }
        },
    )
    (out / "settings.json").write_text(json.dumps(settings, indent=2))
    rows, probes, profiles = [], [], []
    max_m = max(args.m_values + ([args.profile_m] if args.profile else []))

    def save():
        if rows:
            pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
        if probes:
            pd.DataFrame(probes).to_csv(out / "batch_search.csv", index=False)
        if profiles:
            pd.DataFrame(profiles).to_csv(out / "profiles.csv", index=False)

    for seed in cfg.seeds:
        seed_dir = out / "jobs" / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_cfg = replace(cfg, seeds=[seed], m=max_m, lite_m=None, tera_m=None, vecchia_m=None)
        config_path = seed_dir / "config.json"
        config_path.write_text(json.dumps(asdict(seed_cfg)))
        cache_path = seed_dir / "training_state.pt"
        meta = _worker(config_path, cache_path, seed_dir / "prepare", mode="prepare", m=max_m)
        if meta["status"] != "ok":
            raise RuntimeError(
                f"Training-state preparation failed: {meta.get('error', '')}; {meta['log_path']}"
            )
        for m in args.m_values:
            for method in cfg.methods:
                base = {k: v for k, v in meta.items() if k not in {"status", "log_path"}}
                base.update(
                    method=method,
                    m=m,
                    device=cfg.device,
                    gradient_noise_model=(
                        "none"
                        if method == "vecchia"
                        else getattr(
                            cfg, f"{'lite' if method == 'lite' else 'tera'}_gradient_noise_model"
                        )
                    ),
                    timing_scope="forward_backward",
                    parameters="initial_fixed",
                    optimizer_included=False,
                    graph_build_included=False,
                )
                job = seed_dir / method / f"m{m}"
                cached = {}

                def measure(batch, *, warmup, count, label):
                    return _worker(
                        config_path,
                        cache_path,
                        job / f"{label}_B{batch}",
                        mode="measure",
                        m=m,
                        method=method,
                        batch_size=batch,
                        warmup_steps=warmup,
                        measurements=count,
                        allow_repeat_targets=args.allow_repeat_targets,
                    )

                def probe(batch):
                    if batch not in cached:
                        result = measure(batch, warmup=1, count=1, label="probe")
                        cached[batch] = result
                        probes.append(dict(base, batch_mode="probe", batch_size=batch, **result))
                        save()
                        if cfg.verbose:
                            print(
                                f"  probe {method} m={m} B={batch}: {result['status']}", flush=True
                            )
                    return cached[batch]

                if args.batch_modes in {"both", "fixed"}:
                    fixed = measure(
                        args.fixed_batch_size,
                        warmup=args.warmup_steps,
                        count=args.measurements,
                        label="fixed",
                    )
                    cached[args.fixed_batch_size] = fixed
                    rows.append(
                        dict(
                            base,
                            batch_mode="fixed",
                            batch_size=args.fixed_batch_size,
                            targets_repeated=args.fixed_batch_size > meta["eligible_targets"],
                            **fixed,
                        )
                    )
                    save()
                    if cfg.verbose:
                        print(
                            f"{method} m={m} B={args.fixed_batch_size}: {fixed['status']} "
                            f"step={fixed.get('step_time_sec')} s peak={fixed.get('peak_mem_gib')} GiB",
                            flush=True,
                        )
                if args.batch_modes in {"both", "max"}:
                    cap = (
                        args.max_batch_cap
                        if args.allow_repeat_targets
                        else min(
                            meta["eligible_targets"], args.max_batch_cap or meta["eligible_targets"]
                        )
                    )
                    result, batch = dict(status="oom", error="No batch fits, including B=1."), 0
                    try:
                        search_cap = cap
                        while search_cap is None or search_cap > 0:
                            batch = largest_fitting_batch(
                                probe, search_cap, start=args.fixed_batch_size
                            )
                            if batch == 0:
                                break
                            result = measure(
                                batch,
                                warmup=args.warmup_steps,
                                count=args.measurements,
                                label="max",
                            )
                            if result["status"] != "oom":
                                break
                            cached[batch] = result
                            probes.append(
                                dict(base, batch_mode="validation_oom", batch_size=batch, **result)
                            )
                            search_cap = batch - 1
                            if search_cap == 0:
                                batch = 0
                        if batch == 0:
                            result = dict(status="oom", error="No batch fits, including B=1.")
                    except RuntimeError as exc:
                        result = dict(status="error", error=str(exc))
                    upper_oom = min(
                        (b for b, r in cached.items() if r["status"] == "oom" and b > batch),
                        default=None,
                    )
                    rows.append(
                        dict(
                            base,
                            batch_mode="max",
                            batch_size=batch,
                            max_batch_bound=cap,
                            max_batch_at_cap=(cap is not None and batch == cap),
                            targets_repeated=batch > meta["eligible_targets"],
                            next_oom_batch=upper_oom,
                            **result,
                        )
                    )
                    save()
                    if cfg.verbose:
                        print(
                            f"{method} m={m} largest B={batch}: {result['status']} "
                            f"step={result.get('step_time_sec')} s peak={result.get('peak_mem_gib')} GiB",
                            flush=True,
                        )
        if args.profile:
            profile_dir = (
                out / "profile" / f"lite_m{args.profile_m}_B{args.profile_batch_size}_seed{seed}"
            )
            result = _worker(
                config_path,
                cache_path,
                seed_dir / "profile",
                mode="profile",
                m=args.profile_m,
                method="lite",
                batch_size=args.profile_batch_size,
                warmup_steps=args.warmup_steps,
                measurements=args.profile_steps,
                profile_dir=profile_dir,
                allow_repeat_targets=args.allow_repeat_targets,
            )
            profiles.append(
                dict(
                    seed=seed,
                    method="lite",
                    m=args.profile_m,
                    batch_size=args.profile_batch_size,
                    **result,
                )
            )
            save()
        cache_path.unlink(missing_ok=True)
    return rows


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = config_from_args(args)
    if args.datasets is None:
        cfg.datasets = ["buckyball-catcher"]
    if args.methods is None:
        cfg.methods = ["tera_batched", "lite"]
    run_step_scaling(cfg, args)
    print(f"Saved step scaling results to {args.outdir}")


if __name__ == "__main__":
    main()
