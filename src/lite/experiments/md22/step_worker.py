from __future__ import annotations

import argparse
import gc
import json
import math
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lite.experiments.md22.config import MD22Config
from lite.experiments.md22.data import load_md22_raw, make_split
from lite.experiments.md22.runner import _make_model
from lite.methods.common.initialization import resolve_kernel_lengthscale
from lite.methods.common.profiling import step_scope
from lite.methods.common.utils import dtype_from_name
from lite.methods.tera.model import VecchiaTrainingState, build_training_state

STAGES = ("gather_gram", "assembly", "cholesky", "backward", "other")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def prepare_cache(cfg, cache_path, max_m):
    device = torch.device(cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    raw = load_md22_raw(
        cfg.data_dir, cfg.datasets[0], device=device, dtype=dtype_from_name(cfg.dtype)
    )
    split = make_split(
        raw,
        seed=cfg.seeds[0],
        train_frac=cfg.train_frac,
        test_frac=cfg.test_frac,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
        x_scale=cfg.x_scale,
        preprocessing_version=cfg.preprocessing_version,
    )
    if max_m >= len(split.X_train):
        raise ValueError("Largest m must be smaller than the training sample count.")
    ell = resolve_kernel_lengthscale(
        split.X_train,
        lengthscale=cfg.lengthscale,
        lengthscale_init=cfg.lengthscale_init,
        lengthscale_init_max_points=cfg.lengthscale_init_max_points,
        use_ard=cfg.use_ard,
    )
    state = build_training_state(split, ell, max_m)
    gen = torch.Generator().manual_seed(cfg.seeds[0])
    eligible = torch.arange(max_m, len(state.X))
    targets = eligible[torch.randperm(len(eligible), generator=gen)]
    meta = dict(
        dataset=split.name,
        seed=cfg.seeds[0],
        split_id=split.split_id,
        n_train=len(state.X),
        d=split.d,
        max_m=max_m,
        eligible_targets=len(targets),
        kernel=cfg.kernel,
        use_ard=cfg.use_ard,
        dtype=cfg.dtype,
        preprocessing_version=cfg.preprocessing_version,
        lengthscale=ell.detach().cpu().reshape(-1).tolist(),
        sigma_f=cfg.sigma_f,
        sigma_g=cfg.sigma_g,
        outputscale=cfg.outputscale,
    )
    torch.save(
        dict(
            state={
                name: getattr(state, name).detach().cpu()
                for name in ("X", "y", "g", "neighbors", "sample_positions")
            },
            targets=targets,
            lengthscale=ell.detach().cpu(),
            metadata=meta,
        ),
        cache_path,
    )
    return meta


class StepCase:
    def __init__(self, cfg, payload, method, m, batch_size, *, allow_repeat_targets=False):
        if method not in {"vecchia", "lite", "tera_batched"}:
            raise ValueError("Use vecchia, lite or tera_batched; sequential TERA is not supported.")
        if batch_size < 1 or (batch_size > len(payload["targets"]) and not allow_repeat_targets):
            raise ValueError("Batch size exceeds the number of eligible distinct targets.")
        if not 1 <= m <= payload["metadata"]["max_m"]:
            raise ValueError("m exceeds the cached graph size.")
        self.device = torch.device(cfg.device)
        if self.device.type == "cuda":
            torch.cuda.set_device(self.device)
        self.model = _make_model(
            method, replace(cfg, m=m), cfg.seeds[0], dataset_name=cfg.datasets[0]
        )
        self.method = method
        state = payload["state"]
        if method == "vecchia":
            state = dict(state, g=state["X"].new_empty((len(state["X"]), 0)))
        self.state = VecchiaTrainingState(
            **{
                name: (value[:, :m].contiguous() if name == "neighbors" else value).to(
                    "cpu" if name == "sample_positions" else self.device
                )
                for name, value in state.items()
            }
        )
        self.targets = (
            payload["targets"]
            .repeat(math.ceil(batch_size / len(payload["targets"])))[:batch_size]
            .contiguous()
        )
        self.ell = payload["lengthscale"].to(self.device)
        dtype = self.ell.dtype
        eps = torch.finfo(dtype).eps
        self.logs = dict(
            lengthscale=torch.nn.Parameter(self.ell.detach().clone().clamp_min(eps).log()),
            outputscale=torch.nn.Parameter(
                self.ell.new_tensor(cfg.outputscale).clamp_min(eps).log()
            ),
            sigma_f=torch.nn.Parameter(
                self.ell.new_tensor(max(cfg.sigma_f - self.model.min_sigma_f, eps)).log()
            ),
            sigma_g=torch.nn.Parameter(
                self.ell.new_tensor(max(cfg.sigma_g - self.model.min_sigma_g, eps)).log()
            ),
        )
        self.initial = dict(
            lengthscale=self.ell.detach(),
            outputscale=self.ell.new_tensor(cfg.outputscale),
            sigma_f=self.ell.new_tensor(cfg.sigma_f),
            sigma_g=self.ell.new_tensor(cfg.sigma_g),
        )
        self.active = [p for name, p in self.logs.items() if getattr(self.model, f"learn_{name}")]
        if not self.active:
            raise ValueError("A forward/backward step needs at least one learned parameter.")

    def zero_grad(self):
        for value in self.logs.values():
            value.grad = None

    def forward(self, profile_regions=False):
        with step_scope("other", profile_regions):
            values = {}
            for name, parameter in self.logs.items():
                if getattr(self.model, f"learn_{name}"):
                    value = parameter.exp()
                    if name in {"sigma_f", "sigma_g"}:
                        value = value + getattr(self.model, f"min_{name}")
                else:
                    value = self.initial[name]
                values[name] = value
            values["lengthscale"] = self.model._likelihood_lengthscale(values["lengthscale"])
        kwargs = dict(
            state=self.state,
            target_positions=self.targets,
            kernel=self.model.kernel,
            gradient_noise_model=self.model.gradient_noise_model,
            **values,
        )
        if self.method == "lite":
            kwargs["profile_regions"] = profile_regions
        return self.model._batch_nll(**kwargs)


def measure_case(case, *, warmup_steps=2, measurements=5):
    if warmup_steps < 0 or measurements < 1:
        raise ValueError("Invalid warmup/measurement count.")
    device = case.device
    for _ in range(warmup_steps):
        case.zero_grad()
        loss = case.forward()
        loss.backward()
        del loss
    case.zero_grad()
    gc.collect()
    sync(device)
    if device.type == "cuda":
        baseline = torch.cuda.memory_allocated(device) / 1024**3
        free, total = torch.cuda.mem_get_info(device)
    else:
        baseline, free, total = 0.0, 0, 0
    times, peaks, reserved, cuda_times, forward_times, backward_times = [], [], [], [], [], []
    losses = []
    for _ in range(measurements):
        case.zero_grad()
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            start_event, forward_event, end_event = [
                torch.cuda.Event(enable_timing=True) for _ in range(3)
            ]
            start_event.record()
        start = time.perf_counter()
        loss = case.forward()
        if device.type == "cuda":
            forward_event.record()
        loss.backward()
        if device.type == "cuda":
            end_event.record()
        sync(device)
        times.append(time.perf_counter() - start)
        if device.type == "cuda":
            peaks.append(torch.cuda.max_memory_allocated(device) / 1024**3)
            reserved.append(torch.cuda.max_memory_reserved(device) / 1024**3)
            cuda_times.append(start_event.elapsed_time(end_event) / 1000)
            forward_times.append(start_event.elapsed_time(forward_event) / 1000)
            backward_times.append(forward_event.elapsed_time(end_event) / 1000)
        else:
            peaks.append(0.0)
            reserved.append(0.0)
        losses.append(float(loss.detach()))
        if not math.isfinite(losses[-1]) or any(
            not bool(torch.isfinite(p.grad).all()) for p in case.active if p.grad is not None
        ):
            raise FloatingPointError("Nonfinite loss or parameter gradients.")
        if any(p.grad is None for p in case.active):
            raise RuntimeError("Missing gradient for a learned parameter.")
        del loss
    return dict(
        status="ok",
        step_time_sec=float(np.median(times)),
        step_times_sec=json.dumps(times),
        peak_mem_gib=max(peaks),
        peak_reserved_gib=max(reserved),
        baseline_mem_gib=baseline,
        incremental_peak_mem_gib=max(peaks) - baseline,
        cuda_step_sec=float(np.median(cuda_times)) if cuda_times else None,
        cuda_forward_sec=float(np.median(forward_times)) if forward_times else None,
        cuda_backward_sec=float(np.median(backward_times)) if backward_times else None,
        loss=losses[-1],
        warmup_steps=warmup_steps,
        measurements=measurements,
        learned_parameters=json.dumps(
            [name for name in case.logs if getattr(case.model, f"learn_{name}")]
        ),
        gpu_total_mem_gib=total / 1024**3,
        gpu_free_mem_gib_before=free / 1024**3,
        gpu_name=torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    )


def stage_for_event(event, regions):
    current = event
    root_found = False
    while current is not None:
        if current.name.startswith("step/"):
            name = current.name.split("/", 1)[1]
            if name in STAGES:
                return name
            root_found |= name == "iteration"
        current = current.cpu_parent
    start = event.time_range.start
    containing = [r for r in regions if r.time_range.start <= start < r.time_range.end]
    if containing:
        inner = min(containing, key=lambda r: r.time_range.end - r.time_range.start)
        name = inner.name.split("/", 1)[1]
        return name if name in STAGES else "other"
    return "other" if root_found else None


def profile_case(case, outdir, *, steps=3, warmup_steps=2):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for _ in range(warmup_steps):
        case.zero_grad()
        loss = case.forward()
        loss.backward()
        del loss
    sync(case.device)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if case.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        for _ in range(steps):
            case.zero_grad()
            with step_scope("iteration", True):
                loss = case.forward(profile_regions=True)
                with step_scope("backward", True):
                    loss.backward()
                del loss
        sync(case.device)
    prof.export_chrome_trace(str(outdir / "trace.json"))
    events = list(prof.events())
    regions = [event for event in events if event.name.startswith("step/")]
    records = []
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CPU:
            continue
        stage = stage_for_event(event, regions)
        if stage is None:
            continue
        gpu_us = getattr(event, "self_device_time_total", None)
        if gpu_us is None:
            gpu_us = event.self_cuda_time_total
        records.append(
            dict(
                stage=stage,
                operator=event.name,
                cpu_self_us=event.self_cpu_time_total,
                cuda_kernel_us=gpu_us,
                calls=1,
            )
        )
    operators = pd.DataFrame(records).groupby(["stage", "operator"], as_index=False).sum()
    operators.to_csv(outdir / "operators.csv", index=False)
    totals = (
        operators.groupby("stage")[["cpu_self_us", "cuda_kernel_us"]]
        .sum()
        .reindex(STAGES, fill_value=0)
    )
    summary = totals / (steps * 1000)
    summary.columns = ["cpu_self_ms_per_step", "cuda_kernel_ms_per_step"]
    for column in list(summary.columns):
        total = summary[column].sum()
        summary[column.replace("_ms_per_step", "_percent")] = (
            summary[column] / total * 100 if total else 0.0
        )
    summary.reset_index().to_csv(outdir / "breakdown.csv", index=False)
    sort_key = (
        (
            "self_device_time_total"
            if events and hasattr(events[0], "self_device_time_total")
            else "self_cuda_time_total"
        )
        if case.device.type == "cuda"
        else "self_cpu_time_total"
    )
    (outdir / "profiler.txt").write_text(prof.key_averages().table(sort_by=sort_key, row_limit=60))
    return dict(
        status="ok", profile_steps=steps, profile_dir=str(outdir), trace=str(outdir / "trace.json")
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "measure", "profile"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--method", default="lite")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measurements", type=int, default=5)
    parser.add_argument("--profile-dir")
    parser.add_argument("--allow-repeat-targets", action="store_true")
    args = parser.parse_args(argv)
    cfg = MD22Config(**json.loads(Path(args.config).read_text()))
    try:
        if args.mode == "prepare":
            result = prepare_cache(cfg, args.cache, args.m)
            result["status"] = "ok"
        else:
            payload = torch.load(args.cache, map_location="cpu", weights_only=False)
            case = StepCase(
                cfg,
                payload,
                args.method,
                args.m,
                args.batch_size,
                allow_repeat_targets=args.allow_repeat_targets,
            )
            if args.mode == "measure":
                result = measure_case(
                    case, warmup_steps=args.warmup_steps, measurements=args.measurements
                )
            else:
                result = profile_case(
                    case, args.profile_dir, steps=args.measurements, warmup_steps=args.warmup_steps
                )
    except torch.cuda.OutOfMemoryError as exc:
        result = dict(status="oom", error=str(exc), step_time_sec=None, peak_mem_gib=None)
    except Exception as exc:
        result = dict(
            status="error",
            error=f"{type(exc).__name__}: {exc}",
            step_time_sec=None,
            peak_mem_gib=None,
        )
    Path(args.result).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
