from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style

PROFILE_LABELS = {
    "gather_gram": "Gather and Gram",
    "assembly": "Assembly",
    "cholesky": "Cholesky and solves",
    "backward": "Backward",
    "other": "Other",
}


def _save_figure(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def _read_fixed_runs(csv, batch_sizes=None):
    paths = [csv] if isinstance(csv, (str, Path)) else list(csv)
    if not paths:
        raise ValueError("Provide at least one results CSV.")
    frames = []
    required = {
        "method",
        "m",
        "seed",
        "batch_mode",
        "batch_size",
        "status",
        "step_time_sec",
        "peak_mem_gib",
    }
    for path in paths:
        frame = pd.read_csv(path)
        if required - set(frame):
            raise ValueError(f"Missing columns in {path}: {sorted(required - set(frame))}")
        frame["source_csv"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df = df[df.method.isin(["vecchia", "lite", "tera_batched"]) & df.batch_mode.eq("fixed")].copy()
    for column in ["m", "seed", "batch_size", "step_time_sec", "peak_mem_gib"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if batch_sizes is not None:
        df = df[df.batch_size.isin(batch_sizes)].copy()
    if df.empty:
        raise ValueError("No matching fixed-batch Vecchia GP/batched TERA/LITE measurements.")
    for column in ["m", "seed", "batch_size"]:
        values = df[column]
        if not np.isfinite(values).all() or not values.eq(values.round()).all():
            raise ValueError(f"{column} must contain finite integer values.")
        if column != "seed" and not values.gt(0).all():
            raise ValueError(f"{column} must be positive.")
        df[column] = values.astype(int)
    keys = ["method", "m", "seed", "batch_size"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate method/m/seed/batch-size rows across input CSVs.")
    for column in [
        "dataset",
        "kernel",
        "dtype",
        "use_ard",
        "timing_scope",
        "parameters",
        "n_train",
        "d",
        "outputscale",
        "sigma_f",
        "sigma_g",
        "preprocessing_version",
    ]:
        if column in df and df[column].nunique(dropna=False) > 1:
            raise ValueError(f"CSV files mix {column} settings.")
    for column in ["split_id", "lengthscale"]:
        if column in df and (df.groupby("seed")[column].nunique(dropna=False) > 1).any():
            raise ValueError(f"Batch settings do not share {column} within each seed.")
    if "gradient_noise_model" in df:
        if (df.groupby("method").gradient_noise_model.nunique(dropna=False) > 1).any():
            raise ValueError("A method uses different gradient-noise models across batch settings.")
    df["time_included"] = (
        df.status.eq("ok") & np.isfinite(df.step_time_sec) & df.step_time_sec.gt(0)
    )
    df["memory_included"] = (
        df.status.eq("ok") & np.isfinite(df.peak_mem_gib) & df.peak_mem_gib.gt(0)
    )
    return df


@plt.rc_context()
def plot_profile(profile_csv, out_path, *, profile_metric="cuda"):
    if profile_metric not in {"cuda", "cpu"}:
        raise ValueError("Profiler metric must be cuda or cpu.")
    setup_style()
    profile = pd.read_csv(profile_csv)
    if "stage" not in profile or profile.stage.duplicated().any():
        raise ValueError("Profiler CSV must contain one row per stage.")
    profile = profile.set_index("stage").reindex(PROFILE_LABELS)
    column = "cuda_kernel_percent" if profile_metric == "cuda" else "cpu_self_percent"
    if column not in profile:
        raise ValueError(f"Missing profiler column: {column}")
    percentages = pd.to_numeric(profile[column], errors="coerce").to_numpy(float)
    if not np.isfinite(percentages).all() or (percentages < 0).any() or percentages.sum() <= 0:
        raise ValueError("No valid profiler times for the selected CPU/CUDA metric.")
    fig, ax = plt.subplots(figsize=(4.8, 2.6), layout="constrained")
    positions = np.arange(len(PROFILE_LABELS))
    ax.barh(positions, percentages, color=METHOD_COLORS["lite"], height=0.6)
    ax.set_yticks(positions, list(PROFILE_LABELS.values()), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("CUDA kernel time (%)" if profile_metric == "cuda" else "CPU self time (%)")
    ax.set_xlim(0, max(40, float(percentages.max()) * 1.25))
    for index, value in enumerate(percentages):
        ax.text(value + 1, index, f"{value:.1f}%", va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", linewidth=0.4, alpha=0.3)
    _save_figure(fig, out_path)
    return profile


@plt.rc_context()
def plot_step_scaling(
    csv,
    out_path,
    *,
    errorbar="sem",
    profile_csv=None,
    profile_metric="cuda",
    profile_out=None,
    batch_sizes=None,
    annotate_max_batch=False,
):
    if errorbar not in {"sem", "std"} or profile_metric not in {"cuda", "cpu"}:
        raise ValueError("Invalid errorbar or profiler metric.")
    setup_style()
    df = _read_fixed_runs(csv, batch_sizes=batch_sizes)
    values = sorted(df.m.unique())
    batches = sorted(df.batch_size.unique())
    metrics = ["step_time_sec", "peak_mem_gib"]
    good = df[df.status.eq("ok")].copy()
    if good.empty:
        raise ValueError("No successful fixed-batch step measurements.")
    for metric, included in zip(metrics, ["time_included", "memory_included"]):
        good.loc[~good[included], metric] = np.nan
    agg = good.groupby(["method", "batch_size", "m"])[metrics].agg(["mean", errorbar, "count"])
    agg.columns = [f"{metric}_{stat}" for metric, stat in agg.columns]
    agg = agg.reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
    handles = []
    styles = ["-", "--", ":", "-."]
    labels = {"vecchia": "Vecchia GP", "tera_batched": "TERA", "lite": "LITE"}
    markers = {"vecchia": "s", "tera_batched": "o", "lite": "D"}
    for index, batch in enumerate(batches):
        for method in ["vecchia", "tera_batched", "lite"]:
            if not (df.method.eq(method) & df.batch_size.eq(batch)).any():
                continue
            lite = method == "lite"
            color = METHOD_COLORS[method]
            marker = markers[method]
            linestyle = styles[index % len(styles)]
            facecolor = color if index == 0 else "white"
            label = f"{labels[method]}, B={batch}"
            handles.append(
                Line2D(
                    [],
                    [],
                    color=color,
                    marker=marker,
                    markersize=6,
                    markerfacecolor=facecolor,
                    markeredgecolor=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=label,
                )
            )
            sub = (
                agg[agg.method.eq(method) & agg.batch_size.eq(batch)].set_index("m").reindex(values)
            )
            for ax, metric in zip(axes, metrics):
                y = sub[f"{metric}_mean"].to_numpy(float)
                yerr = sub[f"{metric}_{errorbar}"].fillna(0).to_numpy(float)
                valid = np.isfinite(y) & (y > 0)
                y[~valid] = np.nan
                if valid.any():
                    ax.errorbar(
                        values,
                        y,
                        yerr=yerr,
                        color=color,
                        linestyle=linestyle,
                        marker=marker,
                        markersize=6,
                        markerfacecolor=facecolor,
                        markeredgecolor=color,
                        linewidth=2,
                        elinewidth=1,
                        capsize=3,
                        zorder=6 if lite else 4,
                    )
    for ax, ylabel, included in zip(
        axes,
        ["Forward + backward time (s)", "Peak GPU memory (GiB)"],
        ["time_included", "memory_included"],
    ):
        ax.set_xlabel("Conditioning set size")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(values, [f"{m:g}" for m in values])
        ax.tick_params(axis="x", labelsize=8, rotation=30)
        ax.set_xlim(values[0] / 1.15, values[-1] * 1.15)
        ax.margins(y=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, which="major", linewidth=0.4, alpha=0.3)
        if not df[included].any():
            ax.text(
                0.5,
                0.45,
                "No GPU memory measurements"
                if included == "memory_included"
                else "No valid timing measurements",
                transform=ax.transAxes,
                ha="center",
                fontsize=9,
            )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=len(handles),
        frameon=False,
        fontsize=9,
        columnspacing=1.1,
        handlelength=2.2,
    )
    out_path = Path(out_path)
    _save_figure(fig, out_path)
    agg.to_csv(out_path.with_name(out_path.stem + "_summary.csv"), index=False)
    df.to_csv(out_path.with_name(out_path.stem + "_runs.csv"), index=False)
    df.groupby(["method", "batch_size", "m", "status"]).size().rename("runs").reset_index().to_csv(
        out_path.with_name(out_path.stem + "_coverage.csv"), index=False
    )
    if profile_csv is not None:
        if profile_out is None:
            suffix = "cuda_profile" if profile_metric == "cuda" else "cpu_profile"
            profile_out = out_path.with_name(f"{out_path.stem}_{suffix}.png")
        if (
            Path(profile_out).with_suffix(".png").resolve()
            == out_path.with_suffix(".png").resolve()
        ):
            raise ValueError("Profile output must differ from the two-panel output.")
        plot_profile(profile_csv, profile_out, profile_metric=profile_metric)
    return agg


def _parse_batches(value):
    batches = [int(batch.strip()) for batch in value.split(",") if batch.strip()]
    if not batches or min(batches) < 1:
        raise argparse.ArgumentTypeError("Provide positive comma-separated batch sizes.")
    return sorted(set(batches))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fixed-batch MD22 training-step scaling and separate profiling."
    )
    parser.add_argument("--csv", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--batch-sizes", type=_parse_batches)
    parser.add_argument("--errorbar", choices=["sem", "std"], default="sem")
    parser.add_argument("--profile-csv")
    parser.add_argument("--profile-out")
    parser.add_argument("--profile-metric", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--annotate-max-batch", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.profile_only:
        if args.profile_csv is None:
            parser.error("--profile-only requires --profile-csv")
        plot_profile(
            args.profile_csv, args.profile_out or args.out, profile_metric=args.profile_metric
        )
    else:
        if args.csv is None:
            parser.error("--csv is required for the two-panel scaling figure")
        plot_step_scaling(
            args.csv,
            args.out,
            errorbar=args.errorbar,
            batch_sizes=args.batch_sizes,
            profile_csv=args.profile_csv,
            profile_metric=args.profile_metric,
            profile_out=args.profile_out,
        )


if __name__ == "__main__":
    main()
