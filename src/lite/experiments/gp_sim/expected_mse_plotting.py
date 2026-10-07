from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style


def read_runs(csv_paths, *, d=None, n_train=None, m_values=None):
    if isinstance(csv_paths, (str, Path)):
        csv_paths = [csv_paths]
    frames = []
    for path in csv_paths:
        frame = pd.read_csv(path)
        frame["source_csv"] = str(path)
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    required = [
        "evaluation_mode",
        "method",
        "seed",
        "repeat",
        "d",
        "n_train",
        "n_eval",
        "m",
        "status",
        "root_expected_mse",
        "wall_time_sec",
        "peak_mem_gib",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Missing expected-MSE columns: {missing}")
    if not df.evaluation_mode.eq("expected_mse").all():
        raise ValueError("Empirical RMSE and expected MSE cannot be mixed")
    if d is not None:
        df = df[df.d.eq(d)]
    if n_train is not None:
        df = df[df.n_train.eq(n_train)]
    if m_values is not None:
        df = df[df.m.isin(m_values)]
    if df.empty:
        raise ValueError("No matching expected-MSE results")
    keys = ["method", "seed", "repeat", "m"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate method/seed/repeat/m rows; do not combine repeated executions")
    for column in [
        "kernel",
        "use_ard",
        "d",
        "n_train",
        "n_eval",
        "sigma_f",
        "sigma_g",
        "outputscale",
        "target_median_correlation",
        "dtype",
        "device",
        "timing_scope",
    ]:
        if column in df and df[column].nunique(dropna=False) != 1:
            raise ValueError(f"Results mix {column} settings; plot one configuration at a time")
    for column in ["input_design_id", "lengthscale_values"]:
        if (
            column in df
            and (df.groupby(["seed", "repeat"])[column].nunique(dropna=False) > 1).any()
        ):
            raise ValueError(f"Methods or m values do not share {column}")
    for column in ["root_expected_mse", "wall_time_sec", "peak_mem_gib"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["included"] = (
        df.status.eq("ok")
        & np.isfinite(df.root_expected_mse)
        & df.root_expected_mse.ge(0)
        & np.isfinite(df.wall_time_sec)
        & df.wall_time_sec.gt(0)
    )
    if not df.included.any():
        raise ValueError("No successful expected-MSE predictions")
    return df


def aggregate_runs(df):
    metrics = ["root_expected_mse", "wall_time_sec", "peak_mem_gib"]
    grouped = df[df.included].groupby(["method", "m"])[metrics].agg(["mean", "std", "count"])
    grouped.columns = ["_".join(column) for column in grouped.columns]
    result = grouped.reset_index()
    for metric in metrics:
        result[f"{metric}_std"] = result[f"{metric}_std"].fillna(0)
        result[f"{metric}_sem"] = result[f"{metric}_std"] / np.sqrt(
            result[f"{metric}_count"].clip(lower=1)
        )
    return result


@plt.rc_context()
def plot_expected_mse(
    csv_paths,
    out_path,
    *,
    d=None,
    n_train=None,
    m_values=None,
    errorbar="sem",
    cost_scale="log",
    error_scale="linear",
    annotate_m=True,
):
    if (
        errorbar not in {"sem", "std"}
        or cost_scale not in {"log", "linear"}
        or error_scale not in {"log", "linear"}
    ):
        raise ValueError("Invalid plot setting")
    setup_style()
    audit = read_runs(csv_paths, d=d, n_train=n_train, m_values=m_values)
    m_values = sorted(audit.m.unique()) if m_values is None else sorted(m_values)
    agg = aggregate_runs(audit)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
    methods = [
        name for name in ["Vecchia GP", "TERA (batched)", "LITE"] if name in audit.method.unique()
    ]
    keys = {"Vecchia GP": "vecchia", "TERA (batched)": "tera_batched", "LITE": "lite"}
    markers = {"Vecchia GP": "s", "TERA (batched)": "o", "LITE": "D"}
    labels = {"Vecchia GP": "Vecchia GP", "TERA (batched)": "TERA", "LITE": "LITE"}
    for ax, metric, label in zip(
        axes, ["wall_time_sec", "peak_mem_gib"], ["Wall-clock time (s)", "Peak memory (GB)"]
    ):
        has_data = False
        for method in methods:
            lite = method == "LITE"
            color = METHOD_COLORS[keys[method]]
            part = agg[agg.method.eq(method)].set_index("m").reindex(m_values)
            x = part[f"{metric}_mean"].to_numpy(float)
            y = part["root_expected_mse_mean"].to_numpy(float)
            xerr = part[f"{metric}_{errorbar}"].fillna(0).to_numpy(float)
            yerr = part[f"root_expected_mse_{errorbar}"].fillna(0).to_numpy(float)
            valid = np.isfinite(x) & (x > 0) & np.isfinite(y)
            if error_scale == "log":
                valid &= y > 0
            x[~valid], y[~valid] = np.nan, np.nan
            if valid.any():
                has_data = True
                ax.errorbar(
                    x,
                    y,
                    xerr=xerr,
                    yerr=yerr,
                    color=color,
                    marker=markers[method],
                    markersize=6 if lite else 4.5,
                    linewidth=2,
                    elinewidth=1,
                    capsize=3,
                    zorder=6 if lite else 4,
                )
                if annotate_m:
                    offset, ha, va = {
                        "Vecchia GP": ((-7, 0), "right", "center"),
                        "TERA (batched)": ((5, 7), "left", "bottom"),
                        "LITE": ((7, 0), "left", "top"),
                    }[method]

                    for m, xv, yv in zip(m_values, x, y):
                        if np.isfinite(xv) and np.isfinite(yv):
                            ax.annotate(
                                f"{m:g}",
                                (xv, yv),
                                xytext=offset,
                                textcoords="offset points",
                                ha=ha,
                                va=va,
                                color=color,
                                fontsize=8,
                                zorder=7,
                            )
        ax.set_xlabel(label)
        ax.set_ylabel("RMSE")
        ax.set_xscale(cost_scale)
        ax.set_yscale(error_scale)
        if metric == "wall_time_sec" and cost_scale == "log":
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
            ax.xaxis.set_minor_locator(NullLocator())
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", which="both")
        ax.xaxis.get_offset_text()
        ax.yaxis.get_offset_text()
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
        ax.margins(x=0.15, y=0.18)
        if not has_data:
            ax.text(
                0.5,
                0.5,
                "No CUDA memory measurements"
                if metric == "peak_mem_gib"
                else "No successful measurements",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
            )
    low, high = min(ax.get_ylim()[0] for ax in axes), max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(low, high)
    handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[keys[m]],
            marker=markers[m],
            markersize=6 if m == "LITE" else 4.5,
            linewidth=2,
            label=labels[m],
        )
        for m in methods
    ]
    axes[0].legend(
        handles=handles,
        loc="lower right",
        frameon=False,
        fontsize=9,
        handlelength=2.2,
    )
    oom = []
    for method in methods:
        sizes = [
            m
            for m in m_values
            if not audit[audit.method.eq(method) & audit.m.eq(m)].empty
            and audit[audit.method.eq(method) & audit.m.eq(m)].status.eq("oom").all()
        ]
        if sizes:
            oom.append(f"{labels[method]} OOM: m=" + ", ".join(map(str, sizes)))
    if oom:
        axes[1].text(
            0.98,
            0.04,
            "\n".join(oom),
            transform=axes[1].transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="black",
        )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for ext in ["png", "pdf"]:
        fig.savefig(out_path.with_suffix(f".{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    agg.to_csv(out_path.with_name(f"{out_path.stem}_summary.csv"), index=False)
    audit.to_csv(out_path.with_name(f"{out_path.stem}_runs.csv"), index=False)
    coverage = audit.groupby(["method", "m", "status"]).size().rename("runs").reset_index()
    coverage.to_csv(out_path.with_name(f"{out_path.stem}_coverage.csv"), index=False)
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="Root expected MSE versus conditional prediction cost"
    )
    parser.add_argument("--csv", nargs="+", required=True, dest="csv_paths")
    parser.add_argument("--out", required=True, dest="out_path")
    parser.add_argument("--d", type=int)
    parser.add_argument("--n-train", type=int)
    parser.add_argument("--m-values", type=lambda value: [int(v) for v in value.split(",")])
    parser.add_argument("--errorbar", choices=["sem", "std"], default="sem")
    parser.add_argument("--cost-scale", choices=["log", "linear"], default="log")
    parser.add_argument("--error-scale", choices=["log", "linear"], default="linear")
    parser.add_argument("--annotate-m", action=argparse.BooleanOptionalAction, default=True)
    plot_expected_mse(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
