from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterMathtext, LogLocator, NullLocator

from lite.experiments.md22.io import read_results
from lite.experiments.md22.observation_noise import NOISE_COLUMNS, noise_condition_frame, noise_tag
from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style


@plt.rc_context()
def plot(csv_path, outdir, *, phase="train", errorbar="sem"):
    setup_style()
    df = read_results(csv_path)
    if "sweep" in df:
        df = df[df["sweep"].eq("m")]
    df = noise_condition_frame(df)
    if phase not in {"train", "total"} or errorbar not in {"sem", "std"}:
        raise ValueError("Invalid phase or errorbar setting")
    metrics = [
        ("fit_time_sec", "Training time (s)"),
        ("fit_peak_mem_gb", "Peak training memory (GiB)"),
    ]
    if phase == "total":
        metrics = [
            ("wall_time_sec", "Wall-clock time (s)"),
            ("peak_mem_gb", "Peak GPU memory (GB)"),
        ]
    required = {"dataset", "method", "seed", "m", "status", "raw_energy_rmse_per_atom"} | {
        x[0] for x in metrics
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    for name in ("m", "raw_energy_rmse_per_atom", *(x[0] for x in metrics)):
        df[name] = pd.to_numeric(df[name], errors="coerce")
    keys = ["dataset", *NOISE_COLUMNS, "method", "seed", "m"]
    if df.duplicated(keys).any():
        raise ValueError(
            "Duplicate dataset/noise/method/seed/m runs. Select one experiment setting."
        )
    if df.empty:
        raise ValueError("No m-scaling rows")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    labels = {
        "vecchia": "Vecchia GP",
        "tera_batched": "TERA",
        "tera": "TERA (sequential)",
        "lite": "LITE",
    }
    markers = {"vecchia": "s", "tera_batched": "o", "tera": "o", "lite": "D"}
    for (dataset, energy, force), part in df.groupby(["dataset", *NOISE_COLUMNS]):
        successful = part[part["status"].eq("ok")]
        if successful.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
        order = {"vecchia": 0, "tera": 1, "tera_batched": 2, "lite": 3}
        methods = sorted(successful["method"].unique(), key=lambda m: (order.get(m, 4), m))
        columns = ["raw_energy_rmse_per_atom", *(x[0] for x in metrics)]
        for method in methods:
            stats = (
                successful[successful["method"].eq(method)]
                .groupby("m")[columns]
                .agg(["mean", errorbar])
            )
            for ax, (metric, xlabel) in zip(axes, metrics):
                x = stats[(metric, "mean")].to_numpy()
                y = stats[("raw_energy_rmse_per_atom", "mean")].to_numpy()
                xerr = stats[(metric, errorbar)].fillna(0).to_numpy()
                yerr = stats[("raw_energy_rmse_per_atom", errorbar)].fillna(0).to_numpy()
                good = np.isfinite(x) & np.isfinite(y) & (x > 0)
                if not good.any():
                    continue
                ax.errorbar(
                    x[good],
                    y[good],
                    xerr=xerr[good],
                    yerr=yerr[good],
                    color=METHOD_COLORS.get(method),
                    label=labels.get(method, method),
                    marker=markers.get(method, "o"),
                    markersize=6,
                    linewidth=2,
                    elinewidth=1,
                    capsize=3,
                    linestyle="--" if method == "vecchia" else "-",
                    zorder=5 if method == "lite" else 3,
                )
                for index, (m, xi, yi) in enumerate(zip(stats.index[good], x[good], y[good])):
                    offset = (
                        (-5, 10 if index % 2 == 0 else -14)
                        if method == "lite"
                        else (5, -14 if index % 2 == 0 else 10)
                    )
                    ax.annotate(
                        f"{m:g}",
                        (xi, yi),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=8,
                        color=METHOD_COLORS.get(method),
                        ha="right" if method == "lite" else "left",
                    )
        for ax, (metric, xlabel) in zip(axes, metrics):
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Test RMSE")
            ax.set_xscale("log")
            if "time" in metric:
                ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
                ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10))
                ax.xaxis.set_minor_locator(NullLocator())
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(True, linewidth=0.4, alpha=0.3)
            if ax.has_data():
                ax.legend(frameon=False, fontsize=9)
            else:
                ax.text(0.5, 0.5, "No measurements", ha="center", transform=ax.transAxes)
        name = f"{dataset}_{noise_tag(energy, force)}_tradeoff_{phase}"
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{name}.{extension}", dpi=300, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="MD22 RMSE-cost curves traced over conditioning set size"
    )
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--phase", choices=["train", "total"], default="train")
    parser.add_argument("--errorbar", choices=["sem", "std"], default="sem")
    args = parser.parse_args()
    plot(args.csv, args.outdir, phase=args.phase, errorbar=args.errorbar)


if __name__ == "__main__":
    main()
