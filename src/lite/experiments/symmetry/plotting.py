import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterMathtext, LogLocator, ScalarFormatter

from lite.plotting.style import setup_style


def summarize(data):
    required = {"d", "repeat", "m", "n_targets", "mean_cs_error", "mean_expected_kl", "status"}
    if not required.issubset(data.columns) or data.empty:
        raise ValueError("Expected nonempty symmetry results.csv with repeat-level metrics.")
    if not (data.status == "ok").all() or data.duplicated(["d", "repeat"]).any():
        raise ValueError("Failed or duplicate repeat rows present.")
    if data.m.nunique() != 1 or data.n_targets.nunique() != 1:
        raise ValueError("Keep m and the number of targets fixed across the dimension sweep.")
    for name in ("seed", "kernel", "outputscale", "sigma_f", "sigma_g"):
        if name in data and data[name].nunique() != 1:
            raise ValueError(f"Do not pool different {name} settings in one sweep.")
    repeat_sets = data.groupby("d").repeat.apply(lambda values: tuple(sorted(values)))
    if repeat_sets.nunique() != 1:
        raise ValueError("Every dimension must have the same completed repeats.")
    rows = []
    for d, part in data.groupby("d", sort=True):
        row = dict(d=d, repeats=len(part))
        for metric in ("cs_error", "expected_kl"):
            values = part[f"mean_{metric}"].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Invalid {metric} values.")
            row[f"mean_{metric}"] = float(values.mean())
            row[f"sem_{metric}"] = (
                float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else 0.0
            )
        rows.append(row)
    return pd.DataFrame(rows)


@plt.rc_context()
def make_figure(summary, *, kl_scale="log"):
    if kl_scale not in {"linear", "log"}:
        raise ValueError("kl_scale must be linear or log.")
    setup_style()
    fig, left = plt.subplots(figsize=(4.8, 2.6), layout="constrained")
    right = left.twinx()
    dimensions = summary.d.to_numpy()
    handles = []
    for ax, metric, label, linestyle, marker in (
        (left, "cs_error", "CS error", "-", "o"),
        (right, "expected_kl", "Expected KL", "--", "s"),
    ):
        mean = summary[f"mean_{metric}"].to_numpy()
        sem = summary[f"sem_{metric}"].to_numpy()
        lower, upper = np.maximum(mean - sem, 0), mean + sem
        if metric == "expected_kl" and kl_scale == "log":
            mean, lower, upper = [np.maximum(x, 1e-16) for x in (mean, lower, upper)]
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
        (line,) = ax.plot(
            dimensions,
            mean,
            color="#414A4C",
            linestyle=linestyle,
            marker=marker,
            linewidth=2.0,
            ms=6,
            label=label,
        )
        handles.append(line)
        ax.fill_between(dimensions, lower, upper, color="#414A4C", alpha=0.15)
        if metric == "cs_error" or kl_scale == "linear":
            ax.set_ylim(bottom=0)
        ax.set_ylabel(label, fontsize=12)
        ax.yaxis.set_tick_params(labelsize=10)
        ax.spines["top"].set_visible(False)
    left.set_xlabel("Input dimension", fontsize=12)
    left.set_xscale("log")
    left.set_xticks(dimensions)
    left.xaxis.set_major_formatter(ScalarFormatter())
    left.xaxis.set_tick_params(labelsize=10, top=False)
    left.minorticks_off()
    right.minorticks_off()
    left.spines["right"].set_visible(False)
    right.spines[["left", "bottom"]].set_visible(False)
    right.tick_params(axis="y", left=False, right=True)
    left.set_axisbelow(True)
    left.grid(linewidth=0.4, alpha=0.3)
    right.grid(False)
    left.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        fontsize=10,
        handlelength=2.4,
        labelspacing=0.4,
    )
    return fig


def plot(csv, outdir, *, kl_scale="log"):
    summary = summarize(pd.read_csv(csv))
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "plot_summary.csv", index=False)
    fig = make_figure(summary, kl_scale=kl_scale)
    for suffix in ("pdf", "png"):
        fig.savefig(
            out / f"symmetry_d_sweep.{suffix}", dpi=300, bbox_inches="tight", pad_inches=0.02
        )
    plt.close(fig)
    return summary


def main():
    p = argparse.ArgumentParser(
        description="Compound-symmetry error and expected KL on two y axes."
    )
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--kl-scale", choices=["linear", "log"], default="log")
    args = p.parse_args()
    plot(args.csv, args.outdir, kl_scale=args.kl_scale)


if __name__ == "__main__":
    main()
