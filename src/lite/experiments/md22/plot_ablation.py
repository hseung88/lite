import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lite.experiments.md22.io import read_results
from lite.experiments.md22.observation_noise import NOISE_COLUMNS, noise_condition_frame, noise_tag
from lite.experiments.md22.plotting import MARKERS, METHOD_LABELS
from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style


@plt.rc_context()
def plot(csv_path, outdir, *, phase="total", errorbar="std"):

    setup_style()
    if phase not in {"total", "train", "predict"} or errorbar not in {"std", "sem"}:
        raise ValueError("Invalid phase or errorbar setting.")
    df = read_results(csv_path)
    if df.empty:
        raise ValueError("No ablation rows")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    df = noise_condition_frame(df)
    for (dataset, sweep, energy, force), sub in df.groupby(["dataset", "sweep", *NOISE_COLUMNS]):
        if sweep == "noise":
            sub = sub[sub.status.eq("ok")]
            if sub.empty:
                continue
            methods = list(sub.method.unique())
            fig, axes = plt.subplots(
                1,
                len(methods),
                squeeze=False,
                figsize=(4 * len(methods), 3.5),
                layout="constrained",
            )
            for ax, method in zip(axes[0], methods):
                values = sub[sub.method.eq(method)].pivot_table(
                    index="sigma_g_sweep",
                    columns="sigma_f_sweep",
                    values="raw_energy_rmse_per_atom",
                    aggfunc="mean",
                )
                im = ax.imshow(values.to_numpy(), aspect="auto", origin="lower")
                ax.set_xticks(range(len(values.columns)), [f"{x:g}" for x in values.columns])
                ax.set_yticks(range(len(values.index)), [f"{x:g}" for x in values.index])
                ax.set_xlabel("Function noise variance")
                ax.set_ylabel("Gradient noise variance")
                ax.set_title(method)
                fig.colorbar(im, ax=ax, label="Test RMSE")
        else:
            legacy_batch = sweep == "batch" and (
                "batch_size" not in sub or sub["batch_size"].isna().all()
            )
            x = (
                ("prediction_batch_size" if legacy_batch else "batch_size")
                if sweep == "batch"
                else "m"
            )
            effective_phase = "predict" if legacy_batch else phase
            time_metric, memory_metric, time_label, memory_label = {
                "total": (
                    "wall_time_sec",
                    "peak_mem_gb",
                    "Wall-clock time (s)",
                    "Peak memory (GB)",
                ),
                "train": (
                    "fit_time_sec",
                    "fit_peak_mem_gb",
                    "Training time (s)",
                    "Peak training GPU memory (GiB)",
                ),
                "predict": (
                    "predict_time_sec",
                    "predict_peak_mem_gb",
                    "Prediction time (s)",
                    "Peak prediction GPU memory (GiB)",
                ),
            }[effective_phase]
            panels = [
                (time_metric, time_label),
                (memory_metric, memory_label),
            ]
            if sweep != "batch":
                panels.insert(0, ("raw_energy_rmse_per_atom", "Test RMSE"))
            missing = [metric for metric, _ in panels if metric not in sub]
            if missing:
                raise ValueError(f"Missing columns {missing}. Select an available --phase.")
            fig, axes = plt.subplots(
                1, len(panels), figsize=(3.6 * len(panels), 2.6), layout="constrained"
            )
            methods = sorted(sub.method.unique(), key=lambda name: (name == "lite", name))
            handles, labels = [], []

            values = sorted(pd.to_numeric(sub[x], errors="coerce").dropna().unique())
            x_positions = np.arange(len(values), dtype=float)
            bar_width = min(0.36, 0.8 / max(len(methods), 1))

            for ax, (metric, label) in zip(axes, panels):
                any_data = False
                for order, method in enumerate(methods):
                    part = sub[sub.method.eq(method)].copy()
                    part[x] = pd.to_numeric(part[x], errors="coerce")
                    grid = values
                    part[metric] = pd.to_numeric(part[metric], errors="coerce")
                    good = part[part.status.eq("ok")]
                    agg = good.groupby(x)[metric].agg(["mean", errorbar]).reindex(grid)
                    y = agg["mean"].to_numpy(dtype=float)
                    err = agg[errorbar].fillna(0).to_numpy(dtype=float)
                    color = METHOD_COLORS.get(method)
                    is_lite = method == "lite"
                    if metric == "raw_energy_rmse_per_atom":
                        handle = ax.errorbar(
                            x_positions,
                            y,
                            yerr=err,
                            label=METHOD_LABELS.get(method, method.upper()),
                            color=color,
                            marker="P" if is_lite else MARKERS.get(method, "o"),
                            markersize=7.0 if is_lite else 6.5,
                            linewidth=2.0,
                            elinewidth=1.0,
                            capsize=5,
                            capthick=1.0,
                            zorder=5 if is_lite else 3,
                        )
                    else:
                        offset = (order - (len(methods) - 1) / 2) * bar_width
                        valid = np.isfinite(y) & (y > 0)

                        handle = ax.bar(
                            x_positions[valid] + offset,
                            y[valid],
                            width=bar_width,
                            yerr=err[valid],
                            label=METHOD_LABELS.get(method, method.upper()),
                            color=color,
                            edgecolor="white",
                            linewidth=0.6,
                            capsize=3,
                            error_kw={
                                "elinewidth": 1.0,
                                "capthick": 1.0,
                                "ecolor": "black",
                            },
                            zorder=3,
                        )
                    if ax is axes[0]:
                        handles.append(handle)
                        labels.append(METHOD_LABELS.get(method, method.upper()))
                    any_data |= bool(np.isfinite(y).any())

                ax.set_xlabel(
                    (
                        "Prediction batch size"
                        if legacy_batch
                        else "Training and prediction batch size"
                    )
                    if sweep == "batch"
                    else "Conditioning set size"
                )
                ax.set_ylabel(label)
                ax.spines[["top", "right"]].set_visible(False)
                ax.set_axisbelow(True)
                ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
                if not any_data:
                    ax.set_yticks([])
                    ax.text(
                        0.5,
                        0.45,
                        "No GPU memory measurement"
                        if "mem" in metric
                        else "No successful measurements",
                        transform=ax.transAxes,
                        ha="center",
                        fontsize=8,
                    )
                elif metric != "raw_energy_rmse_per_atom":
                    positive = pd.to_numeric(sub.loc[sub.status.eq("ok"), metric], errors="coerce")
                    if (positive > 0).any():
                        ax.set_yscale("log")

                ax.set_xticks(x_positions, [f"{v:g}" for v in values])
                ax.set_xlim(-0.6, len(values) - 0.4)
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.16),
                frameon=False,
                ncol=len(methods),
                handlelength=2.3,
                columnspacing=1.1,
            )
        for ext in ("png", "pdf"):
            suffix = "_" + effective_phase if sweep in {"m", "batch"} else ""
            noise_suffix = "_" + noise_tag(energy, force) if energy or force else ""
            fig.savefig(
                out / f"{dataset}_{sweep}{suffix}{noise_suffix}.{ext}", dpi=300, bbox_inches="tight"
            )
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--phase", choices=["total", "train", "predict"], default="total")
    p.add_argument("--errorbar", choices=["std", "sem"], default="std")
    a = p.parse_args()
    plot(a.csv, a.outdir, phase=a.phase, errorbar=a.errorbar)


if __name__ == "__main__":
    main()
