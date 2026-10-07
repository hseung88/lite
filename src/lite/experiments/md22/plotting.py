from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lite.experiments.md22.io import read_results
from lite.experiments.md22.observation_noise import require_single_noise_condition
from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style

METHOD_ORDER = [
    "standard_gp",
    "vecchia",
    "ddsvgp",
    "dsoftki",
    "tera",
    "tera_batched",
    "lite",
    "lite_normalized",
]

METHOD_LABELS = {
    "standard_gp": "Standard GP",
    "vecchia": "Vecchia GP",
    "ddsvgp": "DDSVGP",
    "dsoftki": "DSoftKI",
    "tera": "TERA",
    "tera_batched": "TERA",
    "lite": "LITE",
    "lite_normalized": "LITE (normalized)",
}

MARKERS = {
    "standard_gp": "o",
    "vecchia": "s",
    "ddsvgp": "^",
    "dsoftki": "v",
    "tera": "D",
    "tera_batched": "D",
    "lite": "P",
    "lite_normalized": "X",
}

LINESTYLES = {
    "standard_gp": "-",
    "vecchia": "--",
    "ddsvgp": ":",
    "dsoftki": "-.",
    "tera": "-",
    "tera_batched": "-",
    "lite": "-",
    "lite_normalized": "--",
}


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    require_single_noise_condition(df)
    out = df.copy()
    if "status" in out.columns:
        out = out[out["status"].astype(str).isin(["ok", "external_csv"])]
    for col in [
        "normalized_energy_rmse_per_atom",
        "raw_energy_rmse_per_atom",
        "wall_time_sec",
        "peak_mem_gb",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["raw_energy_rmse_per_atom"])


def _aggregate(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["dataset", "method"]
    vals = [
        "normalized_energy_rmse_per_atom",
        "raw_energy_rmse_per_atom",
        "wall_time_sec",
        "peak_mem_gb",
    ]
    agg = df.groupby(keys, as_index=False)[vals].agg(["mean", "std", "count"])
    agg.columns = ["_".join([c for c in col if c]) for col in agg.columns.to_flat_index()]
    agg = agg.rename(columns={"dataset_": "dataset", "method_": "method"})
    return agg


def _ordered(values: Iterable[str]) -> list[str]:
    existing = list(dict.fromkeys(values))
    return [m for m in METHOD_ORDER if m in existing] + [
        m for m in existing if m not in METHOD_ORDER
    ]


@plt.rc_context()
def plot_md22_results(
    csv_path: str | Path, out_path: str | Path, *, title: str | None = None
) -> None:
    setup_style()
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = _clean(read_results(csv_path))
    if df.empty:
        raise ValueError(f"No successful rows found in {csv_path}")
    agg = _aggregate(df)
    datasets = list(dict.fromkeys(df["dataset"].astype(str)))
    display_names = {
        "DHA": "DHA",
        "AT-AT": "AT-AT",
        "stachyose": "Stach",
        "AT-AT-CG-CG": "AT-CG",
        "buckyball-catcher": "Bucky",
        "buckyball_catcher": "Bucky",
        "double-walled-nanotube": "DWNT",
        "double_walled_nanotube": "DWNT",
        "Ac-Ala3-NHMe": "Ac-Ala3",
    }
    dataset_order = ["Ac-Ala3", "DHA", "AT-AT", "Stach", "AT-CG", "Bucky", "DWNT"]
    rank = {name: i for i, name in enumerate(dataset_order)}
    datasets.sort(key=lambda d: rank.get(display_names.get(d, d), len(rank)))
    labels = [display_names.get(d, d) for d in datasets]
    methods = _ordered(df["method"].astype(str))
    x = np.arange(len(datasets), dtype=float)
    width = min(0.12, 0.8 / max(len(methods), 1))
    offsets = (np.arange(len(methods)) - (len(methods) - 1) / 2.0) * width

    panels = [
        ("raw_energy_rmse_per_atom", "Test RMSE"),
        ("wall_time_sec", "Wall-clock time (s)"),
        ("peak_mem_gb", "Peak memory (GB)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.6), constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=11)

    for ax, (metric, ylabel) in zip(axes, panels):
        for j, method in enumerate(methods):
            sub = agg[agg["method"].astype(str) == method]
            means = []
            errs = []
            for dataset in datasets:
                row = sub[sub["dataset"].astype(str) == dataset]
                if row.empty:
                    means.append(np.nan)
                    errs.append(np.nan)
                else:
                    means.append(float(row[f"{metric}_mean"].iloc[0]))
                    errs.append(
                        float(row[f"{metric}_std"].iloc[0]) if f"{metric}_std" in row else np.nan
                    )
            y = np.asarray(means, dtype=float)
            yerr = np.asarray(errs, dtype=float)
            yerr[~np.isfinite(yerr)] = 0.0
            ax.errorbar(
                x + offsets[j],
                y,
                yerr=yerr,
                marker=MARKERS.get(method, "o"),
                linestyle="none",
                capsize=5.0,
                linewidth=2.0,
                markersize=7.5 if method == "lite" else 7.0,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                zorder=5 if method == "lite" else 3,
            )

        for boundary in np.arange(len(datasets) - 1) + 0.5:
            ax.axvline(
                boundary,
                color="black",
                linewidth=1.0,
                alpha=0.6,
                zorder=0,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=0, ha="center")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.5)
        if metric in {"wall_time_sec", "peak_mem_gb"}:
            positive = df[metric][df[metric] > 0]
            if len(positive) > 0:
                ax.set_yscale("log")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.14),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_md22_training_curves(
    csv_path: str | Path, out_path: str | Path, *, title: str | None = None
) -> None:
    csv_path = Path(csv_path)
    out_path = Path(out_path)
    df = read_results(csv_path)
    require_single_noise_condition(df)
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(["ok", "external_csv"])]
    for col in ["step", "raw_energy_rmse_per_atom"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["dataset", "method", "step", "raw_energy_rmse_per_atom"])
    if df.empty:
        raise ValueError(f"No training-curve rows found in {csv_path}")

    datasets = list(dict.fromkeys(df["dataset"].astype(str)))
    methods = _ordered(df["method"].astype(str))
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.6), constrained_layout=True)
    axes = axes.ravel()
    if title:
        fig.suptitle(title, fontsize=11)

    for ax, dataset in zip(axes, datasets):
        sub_ds = df[df["dataset"].astype(str) == dataset].copy()
        for method in methods:
            sub = sub_ds[sub_ds["method"].astype(str) == method].copy()
            if sub.empty:
                continue
            grouped = (
                sub.groupby("step")["raw_energy_rmse_per_atom"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            grouped.columns = ["step", "mean", "std", "count"]
            x = grouped["step"].to_numpy(dtype=float)
            y = grouped["mean"].to_numpy(dtype=float)
            ystd = (grouped["std"].fillna(0.0) / np.sqrt(grouped["count"])).to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                linestyle=LINESTYLES.get(method, "-"),
                linewidth=2.0,
                label=METHOD_LABELS.get(method, method),
                color=METHOD_COLORS.get(method),
                zorder=5 if method == "lite" else 3,
            )
            if np.any(ystd > 0):
                ax.fill_between(x, y - ystd, y + ystd, alpha=0.15)
        ax.set_title(dataset)
        ax.set_xlabel("Training step", fontsize=11)
        ax.set_ylabel("Test RMSE", fontsize=11)
        ax.grid(True, linewidth=0.4, alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 6),
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    if out_path.suffix.lower() != ".pdf":
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
