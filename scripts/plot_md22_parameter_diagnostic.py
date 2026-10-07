from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style

DATASETS = [
    "DHA",
    "AT-AT",
    "stachyose",
    "AT-AT-CG-CG",
    "buckyball-catcher",
    "double-walled-nanotube",
]
TITLES = {
    "stachyose": "Stach",
    "AT-AT-CG-CG": "AT-CG",
    "buckyball-catcher": "Bucky",
    "double-walled-nanotube": "DWNT",
}
METRIC = "raw_energy_rmse_per_atom"
SERIES = {
    "vecchia_fixed": ("fixed", "vecchia", "Vecchia GP", "vecchia", "s", "--", False),
    "tera_fixed": ("fixed", "tera_batched", "TERA", "tera_batched", "o", "-", False),
    "tera_trained": ("trained", "tera_batched", "TERA (trained)", "tera_batched", "o", "--", True),
    "lite_fixed": ("fixed", "lite", "LITE", "lite", "D", "-", False),
    "lite_refit": ("refit", "lite", "LITE (trained)", "lite", "D", "--", True),
}


def csv_names(value):
    return list(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def collect(root, datasets, seeds, warmstart_folder, tera_trained_folder=None):
    frames, missing = [], []
    sources = [
        ("fixed", "fixed_lite_vecchia", {"lite", "vecchia"}),
        ("fixed", "fixed_tera", {"tera_batched"}),
        ("refit", warmstart_folder, {"lite"}),
    ]
    if tera_trained_folder:
        sources.append(("trained", tera_trained_folder, {"tera_batched"}))
    required = {"dataset", "method", "seed", "m", "status", METRIC, "split_id"}
    for dataset in datasets:
        for seed in seeds:
            for mode, folder, expected in sources:
                path = root / dataset / f"seed{seed}" / folder / "ablation_results.csv"
                if not path.exists():
                    missing.append(
                        dict(dataset=dataset, seed=seed, parameter_mode=mode, source_csv=str(path))
                    )
                    continue
                frame = pd.read_csv(path)
                if required.difference(frame):
                    raise ValueError(
                        f"Missing columns in {path}: {sorted(required.difference(frame))}"
                    )
                if (
                    not frame.dataset.eq(dataset).all()
                    or not pd.to_numeric(frame.seed).eq(seed).all()
                ):
                    raise ValueError(f"CSV dataset/seed does not match its folder: {path}")
                if not set(frame.method).issubset(expected):
                    raise ValueError(f"Unexpected methods in {path}")
                for column in ("train_epochs", "train_steps"):
                    if (
                        mode == "fixed"
                        and column in frame
                        and pd.to_numeric(frame[column]).fillna(0).ne(0).any()
                    ):
                        raise ValueError(f"Fixed-parameter results include training: {path}")
                if mode == "trained" and not {"train_epochs", "train_steps"}.issubset(frame):
                    raise ValueError(
                        f"Trained TERA results must include train_epochs and train_steps: {path}"
                    )
                if mode in {"refit", "trained"} and {"train_epochs", "train_steps"}.issubset(frame):
                    trained = pd.to_numeric(frame.train_epochs).fillna(0).gt(0) | pd.to_numeric(
                        frame.train_steps
                    ).fillna(0).gt(0)
                    if not trained.all():
                        raise ValueError(f"Trained results have zero training budget: {path}")
                frame = frame.copy()
                frame["parameter_mode"] = mode
                frame["source_csv"] = str(path)
                frames.append(frame)
    if not frames:
        raise ValueError("No ablation_results.csv files found under the selected root.")
    df = pd.concat(frames, ignore_index=True)
    keys = ["dataset", "seed", "method", "m", "parameter_mode"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate dataset/seed/method/m/mode results.")
    for column in ("m", "seed", METRIC):
        df[column] = pd.to_numeric(df[column], errors="raise")
    if not df.m.gt(0).all():
        raise ValueError("m must be positive.")
    good = df[df.status.eq("ok")].copy()
    if good.split_id.isna().any():
        raise ValueError("Successful diagnostic rows must include split_id.")
    for column in ("split_id", "kernel", "use_ard", "dtype", "preprocessing_version", "x_scale"):
        if (
            column in good
            and (good.groupby(["dataset", "seed"])[column].nunique(dropna=True) > 1).any()
        ):
            raise ValueError(f"The compared results mix {column} within a dataset/seed.")
    for column in ("energy_noise_fraction", "force_noise_fraction"):
        if column in good and pd.to_numeric(good[column], errors="raise").fillna(0).ne(0).any():
            raise ValueError("This plot expects clean observations.")
    return df, pd.DataFrame(missing, columns=["dataset", "seed", "parameter_mode", "source_csv"])


def paired_results(good):
    keys = ["dataset", "seed", "m"]
    fixed = good[good.parameter_mode.eq("fixed") & good.method.eq("lite")][keys + [METRIC]].rename(
        columns={METRIC: "fixed_rmse"}
    )
    refit = good[good.parameter_mode.eq("refit") & good.method.eq("lite")][keys + [METRIC]].rename(
        columns={METRIC: "refit_rmse"}
    )
    paired = fixed.merge(refit, on=keys, how="inner", validate="one_to_one")
    paired["refit_minus_fixed_rmse"] = paired.refit_rmse - paired.fixed_rmse
    paired["rmse_change_percent"] = (
        100 * paired.refit_minus_fixed_rmse / paired.fixed_rmse.replace(0, np.nan)
    )
    return paired


def limits(good, dataset):
    y = good.loc[good.dataset.eq(dataset), METRIC].to_numpy(float)
    y = y[np.isfinite(y)]
    if not len(y):
        return None
    low, high = float(y.min()), float(y.max())
    span = max(high - low, abs(high) * 0.05, 1e-10)
    return max(0, low - 0.15 * span), high + 0.30 * span


def draw_figure(good, df, datasets, outpath, kind, errorbar, *, limits_frame=None):
    ncols = min(3, len(datasets))
    nrows = (len(datasets) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, squeeze=False, figsize=(3.6 * ncols, 2.8 * nrows), layout="constrained"
    )
    series = (
        ["vecchia_fixed", "tera_fixed", "lite_fixed"]
        if kind == "fixed"
        else ["tera_fixed", "tera_trained", "lite_fixed", "lite_refit"]
    )
    labels = {
        "vecchia_fixed": "Vecchia GP",
        "tera_fixed": "TERA" if kind == "fixed" else "TERA (fixed)",
        "tera_trained": "TERA (trained)",
        "lite_fixed": "LITE" if kind == "fixed" else "LITE (fixed)",
        "lite_refit": "LITE (trained)",
    }
    legend_drawn = False
    for ax, dataset in zip(axes.flat, datasets):
        m_values = sorted(df.loc[df.dataset.eq(dataset), "m"].unique())
        positions = np.arange(len(m_values), dtype=float)
        for name in series:
            mode, method, _, color_key, marker, linestyle, hollow = SERIES[name]
            part = good[
                good.dataset.eq(dataset) & good.parameter_mode.eq(mode) & good.method.eq(method)
            ]
            if part.empty:
                continue
            stats = part.groupby("m")[METRIC].agg(["mean", errorbar]).reindex(m_values)
            color = METHOD_COLORS[color_key]
            ax.errorbar(
                positions,
                stats["mean"].to_numpy(float),
                yerr=stats[errorbar].fillna(0).to_numpy(float),
                color=color,
                marker=marker,
                markerfacecolor="white" if hollow else color,
                markeredgecolor=color,
                markeredgewidth=1,
                markersize=6,
                linestyle=linestyle,
                linewidth=2,
                elinewidth=1,
                capsize=3,
                zorder=7 if name == "lite_refit" else (6 if method == "lite" else 3),
            )
        ax.set_title(TITLES.get(dataset, dataset))
        ax.set_xlabel("Conditioning set size")
        ax.set_ylabel("Test RMSE")
        ax.set_xticks(positions, [f"{m:g}" for m in m_values])
        ax.tick_params(axis="x", labelsize=9)
        if len(m_values):
            ax.set_xlim(-0.35, len(m_values) - 0.65)
        ylim = limits(good if limits_frame is None else limits_frame, dataset)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)
        if not ax.has_data():
            ax.text(
                0.5,
                0.5,
                "No successful measurements",
                ha="center",
                transform=ax.transAxes,
                fontsize=9,
            )
        elif not legend_drawn:
            handles = []
            for name in series:
                mode, method, _, color_key, marker, linestyle, hollow = SERIES[name]
                if not (good.parameter_mode.eq(mode) & good.method.eq(method)).any():
                    continue
                color = METHOD_COLORS[color_key]
                handles.append(
                    Line2D(
                        [],
                        [],
                        label=labels[name],
                        color=color,
                        marker=marker,
                        markerfacecolor="white" if hollow else color,
                        markeredgecolor=color,
                        markersize=6,
                        linestyle=linestyle,
                        linewidth=2,
                    )
                )
            ax.legend(handles=handles, loc="best", frameon=False, fontsize=8.5)
            legend_drawn = True
    for ax in list(axes.flat)[len(datasets) :]:
        ax.set_visible(False)
    for suffix in ("png", "pdf"):
        fig.savefig(
            outpath.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.03
        )
    plt.close(fig)


@plt.rc_context()
def run(root, outdir, *, datasets, seeds, warmstart_folder, errorbar, tera_trained_folder=None):
    setup_style()
    df, missing = collect(root, datasets, seeds, warmstart_folder, tera_trained_folder)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "combined_results.csv", index=False)
    missing.to_csv(outdir / "missing_results.csv", index=False)
    good = df[df.status.eq("ok") & np.isfinite(df[METRIC])].copy()
    if good.empty:
        raise ValueError("No successful finite RMSE results.")
    summary = (
        good.groupby(["parameter_mode", "dataset", "method", "m"])[METRIC]
        .agg(["mean", "std", "sem", "count"])
        .reset_index()
    )
    summary.to_csv(outdir / "rmse_summary.csv", index=False)
    df.groupby(["parameter_mode", "dataset", "method", "m", "status"]).size().rename(
        "runs"
    ).reset_index().to_csv(outdir / "coverage.csv", index=False)
    paired = paired_results(good)
    paired.to_csv(outdir / "paired_lite_results.csv", index=False)
    if not paired.empty:
        stats = paired.groupby(["dataset", "m"])[
            ["fixed_rmse", "refit_rmse", "refit_minus_fixed_rmse", "rmse_change_percent"]
        ].agg(["mean", "sem", "count"])
        stats.columns = [f"{metric}_{stat}" for metric, stat in stats.columns]
        stats.reset_index().to_csv(outdir / "paired_lite_summary.csv", index=False)
    if good.parameter_mode.eq("fixed").any():
        draw_figure(good, df, datasets, outdir / "md22_fixed_parameters", "fixed", errorbar)
    if not paired.empty:
        matched = good[good.method.eq("lite")].merge(
            paired[["dataset", "seed", "m"]],
            on=["dataset", "seed", "m"],
            how="inner",
            validate="many_to_one",
        )
        reference = good[
            good.parameter_mode.isin(["fixed", "trained"]) & good.method.eq("tera_batched")
        ]
        comparison = pd.concat([reference, matched], ignore_index=True)
        draw_figure(
            comparison,
            df,
            datasets,
            outdir / "md22_warmstart_comparison",
            "refit",
            errorbar,
            limits_frame=good,
        )
    else:
        print(
            "Warm-start comparison omitted: successful matched fixed/refit LITE results are required."
        )
    print(f"Saved CSVs and figures to {outdir}")
    if len(missing):
        print(f"Missing {len(missing)} expected result files; see missing_results.csv.")
    return df, summary


def main():
    parser = argparse.ArgumentParser(
        description="Merge and plot existing fixed/refit MD22 parameter diagnostics; does not run experiments."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--outdir")
    parser.add_argument("--datasets", type=csv_names, default=DATASETS)
    parser.add_argument("--seeds", default="1,27,42,86,99")
    parser.add_argument("--warmstart-folder", default="warmstart_lite_m_sweep_no_validation")
    parser.add_argument(
        "--tera-trained-folder",
        help="Per-dataset/seed folder containing ablation_results.csv from TERA trained separately at each m.",
    )
    parser.add_argument("--errorbar", choices=["sem", "std"], default="sem")
    args = parser.parse_args()
    names = [*args.datasets, args.warmstart_folder]
    if args.tera_trained_folder:
        names.append(args.tera_trained_folder)
        if args.tera_trained_folder in {"fixed_lite_vecchia", "fixed_tera", args.warmstart_folder}:
            parser.error(
                "The trained TERA folder must be different from the fixed and LITE folders."
            )
    for name in names:
        if Path(name).name != name or name in {".", ".."}:
            parser.error("Dataset and result folder names must not contain path components.")
    try:
        seeds = list(dict.fromkeys(int(item) for item in csv_names(args.seeds)))
    except ValueError:
        parser.error("Use comma-separated integer seeds.")
    if not args.datasets or not seeds:
        parser.error("Specify at least one dataset and seed.")
    root = Path(args.root).resolve()
    outdir = Path(args.outdir).resolve() if args.outdir else root / "figures_parameter_diagnostic"
    run(
        root,
        outdir,
        datasets=args.datasets,
        seeds=seeds,
        warmstart_folder=args.warmstart_folder,
        errorbar=args.errorbar,
        tera_trained_folder=args.tera_trained_folder,
    )


if __name__ == "__main__":
    main()
