from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from lite.plotting.palette import METHOD_COLORS
from lite.plotting.style import setup_style


def _require(df, columns, source):
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise ValueError(f"{source}: missing columns {missing}")


def _numeric(df, columns):
    df = df.copy()
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _stats(df, keys, metrics, errorbar):
    result = df.groupby(keys, sort=True)[metrics].agg(["mean", errorbar, "count"])
    result.columns = [f"{metric}_{stat}" for metric, stat in result.columns]
    return result.reset_index()


def md22_runs(csv, dataset, tera_method, batches, seeds=None):
    df = pd.read_csv(csv)
    _require(
        df,
        [
            "dataset",
            "sweep",
            "method",
            "seed",
            "status",
            "batch_size",
            "wall_time_sec",
            "peak_mem_gb",
        ],
        "MD22",
    )
    df = df[
        df.dataset.eq(dataset) & df.sweep.eq("batch") & df.method.isin([tera_method, "lite"])
    ].copy()
    df = _numeric(df, ["seed", "batch_size", "wall_time_sec", "peak_mem_gb"])
    df = df[df.batch_size.isin(batches)]
    if seeds is not None:
        df = df[df.seed.isin(seeds)]
    if df.empty:
        raise ValueError("No matching MD22 batch sweep rows. Check dataset/method/batch selectors.")
    if df.duplicated(["method", "seed", "batch_size"]).any():
        raise ValueError("Duplicate MD22 method/seed/batch rows; use one experiment CSV.")
    valid = (
        df.status.eq("ok")
        & np.isfinite(df.wall_time_sec)
        & np.isfinite(df.peak_mem_gb)
        & df.wall_time_sec.gt(0)
        & df.peak_mem_gb.gt(0)
    )
    df["included"] = valid
    if not valid.all():
        warnings.warn(f"MD22: excluded {(~valid).sum()} failed or invalid runs; no costs invented.")
    if not valid.any():
        raise ValueError("No successful MD22 runs with positive total time and GPU memory.")
    return df[valid].copy(), df


def bo_runs(csv, benchmark, tera_method, m_values, budget=None, seeds=None):
    if budget is not None and budget < 1:
        raise ValueError("--bo-budget must be positive.")
    df = pd.read_csv(csv)
    times = ["fit_time_sec", "acq_time_sec", "eval_time_sec"]
    memory = ["fit_peak_memory_gb", "acq_peak_memory_gb", "iter_peak_memory_gb"]
    _require(
        df, ["benchmark", "method", "seed", "status", "eval_index", "regret"] + times + memory, "BO"
    )
    df = df[df.benchmark.eq(benchmark) & df.method.isin([tera_method, "lite"])].copy()
    if "sweep" in df:
        df = df[df.sweep.eq("m")]
    df = _numeric(df, ["seed", "eval_index", "regret"] + times + memory)
    if "sweep_value" in df:
        df["m"] = pd.to_numeric(df.sweep_value, errors="coerce")
    else:
        _require(df, ["lite_m", "tera_m"], "BO")
        df["m"] = pd.to_numeric(df.lite_m.where(df.method.eq("lite"), df.tera_m), errors="coerce")
    df = df[df.m.isin(m_values)]
    if seeds is not None:
        df = df[df.seed.isin(seeds)]
    if df.empty:
        raise ValueError("No matching BO m sweep rows. Check benchmark/method/m selectors.")
    if "budget" not in df and budget is None:
        raise ValueError(
            "Older BO CSV has no budget metadata. Specify --bo-budget to verify completion."
        )
    records = []
    for (method, seed, m), part in df.groupby(["method", "seed", "m"], sort=True):
        if "run_id" in part and part.run_id.nunique() > 1:
            raise ValueError(f"Multiple run IDs for {method}, seed={seed}, m={m}.")
        if "budget" in part:
            budgets = pd.to_numeric(part.budget, errors="coerce").unique()
            if (
                len(budgets) != 1
                or not np.isfinite(budgets[0])
                or budgets[0] < 1
                or budgets[0] != int(budgets[0])
            ):
                raise ValueError("Each BO run must have one positive integer budget.")
            expected = int(budgets[0])
            if budget is not None and expected != budget:
                raise ValueError(f"BO CSV budget {expected} differs from --bo-budget {budget}.")
        else:
            expected = budget
        successful = part[part.status.isin(["init", "ok"])].sort_values("eval_index")
        if successful.eval_index.duplicated().any():
            raise ValueError(f"Duplicate BO evaluation rows for {method}, seed={seed}, m={m}.")
        complete = (
            len(successful) == expected
            and np.array_equal(successful.eval_index.to_numpy(), np.arange(1, expected + 1))
            and part.status.isin(["init", "ok"]).all()
        )
        record = dict(
            method=method,
            seed=seed,
            m=m,
            budget=expected,
            completed=complete,
            included=False,
            status="incomplete",
            final_regret=np.nan,
            wall_time_sec=np.nan,
            peak_mem_gb=np.nan,
        )
        if complete:
            final = float(successful.regret.iloc[-1])
            elapsed = successful[times].to_numpy(dtype=float)
            peaks = successful[memory].to_numpy(dtype=float)
            valid = (
                np.isfinite(final)
                and final >= 0
                and np.isfinite(elapsed).all()
                and (elapsed >= 0).all()
                and elapsed.sum() > 0
                and np.isfinite(peaks).all()
                and (peaks >= 0).all()
                and peaks.max() > 0
            )
            record.update(included=bool(valid), status="ok" if valid else "invalid_metrics")
            if valid:
                record.update(
                    final_regret=final,
                    wall_time_sec=float(elapsed.sum()),
                    peak_mem_gb=float(peaks.max()),
                )
        records.append(record)
    audit = pd.DataFrame(records)
    if audit.budget.nunique() > 1:
        raise ValueError("BO configurations must use the same evaluation budget for this figure.")
    if not audit.included.all():
        warnings.warn(
            f"BO: excluded {(~audit.included).sum()} incomplete/invalid runs; see coverage CSV."
        )
    runs = audit[audit.included].copy()
    if runs.empty:
        raise ValueError("No completed BO runs with finite final regret and positive costs.")
    return runs, audit


def _coverage(audit, key, values, methods, seeds):
    planned = set(seeds) if seeds is not None else set(audit.seed.dropna())
    rows = []
    for method in methods:
        for value in values:
            part = audit[audit.method.eq(method) & audit[key].eq(value)]
            present = set(part.seed.dropna())
            included = set(part.loc[part.included, "seed"])
            rows.append(
                dict(
                    method=method,
                    **{key: value},
                    expected_runs=len(planned),
                    observed_runs=len(present),
                    included_runs=len(included),
                    missing_seeds=",".join(str(int(s)) for s in sorted(planned - present)),
                    excluded_seeds=",".join(str(int(s)) for s in sorted(present - included)),
                )
            )
    result = pd.DataFrame(rows)
    if (result.included_runs != result.expected_runs).any():
        warnings.warn(
            f"{key}: some configurations have missing/failed seeds. Means use included runs only."
        )
    return result


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.3)


def _series(agg, method, key, values, metric, errorbar):
    part = agg[agg.method.eq(method)].set_index(key).reindex(values)
    return (
        part[f"{metric}_mean"].to_numpy(dtype=float),
        part[f"{metric}_{errorbar}"].fillna(0).to_numpy(dtype=float),
    )


@plt.rc_context()
def plot_scaling(
    md22_csv,
    bo_csv=None,
    out_path=None,
    *,
    md22_dataset="DHA",
    bo_benchmark="levy_800",
    md22_tera_method="tera_batched",
    bo_tera_method="tera_batched",
    batch_sizes=(64, 256, 1024, 4096),
    m_values=(20, 30, 50, 100),
    bo_budget=None,
    seeds=None,
    errorbar="sem",
    cost_scale="log",
    regret_scale="linear",
    annotate_m=True,
    connect_m=True,
):
    setup_style()
    batch_sizes, m_values = list(batch_sizes), list(m_values)
    if (
        errorbar not in {"std", "sem"}
        or cost_scale not in {"linear", "log"}
        or regret_scale not in {"linear", "log"}
    ):
        raise ValueError("Invalid errorbar or scale setting.")
    md, md_audit = md22_runs(md22_csv, md22_dataset, md22_tera_method, batch_sizes, seeds)
    if bo_csv is None:
        raise ValueError("Specify bo_csv.")
    bo, bo_audit = bo_runs(bo_csv, bo_benchmark, bo_tera_method, m_values, bo_budget, seeds)
    outcome_metric, outcome_label, tradeoff_prefix = "final_regret", "Final regret", "bo"
    time_label = "Wall-clock time (s)"
    md_methods, bo_methods = [md22_tera_method, "lite"], [bo_tera_method, "lite"]
    md_cov = _coverage(md_audit, "batch_size", batch_sizes, md_methods, seeds)
    bo_cov = _coverage(bo_audit, "m", m_values, bo_methods, seeds)
    md_agg = _stats(md, ["method", "batch_size"], ["wall_time_sec", "peak_mem_gb"], errorbar)
    bo_agg = _stats(bo, ["method", "m"], [outcome_metric, "wall_time_sec", "peak_mem_gb"], errorbar)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.6), layout="constrained")
    time_ax = axes[0]
    memory_ax = time_ax.twinx()
    time_ax.set_zorder(memory_ax.get_zorder() + 1)
    time_ax.patch.set_visible(False)
    positions = np.arange(len(batch_sizes), dtype=float)
    for j, method in enumerate(md_methods):
        color = METHOD_COLORS[method]
        is_lite = method == "lite"
        mem, mem_err = _series(md_agg, method, "batch_size", batch_sizes, "peak_mem_gb", errorbar)
        time, time_err = _series(
            md_agg, method, "batch_size", batch_sizes, "wall_time_sec", errorbar
        )
        offset = (j - 0.5) * 0.34
        valid = np.isfinite(mem)
        memory_ax.bar(
            positions[valid] + offset,
            mem[valid],
            width=0.34,
            color=color,
            alpha=0.48,
            edgecolor=color,
            linewidth=0.6,
            yerr=mem_err[valid],
            capsize=3,
            error_kw={"elinewidth": 1, "capthick": 1},
            zorder=3,
        )
        time_ax.errorbar(
            positions,
            time,
            yerr=time_err,
            color=color,
            marker="D" if is_lite else "*",
            markersize=6 if is_lite else 4.5,
            linewidth=2.2 if is_lite else 2,
            elinewidth=1,
            capsize=3,
            zorder=6 if is_lite else 4,
        )
    time_ax.set_xticks(positions, [f"{v:g}" for v in batch_sizes])
    time_ax.set_xlim(-0.6, len(batch_sizes) - 0.4)
    time_ax.set_xlabel("Training and prediction batch size", fontsize=10)
    time_ax.set_ylabel("Wall-clock time (s)", fontsize=11)
    memory_ax.set_ylabel("Peak GPU memory (GiB)", fontsize=11)
    _style(time_ax)
    memory_ax.spines[["top", "left"]].set_visible(False)
    memory_ax.grid(False)
    time_ax.set_yscale(cost_scale)
    memory_ax.set_yscale(cost_scale)
    time_ax.margins(y=0.25)
    memory_ax.margins(y=0.25)
    for ax, metric, xlabel in zip(
        axes[1:], ["wall_time_sec", "peak_mem_gb"], [time_label, "Peak GPU memory (GiB)"]
    ):
        for method in bo_methods:
            is_lite = method == "lite"
            x, xerr = _series(bo_agg, method, "m", m_values, metric, errorbar)
            y, yerr = _series(bo_agg, method, "m", m_values, outcome_metric, errorbar)
            if regret_scale == "log" and np.any(y[np.isfinite(y)] <= 0):
                plt.close(fig)
                raise ValueError(
                    "Zero error cannot be plotted on a log axis. Use --regret-scale linear."
                )
            ax.errorbar(
                x,
                y,
                xerr=xerr,
                yerr=yerr,
                color=METHOD_COLORS[method],
                marker="D" if is_lite else "*",
                markersize=6 if is_lite else 4.5,
                linestyle="-" if connect_m else "none",
                linewidth=2.2 if is_lite else 2,
                elinewidth=1,
                capsize=3,
                zorder=6 if is_lite else 4,
            )
            if annotate_m:
                for m, xv, yv in zip(m_values, x, y):
                    if np.isfinite(xv) and np.isfinite(yv):
                        ax.annotate(
                            f"{m:g}",
                            (xv, yv),
                            xytext=(4, -12 if is_lite else 7),
                            textcoords="offset points",
                            fontsize=8,
                            color=METHOD_COLORS[method],
                            zorder=7,
                        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(outcome_label)
        ax.set_xscale(cost_scale)
        ax.set_yscale(regret_scale)
        ax.margins(x=0.12, y=0.18)
        _style(ax)
    low = min(ax.get_ylim()[0] for ax in axes[1:])
    high = max(ax.get_ylim()[1] for ax in axes[1:])
    for ax in axes[1:]:
        ax.set_ylim(low, high)
    methods = list(dict.fromkeys([md22_tera_method, bo_tera_method, "lite"]))
    handles = [
        Line2D(
            [],
            [],
            color=METHOD_COLORS[m],
            marker="D" if m == "lite" else "*",
            markersize=6 if m == "lite" else 4.5,
            linewidth=2,
            label="LITE" if m == "lite" else ("TERA (batched)" if m == "tera_batched" else "TERA"),
        )
        for m in methods
    ]
    metric_handles = [
        Line2D([], [], color="0.35", linewidth=2, label="Time (lines)"),
        Patch(facecolor="0.6", alpha=0.48, label="Memory (bars)"),
    ]
    time_ax.legend(
        handles=metric_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize=8,
        handlelength=1.6,
        columnspacing=0.8,
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=len(handles),
        frameon=False,
        columnspacing=1.2,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(out_path.with_suffix(f".{extension}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    for suffix, table in [
        ("md22_summary", md_agg),
        (f"{tradeoff_prefix}_summary", bo_agg),
        ("md22_coverage", md_cov),
        (f"{tradeoff_prefix}_coverage", bo_cov),
        (f"{tradeoff_prefix}_runs", bo_audit),
    ]:
        table.to_csv(out_path.with_name(f"{out_path.stem}_{suffix}.csv"), index=False)
    return md_agg, bo_agg


def _integer_values(value):
    try:
        values = list(dict.fromkeys(int(v.strip()) for v in value.split(",")))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use comma-separated integers.") from exc
    return values


def _positive_values(value):
    values = _integer_values(value)
    if not values or min(values) < 1:
        raise argparse.ArgumentTypeError("Use comma-separated positive integers.")
    return values


def main(argv=None):
    p = argparse.ArgumentParser(description="MD22 batch costs and BO conditioning-size tradeoffs.")
    p.add_argument("--md22-csv", required=True)
    p.add_argument("--bo-csv", required=True)
    p.add_argument("--out", required=True, help="Output filename stem or .png/.pdf path.")
    p.add_argument("--md22-dataset", default="DHA")
    p.add_argument("--bo-benchmark", default="levy_800")
    for domain in ("md22", "bo"):
        p.add_argument(
            f"--{domain}-tera-method", choices=["tera", "tera_batched"], default="tera_batched"
        )
    p.add_argument("--batch-sizes", type=_positive_values, default="64,256,1024,4096")
    p.add_argument("--m-values", type=_positive_values, default="20,30,50,100")
    p.add_argument("--bo-budget", type=int, help="Required for older CSVs without budget metadata.")
    p.add_argument("--seeds", type=_integer_values)
    p.add_argument("--errorbar", choices=["std", "sem"], default="sem")
    p.add_argument("--cost-scale", choices=["linear", "log"], default="log")
    p.add_argument("--regret-scale", choices=["linear", "log"], default="linear")
    p.add_argument("--annotate-m", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--connect-m", action=argparse.BooleanOptionalAction, default=True)
    args = vars(p.parse_args(argv))
    args["out_path"] = args.pop("out")
    plot_scaling(**args)


if __name__ == "__main__":
    main()
