import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from lite.plotting.io import read_results
from lite.plotting.palette import DIRECTION_COLORS as COLORS
from lite.plotting.style import setup_gp_style

LABELS = {
    "Target": r"$\mathbf{u}_{ij}$",
    "Target + random": r"$(\mathbf{u}_{ij},\,\mathbf{D}_i\mathbf{a}_{ij})$",
    "LITE": r"$(\mathbf{u}_{ij},\,\mathbf{v}_{ij})$ (Ours)",
}


def plot(csv, outdir):
    data = read_results(csv)
    if not (data.status == "ok").all():
        raise ValueError("Failed experiment rows present.")
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    setup_gp_style()
    for d, df in data.groupby("d"):
        fig, ax = plt.subplots(figsize=(8.0, 4.3), layout="constrained")
        for method, color in COLORS.items():
            if method == "Conditional":
                continue
            s = (
                df[df.method == method]
                .groupby("m")
                .mean_marginal_kl.agg(["mean", "min", "max"])
                .sort_index()
            )

            ax.plot(
                s.index,
                s["mean"].clip(lower=1e-16),
                label=LABELS[method],
                color=color,
                marker="o",
                linewidth=3,
                ms=8,
            )
            ax.fill_between(
                s.index,
                s["min"].clip(lower=1e-16),
                s["max"].clip(lower=1e-16),
                color=color,
                alpha=0.12,
            )
        ax.set(xlabel="Conditioning set size", ylabel="KL divergence", yscale="log")
        ax.set_xticks(sorted(df.m.unique()))
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.legend(frameon=False)
        for suffix in ("png", "pdf"):
            fig.savefig(out / f"directions_d{d}.{suffix}", dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--outdir", required=True)
    a = p.parse_args()
    plot(a.csv, a.outdir)


if __name__ == "__main__":
    main()
