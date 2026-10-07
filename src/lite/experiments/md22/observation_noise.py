"""Observation-noise settings and result labels."""

import math

NOISE_COLUMNS = ["energy_noise_fraction", "force_noise_fraction"]


def noise_fractions(common=0.0, energy=None, force=None):
    values = (common if energy is None else energy, common if force is None else force)
    if not math.isfinite(common) or common < 0:
        raise ValueError("observation_noise_fraction must be finite and nonnegative")
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Energy and force noise fractions must be finite and nonnegative")
    return tuple(float(v) for v in values)


def config_noise_fractions(cfg):
    return noise_fractions(
        cfg.observation_noise_fraction, cfg.energy_noise_fraction, cfg.force_noise_fraction
    )


def noise_tag(energy, force):
    if energy == force:
        return f"noise{100 * energy:g}pct"
    return f"energy{100 * energy:g}pct_force{100 * force:g}pct"


def noise_condition_frame(df):
    import pandas as pd

    out = df.copy()
    common = (
        pd.to_numeric(out["observation_noise_fraction"], errors="raise")
        if "observation_noise_fraction" in out
        else pd.Series(0.0, index=out.index)
    )
    for column in NOISE_COLUMNS:
        out[column] = (
            pd.to_numeric(out[column], errors="raise").fillna(common) if column in out else common
        )
        if not out[column].map(lambda v: math.isfinite(v) and v >= 0).all():
            raise ValueError(f"Missing or invalid {column} in results")
    return out


def require_single_noise_condition(df):
    if len(noise_condition_frame(df)[NOISE_COLUMNS].drop_duplicates()) > 1:
        raise ValueError(
            "Select one energy/force noise condition. Use plot_md22_tradeoff.py for multiple conditions."
        )
