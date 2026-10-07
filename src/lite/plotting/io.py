from pathlib import Path

import pandas as pd

from lite.methods.names import normalize_method_name


def normalize_results(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    if "method" in frame:
        frame["method"] = frame["method"].map(normalize_method_name)
    for statistic in ("mean", "var"):
        old = f"maxabs_{statistic}_diff_to_exact_dgp"
        new = f"maxabs_{statistic}_diff_to_standard_dgp"
        if old in frame and new not in frame:
            frame = frame.rename(columns={old: new})
    return frame


def read_results(path: str | Path) -> pd.DataFrame:
    return normalize_results(pd.read_csv(path))
