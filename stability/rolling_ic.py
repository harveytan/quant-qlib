import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
from qlib.data import D

from pipeline.utils import prints


def compute_daily_ic(df_all: pd.DataFrame) -> pd.Series:
    """
    Compute daily Spearman IC from merged predictions + labels.
    df_all must contain: date, pred, label
    """
    ic_series = (
        df_all.groupby("date")
        .apply(lambda g: spearmanr(g["pred"], g["label"]).correlation)
        .sort_index()
    )
    return ic_series


def run_rolling_ic_monitor(
    instruments,
    start_date,
    end_date,
    pred_dir="stability_outputs/daily_predictions",
    window=20,
):
    """
    Full rolling IC monitor:
    - loads all daily prediction CSVs
    - merges with Qlib labels
    - computes daily IC
    - computes rolling IC mean + volatility
    - prints/logs summary
    """

    prints("[IC] Running rolling IC monitor...", level="info")

    pred_dir = Path(pred_dir)
    pred_files = sorted(pred_dir.glob("preds_*.csv"))

    if not pred_files:
        prints("[IC] No prediction files found. Skipping IC monitor.", level="warning")
        return None

    # Load all predictions
    df_all = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)
    df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

    # Load labels
    labels_all = (
        D.features(
            instruments=instruments,
            fields=["$ret_5d"],
            start_time=start_date,
            end_time=end_date,
        )
        .reset_index()
        .rename(columns={"instrument": "symbol", "$ret_5d": "label"})
    )

    labels_all["datetime"] = pd.to_datetime(labels_all["datetime"], errors="coerce")

    # Merge predictions + labels
    df_all = df_all.merge(
        labels_all[["datetime", "symbol", "label"]],
        left_on=["date", "symbol"],
        right_on=["datetime", "symbol"],
        how="left",
    )

    df_all = df_all.dropna(subset=["label"])

    if df_all.empty:
        prints("[IC] No valid rows after merging labels. Skipping.", level="warning")
        return None

    # Compute daily IC
    ic_series = compute_daily_ic(df_all)

    if len(ic_series) == 0:
        summary = {
            "last_date": None,
            "IC_last": None,
            "IC_20_last": None,
            "IC_vol_20_last": None,
            "n_alerts_total": 0,
        }
        prints(f"[IC] Rolling IC summary: {summary}", level="info")
        return summary

    # Latest IC
    ic_last = float(ic_series.iloc[-1])

    # Rolling window
    if len(ic_series) >= window:
        window_vals = ic_series.iloc[-window:]
        ic_20_last = float(np.nanmean(window_vals))
        ic_vol_20_last = float(np.nanstd(window_vals))
    else:
        ic_20_last = None
        ic_vol_20_last = None

    summary = {
        "last_date": str(df_all["date"].max().date()),
        "IC_last": ic_last,
        "IC_20_last": ic_20_last,
        "IC_vol_20_last": ic_vol_20_last,
        "n_alerts_total": 0,
    }

    prints(f"[IC] Rolling IC summary: {summary}", level="info")
    return summary

def run_rolling_ic_monitor_training(ic_series, out_dir, window=20):
    prints("[IC] Running training rolling IC monitor...", level="info")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ic_series is None or len(ic_series) == 0:
        summary = {
            "last_date": None,
            "IC_last": None,
            "IC_20_last": None,
            "IC_vol_20_last": None,
            "n_alerts_total": 0,
        }
        prints(f"[IC] Rolling IC summary: {summary}", level="info")
        return summary

    # Ensure datetime index
    try:
        last_idx = pd.to_datetime(ic_series.index.max())
    except Exception:
        last_idx = None

    ic_last = float(ic_series.iloc[-1])

    if len(ic_series) >= window:
        window_vals = ic_series.iloc[-window:]
        ic_20_last = float(np.nanmean(window_vals))
        ic_vol_20_last = float(np.nanstd(window_vals))
    else:
        ic_20_last = None
        ic_vol_20_last = None

    summary = {
        "last_date": str(last_idx.date()) if last_idx is not None else None,
        "IC_last": ic_last,
        "IC_20_last": ic_20_last,
        "IC_vol_20_last": ic_vol_20_last,
        "n_alerts_total": 0,
    }

    prints(f"[IC] Rolling IC summary: {summary}", level="info")
    return summary