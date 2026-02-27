# pipeline/performance/dashboard.py

from pathlib import Path
import pandas as pd

from pipeline.performance.metrics import (
    hit_rate,
    avg_return,
    long_short_split,
    bucket_monotonicity,
)

ENTRY_LOG_PATH = Path("artifacts/entry_log.parquet")


def load_log():
    if not ENTRY_LOG_PATH.exists():
        raise FileNotFoundError("entry_log.parquet not found.")
    return pd.read_parquet(ENTRY_LOG_PATH)


def compute_dashboard():
    df = load_log()
    # Use the existing direction column
    if "direction" not in df.columns:
        raise KeyError("entry_log is missing 'direction' column")

    long_df, short_df = long_short_split(df)

    horizons = [5, 10, 20, 60]

    dashboard = {
        "total_trades": len(df),
        "long_trades": len(long_df),
        "short_trades": len(short_df),
        "hit_rates": {},
        "avg_returns": {},
        "score_monotonicity": {},
    }

    for h in horizons:
        dashboard["hit_rates"][h] = {
            "all": hit_rate(df, h),
            "long": hit_rate(long_df, h),
            "short": hit_rate(short_df, h),
        }
        dashboard["avg_returns"][h] = {
            "all": avg_return(df, h),
            "long": avg_return(long_df, h),
            "short": avg_return(short_df, h),
        }
        dashboard["score_monotonicity"][h] = bucket_monotonicity(df, h)

    return dashboard

def print_dashboard(d):
    print("\n===== PERFORMANCE DASHBOARD =====")
    print(f"Total trades: {d['total_trades']}")
    print(f"Long trades:  {d['long_trades']}")
    print(f"Short trades: {d['short_trades']}")

    print("\n--- Hit Rates ---")
    for h, vals in d["hit_rates"].items():
        print(f"{h}d: all={vals['all']:.3f}, long={vals['long']:.3f}, short={vals['short']:.3f}")

    print("\n--- Average Returns ---")
    for h, vals in d["avg_returns"].items():
        print(f"{h}d: all={vals['all']:.4f}, long={vals['long']:.4f}, short={vals['short']:.4f}")

    print("\n--- Score Monotonicity (mean fwd return per bucket) ---")
    for h, series in d["score_monotonicity"].items():
        print(f"\n{h}d:")
        print(series.to_string())