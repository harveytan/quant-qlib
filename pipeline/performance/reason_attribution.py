# pipeline/performance/reason_attribution.py

from pathlib import Path
import pandas as pd
import numpy as np
from pipeline.utils import prints

ENTRY_LOG_PATH = Path("artifacts/entry_log.parquet")


HORIZONS = [5, 10, 20, 60]


def load_log():
    if not ENTRY_LOG_PATH.exists():
        raise FileNotFoundError(f"{ENTRY_LOG_PATH} not found")
    return pd.read_parquet(ENTRY_LOG_PATH)


def _valid_mask(df, h):
    filled_col = f"filled_{h}d"
    ret_col = f"ret_{h}d"
    if filled_col not in df.columns or ret_col not in df.columns:
        return df[[]].index == -1  # all False
    return (df[filled_col] == True) & df[ret_col].notna()


def summarize_reason_performance(df):
    results = {}

    reasons = df["reason"].unique()
    for reason in reasons:
        sub = df[df["reason"] == reason]
        reason_stats = {
            "n_trades": len(sub),
            "by_horizon": {},
        }

        for h in HORIZONS:
            mask = _valid_mask(sub, h)
            sub_h = sub[mask]
            ret_col = f"ret_{h}d"

            if len(sub_h) == 0:
                reason_stats["by_horizon"][h] = {
                    "n_valid": 0,
                    "hit_rate": np.nan,
                    "avg_return": np.nan,
                    "median_return": np.nan,
                    "std_return": np.nan,
                }
                continue

            rets = sub_h[ret_col]
            hit_rate = (rets > 0).mean()
            avg_ret = rets.mean()
            med_ret = rets.median()
            std_ret = rets.std(ddof=1)

            reason_stats["by_horizon"][h] = {
                "n_valid": len(sub_h),
                "hit_rate": hit_rate,
                "avg_return": avg_ret,
                "median_return": med_ret,
                "std_return": std_ret,
            }

        results[reason] = reason_stats

    return results


def print_reason_attribution(results):
    prints("\n===== REASON-CODE ATTRIBUTION =====")

    for reason, stats in results.items():
        prints(f"\nReason: {reason}")
        prints(f"Total trades: {stats['n_trades']}")

        for h, hs in stats["by_horizon"].items():
            n_valid = hs["n_valid"]
            if n_valid == 0:
                prints(f"  {h}d: n_valid=0 (no matured trades yet)")
                continue

            prints(
                f"  {h}d: n_valid={n_valid}, "
                f"hit={hs['hit_rate']:.3f}, "
                f"avg={hs['avg_return']:.4f}, "
                f"med={hs['median_return']:.4f}, "
                f"std={hs['std_return']:.4f}"
            )


def run_reason_attribution():
    df = load_log()
    if "reason" not in df.columns:
        raise KeyError("entry_log is missing 'reason' column")

    results = summarize_reason_performance(df)
    print_reason_attribution(results)