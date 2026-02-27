# pipeline/performance/daily_summary.py

from pathlib import Path
import pandas as pd
from pipeline.utils import prints

SAFE_PATH = Path("artifacts/safe_entries.parquet")
DRIFT_SUMMARY_PATH = Path("stability_outputs/feature_drift/feature_drift_alerts_latest.json")
IC_SUMMARY_PATH = Path("stability_outputs/ic/rolling_ic_latest.json")


def load_safe():
    if not SAFE_PATH.exists():
        prints("No SAFE entries found for today.")
        return None
    return pd.read_parquet(SAFE_PATH)


def load_json(path):
    if not path.exists():
        return None
    try:
        import json
        return json.load(open(path, "r"))
    except:
        return None


def summarize_safe(df):
    summary = {}

    summary["n_trades"] = len(df)
    summary["n_long"] = (df["direction"] == "LONG").sum()
    summary["n_short"] = (df["direction"] == "SHORT").sum()

    summary["reason_counts"] = df["reason"].value_counts().to_dict()
    summary["momentum_counts"] = df["momentum"].value_counts().to_dict()
    summary["crash_counts"] = df["crash"].value_counts().to_dict()

    return summary


def print_safe_summary(s):
    prints("\n=== SAFE TRADES SUMMARY ===")
    prints(f"Total SAFE trades: {s['n_trades']}")
    prints(f"Long: {s['n_long']}   Short: {s['n_short']}")

    prints("\n--- Reason Codes ---")
    for r, c in s["reason_counts"].items():
        prints(f"{r}: {c}")

    prints("\n--- Momentum Buckets ---")
    for m, c in s["momentum_counts"].items():
        prints(f"{m}: {c}")

    prints("\n--- Crash Flags ---")
    for c, n in s["crash_counts"].items():
        prints(f"{c}: {n}")


def print_drift_summary(drift):
    prints("\n=== DRIFT SUMMARY ===")
    if drift is None:
        prints("No drift summary available.")
        return

    prints(f"Date: {drift.get('date')}")
    prints(f"Features monitored: {drift.get('n_features')}")
    prints(f"Drift alerts: {drift.get('n_alerts')}")
    prints(f"Max PSI: {drift.get('max_psi'):.3f}")
    prints(f"Max KS: {drift.get('max_ks'):.3f}")


def print_ic_summary(ic):
    prints("\n=== IC SUMMARY ===")
    if ic is None:
        prints("No IC summary available.")
        return

    prints(f"Last IC: {ic.get('IC_last')}")
    prints(f"IC_20_last: {ic.get('IC_20_last')}")
    prints(f"Vol_20_last: {ic.get('IC_vol_20_last')}")
    prints(f"Total alerts: {ic.get('n_alerts_total')}")


def run_daily_summary():
    prints("\n==============================")
    prints("      DAILY SUMMARY REPORT")
    prints("==============================")

    # SAFE entries
    safe_df = load_safe()
    if safe_df is not None:
        safe_summary = summarize_safe(safe_df)
        print_safe_summary(safe_summary)

    # Drift
    drift = load_json(DRIFT_SUMMARY_PATH)
    print_drift_summary(drift)

    # IC
    ic = load_json(IC_SUMMARY_PATH)
    print_ic_summary(ic)

    prints("\n=== END OF DAILY SUMMARY ===\n")