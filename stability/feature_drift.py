import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from pathlib import Path
import json
from pipeline.utils import prints

# [DAILY] Feature drift summary: {'date': '2025-12-26', 'n_features': 12, 'n_alerts': 10, 'max_psi': 1.3676453113723086, 'max_ks': 0.7435897435897436}
# n_feature: total number of features monitored
# n_alerts: 10 out 12 features triggered drift alerts - huge number - not small drift - this is a regime shift or data distribution shift.
# PSI (population stability index) : is the most important drift metric:
#  - psi < 0.1 : no significant drift
#  - 01. - 0.25 : moderate drift
#  - >0.25 : significant drift
#  - > 1: catastrophic drift
# KS (Kolmogorov–Smirnov distance) measures distribution shape difference.
# Interpretation:
#  • 0.0–0.1 → similar distributions
#  • 0.1–0.2 → mild drift
#  • 0.2–0.3 → moderate drift
#  • > 0.3 → strong drift
#  • > 0.5 → severe drift
#  • > 0.7 → massive drift
# Your KS = 0.74 → this is extremely high.
# This means the shape of at least one feature’s distribution is completely different from training.


def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    quantiles = np.linspace(0, 1, buckets + 1)
    # Use expected distribution for bins
    expected_bins = np.quantile(expected, quantiles)
    expected_bins[0] = -np.inf
    expected_bins[-1] = np.inf

    expected_counts, _ = np.histogram(expected, expected_bins)
    actual_counts, _ = np.histogram(actual, expected_bins)

    expected_props = expected_counts / max(expected_counts.sum(), 1)
    actual_props = actual_counts / max(actual_counts.sum(), 1)

    mask = (expected_props > 0) & (actual_props > 0)
    if not mask.any():
        return np.nan

    psi = np.sum(
        (expected_props[mask] - actual_props[mask])
        * np.log(expected_props[mask] / actual_props[mask])
    )
    return float(psi)


def feature_drift_report(
    train_df: pd.DataFrame, daily_df: pd.DataFrame
) -> pd.DataFrame:
    report_rows = []

    common_cols = [c for c in train_df.columns if c in daily_df.columns]

    for col in common_cols:
        train_vals = train_df[col].astype(float)
        daily_vals = daily_df[col].astype(float)

        train_non_null = train_vals.dropna()
        daily_non_null = daily_vals.dropna()

        if train_non_null.empty or daily_non_null.empty:
            psi = np.nan
            ks_stat = np.nan
        else:
            psi = compute_psi(train_non_null.values, daily_non_null.values)
            ks_stat = ks_2samp(train_non_null.values, daily_non_null.values).statistic

        mean_diff = abs(train_non_null.mean() - daily_non_null.mean())
        var_diff = abs(train_non_null.var() - daily_non_null.var())
        missing_diff = abs(train_vals.isna().mean() - daily_vals.isna().mean())

        report_rows.append(
            {
                "feature": col,
                "psi": psi,
                "ks": ks_stat,
                "mean_diff": mean_diff,
                "var_diff": var_diff,
                "missing_diff": missing_diff,
            }
        )

    return pd.DataFrame(report_rows).set_index("feature")


def drift_alerts(drift_df: pd.DataFrame) -> pd.DataFrame:
    if drift_df.empty:
        return drift_df

    median_mean_diff = drift_df["mean_diff"].replace(0, np.nan).median()
    mean_diff_thresh = 0.5 * median_mean_diff if pd.notna(median_mean_diff) else np.inf

    alerts = drift_df[
        (drift_df["psi"] > 0.2)
        | (drift_df["ks"] > 0.2)
        | (drift_df["mean_diff"] > mean_diff_thresh)
        | (drift_df["missing_diff"] > 0.05)
    ]
    return alerts.sort_values(["psi", "ks"], ascending=False)

def run_feature_drift_monitor(
    train_feature_sample: pd.DataFrame,
    daily_features: pd.DataFrame,
    out_dir: str | Path,
    date_str: str,
) -> dict:
    """
    train_feature_sample: sample of training features (rows = samples, cols = features)
    daily_features: features for current trading day
    out_dir: base directory for stability outputs
    date_str: e.g. '2025-01-03'
    """
    prints(f"[DRIFT] Running drift monitor for {date_str}", level="info")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    drift_df = feature_drift_report(train_feature_sample, daily_features)
    alerts_df = drift_alerts(drift_df)

    drift_path = out_dir / f"feature_drift_{date_str}.csv"
    alerts_path = out_dir / f"feature_drift_alerts_{date_str}.csv"

    drift_df.to_csv(drift_path)
    prints(f"[DRIFT] Drift CSV saved to: {drift_path}", level="info")
    alerts_df.to_csv(alerts_path)
    prints(f"[DRIFT] Alerts CSV saved to: {alerts_path}", level="info")

    summary = {
        "date": date_str,
        "n_features": int(len(drift_df)),
        "n_alerts": int(len(alerts_df)),
        "max_psi": float(drift_df["psi"].max(skipna=True)),
        "max_ks": float(drift_df["ks"].max(skipna=True)),
    }
    prints(f"[DRIFT] Summary: {summary}", level="info")

    with open(out_dir / f"feature_drift_summary_{date_str}.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary