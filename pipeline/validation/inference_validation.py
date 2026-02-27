import numpy as np
import pandas as pd

def validate_inference_pipeline(
    X,
    df_today_raw,
    df_full,
    model_cols,
    train_sample,
    latest_date,
    verbose=True,
):
    report = {}

    # ============================================================
    # 1. Column presence + order
    # ============================================================
    missing = set(model_cols) - set(X.columns)
    extra = set(X.columns) - set(model_cols)

    report["missing_columns"] = list(missing)
    report["extra_columns"] = list(extra)
    report["column_order_correct"] = (list(X.columns) == list(model_cols))

    # ============================================================
    # 2. Dtype validation
    # ============================================================
    bad_dtypes = X.dtypes[X.dtypes == "object"].index.tolist()
    report["object_dtypes"] = bad_dtypes

    # ============================================================
    # 3. NaN pollution
    # ============================================================
    nan_cols = X.columns[X.isna().any()].tolist()
    report["nan_columns"] = nan_cols

    # ============================================================
    # 4. Feature engineering version check
    # ============================================================
    import pipeline.features as feat
    report["compute_all_features_version"] = feat.compute_all_features.__code__.co_consts

    # ============================================================
    # 5. Score distribution sanity check
    # ============================================================
    if "score" in df_full.columns:
        scores = df_full["score"]
        report["score_distribution"] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
            "min": float(scores.min()),
            "max": float(scores.max()),
            "skew": float(scores.skew()),
        }

    # ============================================================
    # 6. Compare inference vs training sample distribution
    # ============================================================
    common_cols = [c for c in model_cols if c in train_sample.columns]
    diffs = {}

    for col in common_cols:
        train_mean = train_sample[col].mean()
        infer_mean = df_today_raw[col].mean()
        diffs[col] = float(infer_mean - train_mean)

    report["train_vs_infer_mean_diff"] = diffs

    # ============================================================
    # 7. Universe alignment
    # ============================================================
    infer_universe = set(df_today_raw.index)
    full_universe = set(df_full.xs(latest_date, level="datetime").index)

    report["universe_mismatch"] = list(infer_universe.symmetric_difference(full_universe))

    # ============================================================
    # 8. SAFE feature isolation
    # ============================================================
    SAFE_FEATURES = [
        "mom_3d", "trend_20d", "trend_60d",
        "gap_down_pct",
        "close_1d_change", "close_2d_change", "close_3d_change",
    ]

    leaked = [c for c in SAFE_FEATURES if c in model_cols]
    report["safe_feature_leakage"] = leaked

    # ============================================================
    # 9. Drift vs training sample (simple z-score)
    # ============================================================
    drift = {}
    for col in common_cols:
        train_std = train_sample[col].std()
        if train_std > 0:
            z = (df_today_raw[col].mean() - train_sample[col].mean()) / train_std
            drift[col] = float(z)

    report["feature_drift_zscores"] = drift

    # ============================================================
    # 10. Final summary
    # ============================================================
    if verbose:
        print("\n=== INFERENCE VALIDATION REPORT ===")
        for k, v in report.items():
            print(f"{k}: {v}")

    return report