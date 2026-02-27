# pipeline/features.py

import pandas as pd
import numpy as np

# pipeline/features.py

import pandas as pd
import numpy as np


# ============================================================
# LABEL HELPERS (exported)
# ============================================================

def momentum_label(v):
    if pd.isna(v):
        return "NA"
    if v < -0.01:
        return "0.5x"   # weakening
    if v < 0.01:
        return "1x"     # flat
    if v < 0.02:
        return "2x"     # mild strengthening
    if v < 0.04:
        return "3x"     # strong
    return "4x"          # very strong


def crash_label(v):
    if pd.isna(v):
        return "NA"
    if v == 0:
        return "0x"
    if v == 1:
        return "1x"
    if v == 2:
        return "2x"
    if v == 3:
        return "3x"
    return "4x"   # 4 or 5

def compute_momentum_features(df):
    """
    Adds:
    - score_prev1
    - score_ma_3d
    - mom_raw
    - mom_long
    - mom_short
    """
    df["score_prev1"] = (
        df.groupby(level="instrument")["score"].shift(1)
    )

    df["score_ma_3d"] = (
        df.groupby(level="instrument")["score"]
        .rolling(window=3, min_periods=2)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df["mom_raw"] = df["score"] - df["score_ma_3d"]
    df["mom_long"] = df["mom_raw"]
    df["mom_short"] = -df["mom_raw"]

    return df


def compute_crash_indicators(df, crash_label):
    """
    Adds:
    - ret_3d, ret_5d
    - ma_10, ma_20
    - dist_ma10, dist_ma20
    - tr, atr_5, atr_20, atr_ratio
    - prev_close
    - crash_score
    - crash_label
    """

    # Returns
    df["ret_3d"] = (
        df.groupby(level="instrument")["$close"].pct_change(3)
    )
    df["ret_5d"] = (
        df.groupby(level="instrument")["$close"].pct_change(5)
    )

    # Moving averages
    df["ma_10"] = (
        df.groupby(level="instrument")["$close"]
        .rolling(window=10, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["ma_20"] = (
        df.groupby(level="instrument")["$close"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )

    # Distances
    df["dist_ma10"] = (df["$close"] - df["ma_10"]) / df["ma_10"]
    df["dist_ma20"] = (df["$close"] - df["ma_20"]) / df["ma_20"]

    # ATR
    df["tr"] = df["$high"] - df["$low"]
    df["atr_5"] = (
        df.groupby(level="instrument")["tr"]
        .rolling(window=5, min_periods=3)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["atr_20"] = (
        df.groupby(level="instrument")["tr"]
        .rolling(window=20, min_periods=10)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["atr_ratio"] = df["atr_5"] / df["atr_20"]

    # Previous close
    df["prev_close"] = (
        df.groupby(level="instrument")["$close"].shift(1)
    )

    # Crash score
    df["crash_score"] = 0
    df.loc[df["ret_3d"] < -0.05, "crash_score"] += 1
    df.loc[df["ret_5d"] < -0.08, "crash_score"] += 1
    df.loc[df["dist_ma10"] < -0.03, "crash_score"] += 1
    df.loc[df["dist_ma20"] < -0.05, "crash_score"] += 1
    df.loc[df["atr_ratio"] > 1.5, "crash_score"] += 1

    df["crash_label"] = df["crash_score"].apply(crash_label)

    return df


def compute_all_features(df):
    """
    Full feature engineering pipeline.
    """
    df = compute_momentum_features(df)
    df = compute_crash_indicators(df, crash_label)
    return df