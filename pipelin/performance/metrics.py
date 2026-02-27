# pipeline/performance/metrics.py

import pandas as pd
import numpy as np

FORWARD_COLS = ["ret_5d", "ret_10d", "ret_20d", "ret_60d"]

def hit_rate(df, horizon):
    col = f"ret_{horizon}d"
    valid = df[col].dropna()
    if len(valid) == 0:
        return float("nan")
    return (valid > 0).mean()


def avg_return(df, horizon):
    col = f"ret_{horizon}d"
    return df[col].mean()


def long_short_split(df):
    return df[df["direction"] == "LONG"], df[df["direction"] == "SHORT"]


def score_bucket(df, n=5):
    df = df.copy()
    df["score_bucket"] = pd.qcut(df["score"], n, labels=False, duplicates="drop")
    return df


def bucket_monotonicity(df, horizon):
    df = score_bucket(df)
    col = f"ret_{horizon}d"
    return df.groupby("score_bucket")[col].mean()