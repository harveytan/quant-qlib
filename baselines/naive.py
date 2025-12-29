import numpy as np
import pandas as pd

def naive_predict(df, label_col="ensemble_label"):
    """
    Naive baseline: predict tomorrow's label = today's label,
    but do it PER TICKER to avoid leakage across symbols.
    """
    df = df.copy()
    df["naive_pred"] = df.groupby("symbol")[label_col].shift(1)
    return df["naive_pred"].values