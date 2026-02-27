# stability/recent_ic.py

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from qlib.data import D
from pipeline.utils import prints


def run_recent_ic_monitor(
    df,
    dt_idx,
    instruments,
    start_date,
    end_date,
    window=60,
):
    """
    Computes IC over the last N days of today's predictions.
    """

    unique_dates = np.sort(dt_idx.unique())
    cutoff_idx = max(0, len(unique_dates) - window)
    eval_dates = unique_dates[cutoff_idx:]

    mask_eval = dt_idx.isin(eval_dates)
    preds_eval = df.loc[mask_eval, "score"]

    # Load forward returns
    labels = D.features(
        instruments=instruments,
        fields=["$ret_5d"],
        start_time=start_date,
        end_time=end_date,
    )

    labels = labels.loc[preds_eval.index]
    valid_mask = labels["$ret_5d"].notna()

    ic = spearmanr(preds_eval[valid_mask], labels["$ret_5d"][valid_mask]).correlation

    prints(f"[IC] Performance IC over last {window} days: {ic:.4f}")

    return ic