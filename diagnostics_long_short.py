import qlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from qlib.data import D
from utils import prints
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================
START_DATE = "2018-01-01"
END_DATE = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

MODEL_PATH = "trained_model_2.pkl"

SAFE_FEATURES = [
    "$open", "$high", "$low", "$close",
    "$volume",
    "$vol_5d", "$vol_10d", "$vol_20d",
    "$rank_vol_5d", "$rank_vol_10d", "$rank_vol_20d",
    "$days_since_ipo",
]

FORWARD_RETURNS = ["$ret_5d", "$ret_10d", "$ret_20d"]

IC_WINDOW_DAYS = 60
BUCKETS = 5
IPO_CUTOFF = 600  # ~1 year of trading days


# ============================================================
# MAIN
# ============================================================
def main():

    # -----------------------------
    # Init Qlib
    # -----------------------------
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    # Load instruments
    instrument_path = r"C:/Users/harve/.qlib/qlib_data/us_data/instruments/all.txt"
    with open(instrument_path, "r") as f:
        instruments = [line.strip().split("\t")[0] for line in f if line.strip()]

    # -----------------------------
    # Load model + training columns
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    model_cols = saved["columns"]

    prints(f"Loaded model with {len(model_cols)} features")

    # -----------------------------
    # Load features
    # -----------------------------
    features = D.features(
        instruments=instruments,
        fields=SAFE_FEATURES,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # Feature engineering
    X = features.copy()
    X["$volume_log"] = np.log1p(X["$volume"])
    X.drop(columns=["$volume"], inplace=True)

    # Clean
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # Align columns
    X = X.reindex(columns=model_cols)

    # -----------------------------
    # Load forward returns for attribution
    # -----------------------------
    labels = D.features(
        instruments=instruments,
        fields=FORWARD_RETURNS,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # Align indices
    common_index = X.index.intersection(labels.index)
    X = X.loc[common_index]
    labels = labels.loc[common_index]

    # -----------------------------
    # Predict scores
    # -----------------------------
    scores = model.predict(X)
    df = X.copy()
    df["score"] = scores

    # Attach forward returns
    df = pd.concat([df, labels], axis=1)

    # Drop rows without forward returns
    df = df.dropna(subset=["$ret_5d"])

    # Reset index for easier slicing
    df = df.reset_index()

    # -----------------------------
    # Compute IC over recent window
    # -----------------------------
    unique_dates = np.sort(df["datetime"].unique())
    cutoff_idx = max(0, len(unique_dates) - IC_WINDOW_DAYS)
    eval_dates = unique_dates[cutoff_idx:]

    df_eval = df[df["datetime"].isin(eval_dates)]

    ic = spearmanr(df_eval["score"], df_eval["$ret_5d"]).correlation
    prints(f"\n📈 IC over last {IC_WINDOW_DAYS} days: {ic:.4f}")

    # -----------------------------
    # Bucket attribution
    # -----------------------------
    df["bucket"] = pd.qcut(df["score"], q=BUCKETS, labels=False)

    prints("\n📊 Average forward returns per score bucket:")
    for horizon in ["5", "10", "20"]:
        col = f"$ret_{horizon}d"
        bucket_ret = df.groupby("bucket")[col].mean()
        prints(f"\n⏱ {horizon}d returns:")
        prints(bucket_ret)

    prints("\n✅ Hit rate per bucket (5d):")
    hit_rate = df.groupby("bucket")["$ret_5d"].apply(lambda x: (x > 0).mean())
    prints(hit_rate)

    # -----------------------------
    # IPO cohort attribution
    # -----------------------------
    df["ipo_cohort"] = (df["$days_since_ipo"] < IPO_CUTOFF).astype(int)

    for cohort, name in [(0, "Core (≥600 days)"), (1, "IPO (<600 days)")]:
        df_c = df[df["ipo_cohort"] == cohort]

        prints(f"\n📊 {name} — 5d Return by Bucket:")
        prints(df_c.groupby("bucket")["$ret_5d"].mean())

        prints(f"Hit Rate:")
        prints(df_c.groupby("bucket")["$ret_5d"].apply(lambda x: (x > 0).mean()))


if __name__ == "__main__":
    main()