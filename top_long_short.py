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

TOP_K_LONG = 20
TOP_K_SHORT = 20
IC_WINDOW_DAYS = 60


# ============================================================
# PORTFOLIO CONSTRUCTION (Option A)
# ============================================================
def build_long_short_portfolio(df_today, top_k_long=20, top_k_short=20):
    """
    df_today must contain:
    - instrument
    - score
    """

    df_sorted = df_today.sort_values("score", ascending=False)

    # Initial picks
    longs = df_sorted.head(top_k_long).copy()
    shorts = df_sorted.tail(top_k_short).copy()

    # --- FIX DUPLICATES ---
    long_names = set(longs["instrument"])
    shorts = shorts[~shorts["instrument"].isin(long_names)]

    # If we removed duplicates, refill shorts from remaining names
    if len(shorts) < top_k_short:
        needed = top_k_short - len(shorts)
        remaining = df_sorted[~df_sorted["instrument"].isin(long_names | set(shorts["instrument"]))]
        refill = remaining.tail(needed)
        shorts = pd.concat([shorts, refill], axis=0)

    # Equal weights
    long_weight = 1.0 / len(longs)
    short_weight = -1.0 / len(shorts)

    longs["weight"] = long_weight
    shorts["weight"] = short_weight

    portfolio = pd.concat([longs, shorts], axis=0)

    # Enforce dollar neutrality
    total_weight = portfolio["weight"].sum()
    if abs(total_weight) > 1e-6:
        portfolio["weight"] -= total_weight / len(portfolio)

    return portfolio


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

    prints(f"Loaded model from {MODEL_PATH}")
    prints(f"Model expects {len(model_cols)} features")

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
    # Predict scores
    # -----------------------------
    scores = model.predict(X)
    df = X.copy()
    df["score"] = scores

    # -----------------------------
    # Determine latest date
    # -----------------------------
    dt_idx = df.index.get_level_values("datetime")
    latest_date = dt_idx.max()
    prints(f"Latest available date: {latest_date}")

    df_today = df.loc[dt_idx == latest_date].copy()
    df_today = df_today.reset_index()  # bring instrument + datetime into columns

    # -----------------------------
    # Build long/short portfolio
    # -----------------------------
    portfolio = build_long_short_portfolio(
        df_today,
        top_k_long=TOP_K_LONG,
        top_k_short=TOP_K_SHORT
    )

    prints("\n===== LONG/SHORT PORTFOLIO =====")
    for _, row in portfolio.iterrows():
        side = "LONG " if row["weight"] > 0 else "SHORT"
        prints(f"{side:<6} {row['instrument']:<10}  score={row['score']:.4f}  weight={row['weight']:.4f}")

    # -----------------------------
    # IC evaluation over recent window
    # -----------------------------
    unique_dates = np.sort(dt_idx.unique())
    cutoff_idx = max(0, len(unique_dates) - IC_WINDOW_DAYS)
    eval_dates = unique_dates[cutoff_idx:]

    mask_eval = dt_idx.isin(eval_dates)
    preds_eval = df.loc[mask_eval, "score"]

    # Load forward returns for IC
    labels = D.features(
        instruments=instruments,
        fields=["$ret_5d"],
        start_time=START_DATE,
        end_time=END_DATE,
    )

    labels = labels.loc[preds_eval.index]
    valid_mask = labels["$ret_5d"].notna()

    ic = spearmanr(preds_eval[valid_mask], labels["$ret_5d"][valid_mask]).correlation
    prints(f"\nIC over last {IC_WINDOW_DAYS} days: {ic:.4f}")


if __name__ == "__main__":
    main()