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
START_DATE = "2019-01-01"
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

# 1d forward return for now; we can switch to 5d later
FORWARD_RETURN_FIELD = "$ret_5d"

TC_BPS_PER_SIDE = 5  # 5 bps each way as a placeholder


# ============================================================
# PORTFOLIO CONSTRUCTION (same logic as top_long_short)
# ============================================================
def build_long_short_portfolio(df_today, top_k_long=20, top_k_short=20):
    """
    df_today must contain:
    - instrument
    - score
    """

    df_sorted = df_today.sort_values("score", ascending=False)

    longs = df_sorted.head(top_k_long).copy()
    shorts = df_sorted.tail(top_k_short).copy()

    # Remove duplicates: anything in longs cannot be in shorts
    long_names = set(longs["instrument"])
    shorts = shorts[~shorts["instrument"].isin(long_names)]

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
# BACKTEST LOOP
# ============================================================
def main():

    # -----------------------------
    # Init Qlib
    # -----------------------------
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    instrument_path = r"C:/Users/harve/.qlib/qlib_data/us_data/instruments/all.txt"
    with open(instrument_path, "r") as f:
        instruments = [line.strip().split("\t")[0] for line in f if line.strip()]

    # -----------------------------
    # Load model
    # -----------------------------
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)

    model = saved["model"]
    model_cols = saved["columns"]

    prints(f"Loaded model from {MODEL_PATH}")
    prints(f"Model expects {len(model_cols)} features")

    # -----------------------------
    # Load features & forward returns
    # -----------------------------
    features = D.features(
        instruments=instruments,
        fields=SAFE_FEATURES,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # feature engineering
    X_all = features.copy()
    X_all["$volume_log"] = np.log1p(X_all["$volume"])
    X_all.drop(columns=["$volume"], inplace=True)

    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(0)
    X_all = X_all.reindex(columns=model_cols)

    # forward returns for PnL
    fwd = D.features(
        instruments=instruments,
        fields=[FORWARD_RETURN_FIELD],
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # align indices
    common_index = X_all.index.intersection(fwd.index)
    X_all = X_all.loc[common_index]
    fwd = fwd.loc[common_index]

    # -----------------------------
    # Prepare date loop
    # -----------------------------
    dt_idx = X_all.index.get_level_values("datetime")
    instruments_idx = X_all.index.get_level_values("instrument")
    unique_dates = np.sort(dt_idx.unique())

    # series to accumulate results
    daily_pnl = []
    daily_gross_exposure = []
    daily_turnover = []
    daily_dates = []

    prev_weights = None  # for turnover

    prints(f"Backtest from {unique_dates[0]} to {unique_dates[-1]}")

    for current_date in unique_dates[:-1]:  # last date can't have 1d forward return
        # slice today's features
        mask_today = dt_idx == current_date
        X_today = X_all.loc[mask_today].copy()

        if X_today.empty:
            continue

        df_today = X_today.reset_index()  # instrument, datetime as columns

        # predict scores
        scores = model.predict(X_today)
        df_today["score"] = scores

        # build portfolio
        portfolio = build_long_short_portfolio(
            df_today,
            top_k_long=TOP_K_LONG,
            top_k_short=TOP_K_SHORT,
        )

        # next day's returns for instruments in portfolio
        next_date = unique_dates[np.searchsorted(unique_dates, current_date) + 1]
        mask_next = (dt_idx == next_date)
        fwd_next = fwd.loc[mask_next]

        # align by instrument
        fwd_next = fwd_next.reset_index()
        fwd_next = fwd_next[["instrument", FORWARD_RETURN_FIELD]]

        port = portfolio.merge(fwd_next, on="instrument", how="left")

        # compute PnL: sum(weight * forward_return)
        valid = port[FORWARD_RETURN_FIELD].notna()
        port_valid = port[valid]

        if port_valid.empty:
            continue

        pnl_gross = (port_valid["weight"] * port_valid[FORWARD_RETURN_FIELD]).sum()

        # transaction costs (very simple approximation)
        # turnover = sum |w_t - w_{t-1}| / 2
        if prev_weights is None:
            turnover = port_valid["weight"].abs().sum() / 2.0
        else:
            prev = prev_weights.set_index("instrument")["weight"]
            curr = port_valid.set_index("instrument")["weight"]
            aligned = pd.concat([prev, curr], axis=1, keys=["prev", "curr"]).fillna(0.0)
            turnover = (aligned["curr"] - aligned["prev"]).abs().sum() / 2.0

        tc = turnover * (TC_BPS_PER_SIDE / 10000.0) * 2.0  # both sides

        pnl_net = pnl_gross - tc

        daily_pnl.append(pnl_net)
        daily_gross_exposure.append(port_valid["weight"].abs().sum())
        daily_turnover.append(turnover)
        daily_dates.append(next_date)

        prev_weights = port_valid[["instrument", "weight"]].copy()

    # -----------------------------
    # Aggregate results
    # -----------------------------
    if not daily_pnl:
        prints("No PnL computed; check data coverage.")
        return

    results = pd.DataFrame({
        "date": pd.to_datetime(daily_dates),
        "pnl": daily_pnl,
        "gross_exposure": daily_gross_exposure,
        "turnover": daily_turnover,
    }).set_index("date")

    results["cum_pnl"] = results["pnl"].cumsum()
    results["return"] = results["pnl"]  # assuming 1.0 capital base for now

    # Sharpe (daily -> annualized ~ sqrt(252))
    mu = results["return"].mean()
    sigma = results["return"].std()
    sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else np.nan

    # max drawdown
    cum = results["cum_pnl"]
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = drawdown.min()

    prints("\n===== BACKTEST SUMMARY =====")
    prints(f"Total days: {len(results)}")
    prints(f"Total PnL: {results['cum_pnl'].iloc[-1]:.4f}")
    prints(f"Annualized Sharpe: {sharpe:.2f}")
    prints(f"Max drawdown: {max_dd:.4f}")
    prints(f"Average daily turnover: {results['turnover'].mean():.4f}")
    prints(f"Average daily gross exposure: {results['gross_exposure'].mean():.4f}")

    # optional: print last few rows for sanity
    prints("\nLast 5 days of PnL:")
    prints(results.tail(5))


if __name__ == "__main__":
    main()