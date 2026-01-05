import qlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from qlib.data import D
from utils import prints, initialize
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

# Use 5d forward return as PnL driver
FORWARD_RETURN_FIELD = "$ret_5d"

# Transaction costs: bps per side applied on turnover
TC_BPS_PER_SIDE = 10  # bump a bit from 5 to be less cartoonish

RESULTS_CSV_PATH = "backtest_long_short_results.csv"

initialize("logs/backtest_long_short.log")
# ============================================================
# PORTFOLIO CONSTRUCTION (keep your duplicate-safe version)
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

    # Remove duplicates: anything in longs cannot be in shorts
    long_names = set(longs["instrument"])
    shorts = shorts[~shorts["instrument"].isin(long_names)]

    # Refill shorts if needed
    if len(shorts) < top_k_short:
        needed = top_k_short - len(shorts)
        remaining = df_sorted[
            ~df_sorted["instrument"].isin(long_names | set(shorts["instrument"]))
        ]
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
    # Load features
    # -----------------------------
    features = D.features(
        instruments=instruments,
        fields=SAFE_FEATURES,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    X_all = features.copy()
    X_all["$volume_log"] = np.log1p(X_all["$volume"])
    X_all.drop(columns=["$volume"], inplace=True)

    X_all = X_all.replace([np.inf, -np.inf], np.nan)
    X_all = X_all.fillna(0)
    X_all = X_all.reindex(columns=model_cols)

    # -----------------------------
    # Load forward returns
    # -----------------------------
    fwd = D.features(
        instruments=instruments,
        fields=[FORWARD_RETURN_FIELD],
        start_time=START_DATE,
        end_time=END_DATE,
    )

    # Align indices
    common_index = X_all.index.intersection(fwd.index)
    X_all = X_all.loc[common_index]
    fwd = fwd.loc[common_index]

    dt_idx = X_all.index.get_level_values("datetime")
    unique_dates = np.sort(dt_idx.unique())

    if len(unique_dates) < 10:
        prints("Not enough dates after alignment; check data setup.")
        return

    prints(f"Backtest from {unique_dates[0]} to {unique_dates[-1]}")

    daily_pnl = []
    daily_gross_exposure = []
    daily_turnover = []
    daily_dates = []

    prev_weights = None  # for turnover calc

    # IMPORTANT: since FORWARD_RETURN_FIELD is 5d, skip last 5 days
    horizon = 5
    for i in range(len(unique_dates) - horizon):
        current_date = unique_dates[i]
        next_date = unique_dates[i + 1]  # we still use the label at current_date

        # Features at current_date
        mask_today = dt_idx == current_date
        X_today = X_all.loc[mask_today].copy()
        if X_today.empty:
            continue

        df_today = X_today.reset_index()

        # Predict scores
        scores = model.predict(X_today)
        df_today["score"] = scores

        # Build portfolio at t
        portfolio = build_long_short_portfolio(
            df_today,
            top_k_long=TOP_K_LONG,
            top_k_short=TOP_K_SHORT,
        )

        # Forward return at t for those instruments (already 5d ahead in Qlib label)
        fwd_t = fwd.loc[mask_today].reset_index()
        fwd_t = fwd_t[["instrument", FORWARD_RETURN_FIELD]]

        port = portfolio.merge(fwd_t, on="instrument", how="left")

        valid = port[FORWARD_RETURN_FIELD].notna()
        port_valid = port[valid]
        if port_valid.empty:
            continue

        # Gross PnL: sum(weight * forward_return)
        pnl_gross = (port_valid["weight"] * port_valid[FORWARD_RETURN_FIELD]).sum()

        # Turnover: compare to previous day's weights
        if prev_weights is None:
            turnover = port_valid["weight"].abs().sum() / 2.0
        else:
            prev = prev_weights.set_index("instrument")["weight"]
            curr = port_valid.set_index("instrument")["weight"]
            aligned = pd.concat([prev, curr], axis=1, keys=["prev", "curr"]).fillna(0.0)
            turnover = (aligned["curr"] - aligned["prev"]).abs().sum() / 2.0

        # Transaction costs (both sides)
        tc = turnover * (TC_BPS_PER_SIDE / 10000.0) * 2.0

        pnl_net = pnl_gross - tc

        daily_pnl.append(pnl_net)
        daily_gross_exposure.append(port_valid["weight"].abs().sum())
        daily_turnover.append(turnover)
        daily_dates.append(next_date)  # PnL realized between t and t+1 on 5d label

        prev_weights = port_valid[["instrument", "weight"]].copy()

    if not daily_pnl:
        prints("No PnL computed; check loop logic / data.")
        return

    results = pd.DataFrame(
        {
            "date": pd.to_datetime(daily_dates),
            "pnl": daily_pnl,
            "gross_exposure": daily_gross_exposure,
            "turnover": daily_turnover,
        }
    ).set_index("date")

    results["cum_pnl"] = results["pnl"].cumsum()
    # Assume capital base of 1.0
    results["return"] = results["pnl"]

    mu = results["return"].mean()
    sigma = results["return"].std()
    sharpe = mu / sigma * np.sqrt(252) if sigma > 0 else np.nan

    cum = results["cum_pnl"]
    peak = cum.cummax()
    drawdown = cum - peak
    max_dd = drawdown.min()

    prints("\n===== BACKTEST SUMMARY (Option B v2) =====")
    prints(f"Total days: {len(results)}")
    prints(f"Total PnL: {results['cum_pnl'].iloc[-1]:.4f}")
    prints(f"Annualized Sharpe: {sharpe:.2f}")
    prints(f"Max drawdown: {max_dd:.4f}")
    prints(f"Average daily turnover: {results['turnover'].mean():.4f}")
    prints(f"Average daily gross exposure: {results['gross_exposure'].mean():.4f}")

    prints("\nLast 5 days of PnL:")
    prints(results.tail(5))

    # Save for plotting / further analysis
    results.to_csv(RESULTS_CSV_PATH)
    prints(f"\nSaved backtest results to {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()