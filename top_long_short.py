from pathlib import Path
import qlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from qlib.data import D
from pipeline.utils import prints, init_log_file, get_last_trading_day, calculate_vwap, add_cross_sectional_features, g_safe_features, load_calibrations, lookup_calibration
from pipeline.portfolio_builder import build_long_short_portfolio
from pipeline.display_utils import print_safe_trades
from pipeline.features import compute_all_features, momentum_label
from pipeline.daily_logger import run_daily_logging
from pipeline.performance.daily_summary import run_daily_summary
from pipeline.performance.reason_attribution import run_reason_attribution
from pipeline.execution.intraday import simulate_execution_intraday

# Stability modules
from stability import run_feature_drift_monitor, run_rolling_ic_monitor, run_recent_ic_monitor

# ============================================================
# CONFIG
# ============================================================
START_DATE = "2018-01-01"
END_DATE = (datetime.today() - timedelta(days=0)).strftime("%Y-%m-%d")
MODEL_PATH = "trained_model_2.pkl"
SAFE_FEATURES = g_safe_features()
SAFE_DF_PATH = Path("artifacts/safe_entries.parquet")
TOP_K_LONG = 20
TOP_K_SHORT = 20
IC_WINDOW_DAYS = 60

init_log_file("logs/top_long_short.log")

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
    prints(f"Model expects {len(model_cols)} features with columns: {model_cols}")

    # -----------------------------
    # Load features
    # -----------------------------
    features = D.features(
        instruments=instruments,
        fields=SAFE_FEATURES,
        start_time=START_DATE,
        end_time=END_DATE,
    )

    X = features.copy()

    X = add_cross_sectional_features(X)
    prints(f"Features loaded with shape: {X.shape} and columns: {X.columns.tolist()}")


    # Fill remaining NaNs with 0 only if necessary
    X = X.fillna(0)
    # ============================================================
    # Clean NaN/Inf
    # ============================================================
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    # ============================================================
    # Align columns
    # ============================================================
    missing = set(model_cols) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required model columns: {missing}")

    X = X.reindex(columns=model_cols)


    # -----------------------------
    # Predict scores
    # -----------------------------
    scores = model.predict(X)
    df = X.copy()
    df["score"] = scores

    df = df.sort_index()

    # === SIGNAL MOMENTUM CALCULATIONS & PRICE CRASH INDICATOR ===
    df = compute_all_features(df)
    # ---------------------
    # Determine latest date
    # ---------------------
    # how many instruments should exist on a "complete" day
    n_instruments = df.index.get_level_values('instrument').nunique()
    # count how many rows per date (i.e., how many instruments each date has)
    per_date_counts = df.index.get_level_values('datetime').value_counts()
    # keep only dates where ALL instruments are present
    complete_dates = per_date_counts[per_date_counts == n_instruments].index
    # latest common date across all instruments
    latest_date = complete_dates.max()

    # your df_today slice
    df_today = df.xs(latest_date, level='datetime').copy()
    prints(f"Latest available date: {latest_date}")
    df_today = df_today.reset_index()  # bring instrument + datetime into columns

    # ==============
    # calculate vwap
    # ==============
    df_today = calculate_vwap(df_today)

    # =======================================
    # SAVE TODAY'S PREDICTIONS FOR ROLLING IC
    # =======================================
    df_pred_today = pd.DataFrame({
        "date": pd.to_datetime(latest_date),
        "symbol": df_today["instrument"],
        "pred": df_today["score"],
    })

    # Save daily predictions
    pred_dir = Path("stability_outputs/daily_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    df_pred_today.to_csv(pred_dir / f"preds_{latest_date.date()}.csv", index=False)

    # --------------------------
    # Build long/short portfolio
    # --------------------------
    portfolio = build_long_short_portfolio(
        df_today=df_today,
        df_full=df,
        latest_date=latest_date,
        top_k_long=TOP_K_LONG,
        top_k_short=TOP_K_SHORT,
        momentum_label_fn=momentum_label,
    )

    prints("\n===== LONG/SHORT PORTFOLIO =====")
    calib = load_calibrations()
    for _, row in portfolio.iterrows():
        side = "LONG" if row["weight"] > 0 else "SHORT"
        level = "info"
        symbol = row['instrument']
        score = row['score']
        pad = ' '
        if row['score'] < 0:
            pad = ''
        if row["entry_human"] == "BLOCKED" or row['entry_raw'].startswith('RISKY'):
            level = "error"
        elif (
            (row["entry_human"].startswith("WATCH") and row["entry_raw"] == "SAFE")
            or row["entry_human"] == "SAFE_HI_SPREAD"
        ):
            level = "warning"
        prints(
            f"{side:<6} {symbol:5s} "
            f"score={score:.5f}{pad}  "
            f"mom={row['mom_label']:4s}  "
            f"crash={row['crash_label']:3s} "
            f"{row['entry_human']:14s} "  # human-safe label
            f"{row['entry_raw']:12s} "          # system view (optional)
            f"why={row['entry_reason']:20s} "
            f"weight={row['weight']:.4f} "
            f"vwap=${row['vwap']:.0f}", level
        )
        out = {}
        for horizon in [5, 10, 20]:
            c = lookup_calibration(calib[horizon], score)
            if c is None:
                out[f"prob_{horizon}d"] = None
                out[f"exp_{horizon}d"] = None
            else:
                out[f"prob_{horizon}d"] = c["prob_up"]
                out[f"exp_{horizon}d"] = c["avg_ret"]

        prints(
            f"           {pad}|"
            f"5d: {out['prob_5d']*100:5.1f}% ({out['exp_5d']*100:+.2f}%) | "
            f"10d: {out['prob_10d']*100:5.1f}% ({out['exp_10d']*100:+.2f}%) | "
            f"20d: {out['prob_20d']*100:5.1f}% ({out['exp_20d']*100:+.2f}%)"
        )


    safe_trades = print_safe_trades(portfolio, SAFE_DF_PATH)

    # =====================
    # FEATURE DRIFT MONITOR
    # =====================
    train_sample = pd.read_parquet("artifacts/train_features_sample.parquet")
    run_feature_drift_monitor(
        train_feature_sample=train_sample,
        daily_features=df_today.set_index("instrument")[model_cols],
        out_dir="stability_outputs/feature_drift",
        date_str=str(latest_date.date()),
    )
    # -----------------------------
    # IC evaluation over recent window
    # -----------------------------
    run_recent_ic_monitor(
        df=df,
        dt_idx=df.index.get_level_values("datetime"),
        instruments=instruments,
        start_date=START_DATE,
        end_date=END_DATE,
        window=IC_WINDOW_DAYS,
    )
    # ===============================================
    # ROLLING IC STABILITY (requires labels to exist)
    # ===============================================
    # Rolling IC monitor
    run_rolling_ic_monitor(
        instruments=instruments,
        start_date=START_DATE,
        end_date=END_DATE,
        pred_dir="stability_outputs/daily_predictions",
        window=20,
    )

    # today = get_last_trading_day()
    run_daily_logging(str(latest_date.date()))
    run_daily_summary()
    run_reason_attribution()
 
    # ====================
    # Execution simulation
    # ====================
    if safe_trades.empty:
        prints("No SAFE trades → skipping execution simulation")
        exit(0) # early exit for notebook

    # === ATTACH WEIGHTS FIRST (this fixes the KeyError) ===
    safe_trades = safe_trades.merge(
        portfolio[["instrument", "weight"]],
        left_on="symbol",
        right_on="instrument",
        how="left"
    ).drop(columns=["instrument"])
    safe_trades = safe_trades.rename(columns={"direction": "side"})
    safe_trades["side"] = safe_trades["side"].map({'LONG': 1, 'SHORT': -1})
    required_cols = ["score", "reason"]
    for col in required_cols:
        if col not in safe_trades.columns:
            safe_trades[col] = None

        # Simulate execution
    PORTFOLIO_NOTIONAL = 100000
    # === SYNTHETIC MICROSTRUCTURE FIELDS ===
    # price = today's close
    safe_trades["price"] = np.exp(
        df_today.set_index("instrument").loc[safe_trades["symbol"], "$close"].values
    )
    safe_trades["prev_price"] = np.exp(
        df_today.set_index("instrument").loc[safe_trades["symbol"], "prev_close"].values
    )
    # spread model: 2 bps + volatility adjustment
    safe_trades["spread_bps"] = 2 + 0.1 * safe_trades["crash"]  # or use vol_20d if you prefer

    # volatility proxy
    safe_trades["vol_daily"] = df_today.set_index("instrument").loc[safe_trades["symbol"], "$vol_20d"].values

    # ADV proxy: use today's volume_log exponentiated (approx)
    # since $volume was dropped, we approximate ADV from vol_20d
    safe_trades["adv_shares"] = 1e6 * (1 + safe_trades["vol_daily"])  # simple synthetic ADV

    safe_trades["portfolio_notional"] = PORTFOLIO_NOTIONAL

    safe_exec = simulate_execution_intraday(
        safe_trades,
        model="VWAP",      # or "TWAP", "POV"
        pov_rate=0.1,      # only used for POV
        adv_cap=0.1,       # max 10% of ADV per order
        n_buckets=10,
    )
    safe_exec["price_real"] = safe_exec["price"]
    safe_exec["effective_price_real"] = safe_exec["effective_price"]

    safe_exec["price_log"] = np.log(safe_exec["price"])
    safe_exec["effective_price_log"] = np.log(safe_exec["effective_price"])

    safe_exec.to_parquet(f"execution_outputs/safe_exec_{latest_date.date()}.parquet", index=False)
    # Attach score + reason
    if "score" in safe_trades.columns:
        safe_exec = safe_exec.merge(
            safe_trades[["symbol", "score", "reason"]],
            on="symbol",
            how="left"
        )
    else:
        safe_exec["score"] = None
        safe_exec["reason"] = None

    cols_to_keep = [
        "symbol", "side", "weight", "score", "reason",
        "price", "effective_price", "slippage_bps",
        "order_size_shares", "fill_fraction", "executed_notional"
    ]

    safe_exec_log = safe_exec[cols_to_keep].copy()
    safe_exec_log["date"] = latest_date

    prints("=== EXECUTION v2 SUMMARY ===")
    prints(f"Avg slippage (bps): {safe_exec['slippage_bps'].mean():.2f}")
    prints(f"Avg fill fraction: {safe_exec['fill_fraction'].mean():.2f}")
    prints(f"Total executed notional: {safe_exec['executed_notional'].sum():,.0f}")
    safe_exec_full = safe_exec.copy()
    safe_exec_full["date"] = latest_date
    safe_exec_full.to_parquet(f"execution_outputs/safe_exec_full_{latest_date.date()}.parquet", index=False)
    prints(f"Trades simulated: {len(safe_exec)}")


if __name__ == "__main__":
    main()