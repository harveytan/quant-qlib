from pathlib import Path
import qlib
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from qlib.data import D
from utils import prints, initialize
from scipy.stats import spearmanr
# Stability modules
from stability import run_feature_drift_monitor, run_rolling_ic_monitor
from stability.rolling_ic import compute_daily_ic

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

SAFE_DF_PATH = Path("artifacts/safe_entries.parquet")

TOP_K_LONG = 20
TOP_K_SHORT = 20
IC_WINDOW_DAYS = 60

initialize("logs/top_long_short.log")

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
def base_model_entry_logic(row):
    score = row["score"]
    mom = row["mom_label"]
    crash = row["crash_label"]
    direction = row["direction"]  # +1 long, -1 short

    # ---------- BASE (MODEL / SYSTEM VIEW) ----------
    # entry_raw = how the system views the setup

    # LONGS
    if direction == 1:
        # very weak signal -> blocked
        if score < 0.015:
            entry_raw = "BLOCKED"
            reason = "long_weak_score"
        # hard block on extreme crash
        elif crash in ["3x", "4x"]:
            entry_raw = "BLOCKED"
            reason = "long_extreme_crash"
        # momentum dead
        elif mom == "0x":
            entry_raw = "BLOCKED"
            reason = "long_zero_momentum"
        # moderate crash or soft momentum -> risky
        elif crash == "2x" or mom == "0.5x":
            entry_raw = "RISKY"
            reason = "long_crash_or_soft_momentum"
        else:
            entry_raw = "SAFE"
            reason = "long_clean_trend"

    # SHORTS
    else:
        # very weak short score -> blocked
        if score > -0.01:
            entry_raw = "BLOCKED"
            reason = "short_weak_score"
        # momentum dead
        elif mom == "0x":
            entry_raw = "BLOCKED"
            reason = "short_zero_momentum"
        # soft momentum -> risky
        elif mom == "0.5x":
            entry_raw = "RISKY"
            reason = "short_soft_momentum"
        else:
            entry_raw = "SAFE"
            reason = "short_clean_trend"

    # ---------- LATE-CRASH OVERRIDE FOR SHORTS ----------
    # If we are shorting into a 3x/4x crash, mark as late/dangerous
    if direction == -1 and crash in ["3x", "4x"] and entry_raw in ["SAFE", "RISKY"]:
        entry_raw = "RISKY_LATE_CRASH"
        reason = "short_late_crash"

    # return only the base model view
    return {
        "entry_raw": entry_raw,
        "entry_reason": reason,
    }


def classify_entry(row):
    """
    Human-safe entry classifier.

    - Uses base_model_entry_logic() as the system / model view (entry_raw, entry_reason)
    - Adds human overlays: value setups, trend filter, crash awareness, overextension, volatility guardrails
    - ALWAYS returns: entry_raw, entry_human, entry_reason
    """
    direction = row["direction"]          # 1 = long, -1 = short
    score = row["score"]
    mom_raw = row["mom_raw"]
    dist_ma20 = row["dist_ma20"]
    crash_score = row["crash_score"]      # numeric crash severity
    vol_ratio = row["vol_ratio"]          # from build_long_short_portfolio
    ret_1d = row["ret_1d"]
    close = row["close"]
    ma_20 = row["ma_20"]

    # ---------------------------------
    # 0. BASE MODEL VIEW (SYSTEM LOGIC)
    # ---------------------------------
    base = base_model_entry_logic(row)
    entry_raw = base["entry_raw"]         # "SAFE", "RISKY", "BLOCKED", "RISKY_LATE_CRASH"
    base_reason = base["entry_reason"]

    # If base logic already BLOCKED, respect it
    if entry_raw == "BLOCKED":
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": base_reason,
        }

    # ---------------------------------
    # 1. HARD LATE-CRASH OVERRIDE (NUMERIC)
    # ---------------------------------
    # If numeric crash score is very high, this is a dangerous late crash
    if crash_score >= 3:
        return {
            "entry_raw": entry_raw,
            "entry_human": "WATCH_LATECRASH",
            "entry_reason": "short_late_crash" if direction == -1 else "long_late_crash",
        }

    # ---------------------------------
    # 2. VALUE SETUP LOGIC
    # High score + rising score + price weak + NOT crashing
    # ---------------------------------
    is_high_score = abs(score) > 0.02
    is_rising_score = mom_raw > 0
    is_price_weak = dist_ma20 < 0
    is_not_crashing = crash_score <= 1

    if is_high_score and is_rising_score and is_price_weak and is_not_crashing:
        return {
            "entry_raw": entry_raw,
            "entry_human": "WATCH_VAL_SETUP",
            "entry_reason": "high_score_rising_price_weak_no_crash",
        }

    # ---------------------------------
    # 3. TREND FILTER (STRICT SAFE MODE)
    # ---------------------------------
    if direction == 1:
        trend_ok = close > ma_20
    else:
        trend_ok = close < ma_20

    if not trend_ok:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "trend_misaligned",
        }

    # ---------------------------------
    # 4. OVEREXTENSION GUARDRAIL
    # ---------------------------------
    if direction == 1 and ret_1d > 0.04:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "overextended_up",
        }

    if direction == -1 and ret_1d < -0.04:
        return {
            "entry_raw": entry_raw,
            "entry_human": "BLOCKED",
            "entry_reason": "overextended_down",
        }

    # ---------------------------------
    # 5. VOLATILITY GUARDRAIL
    # ---------------------------------
    if vol_ratio > 1.5:
        return {
            "entry_raw": entry_raw,
            "entry_human": "SAFE_HI_SPREAD" if entry_raw == "SAFE" else "WATCH_HI_SPREAD",
            "entry_reason": "high_volatility_use_spread_limit",
        }

    # ---------------------------------
    # 6. DEFAULT CLEAN ENTRY (BASE-CONSISTENT)
    # ---------------------------------
    # If we made it here:
    # - base did NOT block
    # - trend is aligned
    # - not overextended
    # - not high-vol crash
    # So we can trust the base view.
    if entry_raw == "SAFE":
        entry_human = "SAFE"
    elif entry_raw in ["RISKY", "RISKY_LATE_CRASH"]:
        # Up to you; for now be conservative and BLOCK
        entry_human = "BLOCKED"
    else:
        # Fallback (should not happen, but keeps things safe)
        entry_human = "BLOCKED"

    return {
        "entry_raw": entry_raw,
        "entry_human": entry_human,
        "entry_reason": base_reason,
    }

# ============================================================
# PRINT SAFE TRADES
# ============================================================
def print_safe_trades(portfolio):
    # Filter SAFE trades only
    safe = portfolio[portfolio["entry_human"].str.startswith("SAFE")].copy()

    if safe.empty:
        prints("\n=== SAFE TRADES ===")
        prints("No SAFE trades today.")
        return

    # Sort: LONGS first (direction=1), then by score desc, then momentum desc
    safe = safe.sort_values(
        by=["direction", "score", "mom_value"],
        ascending=[False, False, False]
    )

    prints("\n=== SAFE TRADES (sorted by LONG/SHORT → score → momentum) ===")

    for _, row in safe.iterrows():
        side = "LONG" if row["direction"] == 1 else "SHORT"
        prints(
            f"{side:5} {row['instrument']:6}  "
            f"score={row['score']:.4f}  "
            f"mom={row['mom_value']:.1f}x  "
            f"crash={row['crash_score']}x  "
            f"reason={row['entry_reason']}"
        )

    # Normalize columns for downstream pipeline
    safe_norm = pd.DataFrame({
        "symbol": safe["instrument"].astype(str),
        "direction": safe["direction"].apply(lambda x: "LONG" if x == 1 else "SHORT"),
        "score": safe["score"],
        "momentum": safe["mom_value"],        # or mom_raw if you prefer
        "crash": safe["crash_score"],         # or crash_label if you prefer
        "reason": safe["entry_reason"],
    })

    SAFE_DF_PATH.parent.mkdir(parents=True, exist_ok=True)
    safe_norm.to_parquet(SAFE_DF_PATH, index=False)


# ============================================================
# PORTFOLIO CONSTRUCTION (Option A)
# ============================================================
def build_long_short_portfolio(df_today, top_k_long=20, top_k_short=20):

    # prev_close already computed in df before slicing
    # so df_today["prev_close"] already exists

    # -----------------------------------------
    # 1. SORT BY SCORE
    # -----------------------------------------
    df_sorted = df_today.sort_values("score", ascending=False)

    longs = df_sorted.head(top_k_long).copy()
    shorts = df_sorted.tail(top_k_short).copy()

    # Remove duplicates
    long_names = set(longs["instrument"])
    shorts = shorts[~shorts["instrument"].isin(long_names)]

    if len(shorts) < top_k_short:
        needed = top_k_short - len(shorts)
        remaining = df_sorted[
            ~df_sorted["instrument"].isin(long_names | set(shorts["instrument"]))
        ]
        refill = remaining.tail(needed)
        shorts = pd.concat([shorts, refill], axis=0)

    # -----------------------------------------
    # 2. VOLATILITY-SCALED WEIGHTS
    # -----------------------------------------
    median_vol = df_today["vol_20d"].median()

    longs["vol_20d"] = longs["vol_20d"].fillna(median_vol)
    shorts["vol_20d"] = shorts["vol_20d"].fillna(median_vol)

    vol_floor = median_vol * 0.5
    vol_cap   = median_vol * 3.0

    longs["vol_20d"] = longs["vol_20d"].clip(lower=vol_floor, upper=vol_cap)
    shorts["vol_20d"] = shorts["vol_20d"].clip(lower=vol_floor, upper=vol_cap)

    longs["inv_vol"] = 1.0 / longs["vol_20d"]
    shorts["inv_vol"] = 1.0 / shorts["vol_20d"]

    longs["weight"] = longs["inv_vol"] / longs["inv_vol"].sum()
    shorts["weight"] = -shorts["inv_vol"] / shorts["inv_vol"].sum()

    portfolio = pd.concat([longs, shorts], axis=0)

    # Dollar neutrality correction
    total_weight = portfolio["weight"].sum()
    if abs(total_weight) > 1e-6:
        portfolio["weight"] -= total_weight / len(portfolio)

    # -----------------------------------------
    # 3. DIRECTION & MOMENTUM LABELS
    # -----------------------------------------
    portfolio["direction"] = portfolio["weight"].apply(lambda w: 1 if w > 0 else -1)

    # mom_raw already exists in df_today
    portfolio["mom_value"] = portfolio["mom_raw"] * portfolio["direction"]
    portfolio["mom_label"] = portfolio["mom_value"].apply(momentum_label)

    # -----------------------------------------
    # 4. DERIVED FIELDS FOR NEW RULES
    # -----------------------------------------

    # 1-day return
    portfolio["ret_1d"] = portfolio["close"] / portfolio["prev_close"] - 1

    # Trend alignment
    portfolio["trend_ok"] = (
        (portfolio["direction"] == 1) & (portfolio["close"] > portfolio["ma_20"])
    ) | (
        (portfolio["direction"] == -1) & (portfolio["close"] < portfolio["ma_20"])
    )

    # Volatility ratio
    median_vol = portfolio["vol_20d"].median()
    portfolio["vol_ratio"] = portfolio["vol_20d"] / median_vol

    # -----------------------------------------
    # 5. APPLY CLASSIFIER
    # -----------------------------------------
    entry_df = portfolio.apply(classify_entry, axis=1, result_type="expand")
    portfolio = pd.concat([portfolio, entry_df], axis=1)

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

    df = df.sort_index()

    # === SIGNAL MOMENTUM CALCULATIONS ===
    # 1-day lagged score
    df["score_prev1"] = (df.groupby(level="instrument")["score"].shift(1))

    # 3-day moving average of score
    df["score_ma_3d"] = (df.groupby(level="instrument")["score"].rolling(window=3, min_periods=2).mean().reset_index(level=0, drop=True))

    df["mom_raw"] = df["score"] - df["score_ma_3d"]
    df["mom_long"] = df["mom_raw"]          # long direction
    df["mom_short"] = -df["mom_raw"]        # short direction


    # === PRICE CRASH INDICATOR ===
    # 3-day and 5-day returns
    df["ret_3d"] = (df.groupby(level="instrument")["$close"].pct_change(3))
    df["ret_5d"] = (df.groupby(level="instrument")["$close"].pct_change(5))
    # Short-term moving averages
    df["ma_10"] = (df.groupby(level="instrument")["$close"].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True))
    df["ma_20"] = (df.groupby(level="instrument")["$close"].rolling(window=20, min_periods=10).mean().reset_index(level=0, drop=True))
    # Distance from moving averages
    df["dist_ma10"] = (df["$close"] - df["ma_10"]) / df["ma_10"]
    df["dist_ma20"] = (df["$close"] - df["ma_20"]) / df["ma_20"]
    # ATR (Average True Range)
    df["tr"] = (df["$high"] - df["$low"])
    df["atr_5"] = (df.groupby(level="instrument")["tr"].rolling(window=5, min_periods=3).mean().reset_index(level=0, drop=True))
    df["atr_20"] = (df.groupby(level="instrument")["tr"].rolling(window=20, min_periods=10).mean().reset_index(level=0, drop=True))
    df["atr_ratio"] = df["atr_5"] / df["atr_20"]

    # prev_close
    df["prev_close"] = (df.groupby(level="instrument")["$close"].shift(1))

    # Compute simple crash score
    df["crash_score"] = 0
    # 3-day crash
    df.loc[df["ret_3d"] < -0.05, "crash_score"] += 1
    # 5-day crash
    df.loc[df["ret_5d"] < -0.08, "crash_score"] += 1
    # Below MA10 by > 3%
    df.loc[df["dist_ma10"] < -0.03, "crash_score"] += 1
    # Below MA20 by > 5%
    df.loc[df["dist_ma20"] < -0.05, "crash_score"] += 1
    # Volatility spike
    df.loc[df["atr_ratio"] > 1.5, "crash_score"] += 1

    df["crash_label"] = df["crash_score"].apply(crash_label)
    # -----------------------------
    # Determine latest date
    # -----------------------------
    dt_idx = df.index.get_level_values("datetime")
    latest_date = dt_idx.max()
    prints(f"Latest available date: {latest_date}")

    df_today = df.loc[dt_idx == latest_date].copy()
    df_today = df_today.reset_index()  # bring instrument + datetime into columns

    df_today_raw = df_today.copy()
    # Rename df_today columns
    df_today = df_today.rename(columns={
        "$close": "close",
        "$open": "open",
        "$high": "high",
        "$low": "low",
        "$vol_20d": "vol_20d",
    })

    # ============================================================
    # FEATURE DRIFT MONITOR
    # ============================================================
    try:
        train_sample = pd.read_parquet("artifacts/train_features_sample.parquet")

        drift_summary = run_feature_drift_monitor(
            train_feature_sample=train_sample,
            daily_features=df_today_raw.set_index("instrument")[model_cols],  # same columns as training
            out_dir="stability_outputs/feature_drift",
            date_str=str(latest_date.date()),
        )

        prints(f"[DAILY] Feature drift summary: {drift_summary}")
        # what this means: ex. output: 
        # [DAILY] Feature drift summary: {'date': '2025-12-26', 'n_features': 12, 'n_alerts': 10, 'max_psi': 1.3676453113723086, 'max_ks': 0.7435897435897436}
        # n_feature: total number of features monitored
        # n_alerts: 10 out 12 features triggered drift alerts - huge number - not small drift - this is a regime shift or data distribution shift.
        # PSI (population stability index) : is the most important drift metric:
        #  - psi < 0.1 : no significant drift
        #  - 01. - 0.25 : moderate drift
        #  - >0.25 : significant drift
        #  - > 1: catastrophic drift
        # KS (Kolmogorov–Smirnov distance) measures distribution shape difference.
        # Interpretation:
        #  • 0.0–0.1 → similar distributions
        #  • 0.1–0.2 → mild drift
        #  • 0.2–0.3 → moderate drift
        #  • > 0.3 → strong drift
        #  • > 0.5 → severe drift
        #  • > 0.7 → massive drift
        # Your KS = 0.74 → this is extremely high.
        # This means the shape of at least one feature’s distribution is completely different from training.
    except Exception as e:
        prints(f"[WARNING] Feature drift monitor failed: {e}")

    # ============================================================
    # SAVE TODAY'S PREDICTIONS FOR ROLLING IC
    # ============================================================
    df_pred_today = pd.DataFrame({
        "date": pd.to_datetime(latest_date),
        "symbol": df_today["instrument"],
        "pred": df_today["score"],
    })

    # Save daily predictions
    pred_dir = Path("stability_outputs/daily_predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    df_pred_today.to_csv(pred_dir / f"preds_{latest_date.date()}.csv", index=False)

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
        side = "LONG" if row["weight"] > 0 else "SHORT"
        level = "info"
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
            f"{side:<6} {row['instrument']:5s} "
            f"score={row['score']:.5f}{pad}  "
            f"mom={row['mom_label']:4s}  "
            f"crash={row['crash_label']:3s} "
            f"{row['entry_human']:14s} "  # human-safe label
            f"{row['entry_raw']:12s} "          # system view (optional)
            f"why={row['entry_reason']:20s} "
            f"weight={row['weight']:.4f}", level
        )

    print_safe_trades(portfolio)
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

    # ============================================================
    # ROLLING IC STABILITY (requires labels to exist)
    # ============================================================
    try:
        pred_files = sorted(Path("stability_outputs/daily_predictions").glob("preds_*.csv"))
        if pred_files:
            df_all = pd.concat([pd.read_csv(f) for f in pred_files], ignore_index=True)

            # --- FIX: Force datetime conversion ---
            df_all["date"] = pd.to_datetime(df_all["date"], errors="coerce")

            # Load labels
            labels_all = D.features(
                instruments=instruments,
                fields=["$ret_5d"],
                start_time=START_DATE,
                end_time=END_DATE,
            ).reset_index()

            labels_all = labels_all.rename(columns={"instrument": "symbol", "$ret_5d": "label"})
            labels_all["datetime"] = pd.to_datetime(labels_all["datetime"], errors="coerce")

            # Merge on datetime
            df_all = df_all.merge(
                labels_all[["datetime", "symbol", "label"]],
                left_on=["date", "symbol"],
                right_on=["datetime", "symbol"],
                how="left"
            )

            df_all = df_all.dropna(subset=["label"])

            if not df_all.empty:
                ic_series = compute_daily_ic(df_all)
                if len(ic_series) == 0:
                    ic_last = None
                    ic_20_last = None
                    ic_vol_20_last = None
                else:
                    ic_last = float(ic_series.iloc[-1])

                    if len(ic_series) >= 20:
                        window = ic_series.iloc[-20:]
                        ic_20_last = float(np.nanmean(window))
                        ic_vol_20_last = float(np.nanstd(window))
                    else:
                        ic_20_last = None
                        ic_vol_20_last = None

                ic_summary = {
                    "last_date": str(df_all["date"].max()),
                    "IC_last": ic_last,
                    "IC_20_last": ic_20_last,
                    "IC_vol_20_last": ic_vol_20_last,
                    "n_alerts_total": 0,   # keep your existing logic
                }

                prints(f"[DAILY] Rolling IC summary: {ic_summary}")

    except Exception as e:
        prints(f"[WARNING] Rolling IC monitor failed: {e}")



if __name__ == "__main__":
    main()