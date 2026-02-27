import pandas as pd
import os
import numpy as np

source_dir = r"C:\Users\harve\.qlib\stock_data\normalize\us_data"
target_dir = r"C:\Users\harve\.qlib\stock_data\normalize\us_data_enriched"
os.makedirs(target_dir, exist_ok=True)

enriched_dfs = []
EPS = 1e-8

# ============================================================
#  ADVANCED FEATURE SUITE (Institutional Upgrade v2)
#  All features here are strictly per-symbol, time-series only.
#  Cross-sectional ranks & sector-neutral residuals are handled
#  in training/inference to avoid leakage.
# ============================================================

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------

def rolling_zscore(s, window):
    mean = s.rolling(window).mean()
    std = s.rolling(window).std()
    return (s - mean) / (std + EPS)


# ------------------------------------------------------------
# 1. Volatility-scaled returns
# ------------------------------------------------------------

def compute_vol_scaled_returns(df):
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)

    df['ret_5d_vol_scaled'] = df['ret_5d'] / (df['vol_5d'] + EPS)
    df['ret_10d_vol_scaled'] = df['ret_10d'] / (df['vol_10d'] + EPS)
    df['ret_20d_vol_scaled'] = df['ret_20d'] / (df['vol_20d'] + EPS)
    return df


# ------------------------------------------------------------
# 2. Intraday volatility ratio
# ------------------------------------------------------------

def compute_intraday_vol_ratio(df):
    df['intraday_range'] = (df['high'] - df['low']) / (df['close'] + EPS)
    # df['intraday_vol_ratio'] = df['intraday_range'] / (df['close'] + EPS)
    return df


# ------------------------------------------------------------
# 3. Overnight returns
# ------------------------------------------------------------

def compute_overnight_returns(df):
    df['overnight_ret'] = df['open'] / df['close'].shift(1) - 1
    df['overnight_ret_z_20'] = rolling_zscore(df['overnight_ret'], 20)
    return df


# ------------------------------------------------------------
# 4. Volume-normalized intraday range
# ------------------------------------------------------------

def compute_volume_normalized_range(df):
    df['intraday_range_vol_norm'] = df['intraday_range'] / (df['volume_log'] + EPS)
    return df


# ------------------------------------------------------------
# 5. Trend persistence (MA20 and MA60)
# ------------------------------------------------------------

def compute_trend_persistence(df):
    df['trend_persist_ma20'] = df['price_above_ma20'].rolling(60).mean()
    df['trend_persist_ma60'] = df['price_above_ma60'].rolling(60).mean()
    return df


# ------------------------------------------------------------
# 6. Microstructure imbalance proxy
# ------------------------------------------------------------

def compute_microstructure_imbalance(df):
    df['micro_imbalance'] = (df['close'] - df['open']) / (df['high'] - df['low'] + EPS)
    df['micro_imbalance'] = df['micro_imbalance'].clip(-5, 5)
    df['micro_imbalance_z_20'] = rolling_zscore(df['micro_imbalance'], 20)
    return df


# ------------------------------------------------------------
# 7. Residualized volatility (vs. liquidity)
# ------------------------------------------------------------

def compute_residual_volatility(df):
    if 'vol_20d' not in df.columns or 'volume_log' not in df.columns:
        df['vol_20d_resid_liq'] = np.nan
        return df

    valid = df[['vol_20d', 'volume_log']].dropna()
    if len(valid) < 30:
        df['vol_20d_resid_liq'] = np.nan
        return df

    y = valid['vol_20d'].astype(float)
    X = valid[['volume_log']].astype(float)
    X['const'] = 1.0

    if X['volume_log'].std() < 1e-8:
        resid = y - y.mean()
        df['vol_20d_resid_liq'] = resid.reindex(df.index)
        return df

    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    y_hat = X.values @ beta
    resid = y - y_hat

    df['vol_20d_resid_liq'] = resid.reindex(df.index)
    return df


# ------------------------------------------------------------
# 8. Beta-neutral returns (symbol-level beta)
# ------------------------------------------------------------

def compute_beta_neutral_returns(df):
    if 'beta' not in df.columns or 'mkt_ret' not in df.columns:
        return df

    y = df['ret_20d']
    X = pd.DataFrame({'beta_mkt': df['beta'] * df['mkt_ret'], 'const': 1.0})

    beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
    y_hat = X.values @ beta

    df['ret_20d_beta_neutral'] = y - y_hat
    return df


# ------------------------------------------------------------
# MASTER FUNCTION
# ------------------------------------------------------------

def add_institutional_features(df):
    df = compute_vol_scaled_returns(df)
    df = compute_intraday_vol_ratio(df)
    df = compute_overnight_returns(df)
    df = compute_volume_normalized_range(df)
    df = compute_trend_persistence(df)
    df = compute_microstructure_imbalance(df)
    df = compute_residual_volatility(df)
    df = compute_beta_neutral_returns(df)
    return df


# ============================================================
# MAIN LOOP
# ============================================================

for fname in os.listdir(source_dir):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(source_dir, fname))
    df["symbol"] = fname.replace(".csv", "")
    # ============================================================
    # Ensure EPS columns exist for ALL symbols (ETFs, etc.)
    # ============================================================
    EPS_COLS = [
        "eps_actual_lag3", "eps_est_lag3", "eps_surprise_lag3",
        "eps_ttm", "eps_growth_yoy", "surprise_std", "surprise_pct",
        "beat_streak", "revision_trend", "eps_momentum", "earnings_yield"
    ]

    for col in EPS_COLS:
        if col not in df.columns:
            df[col] = np.nan    

    eps = 1e-6

    # ============================
    # Backward returns
    # ============================
    df["ret_5d"]  = df["close"] / df["close"].shift(5)  - 1
    df["ret_10d"] = df["close"] / df["close"].shift(10) - 1
    df["ret_20d"] = df["close"] / df["close"].shift(20) - 1

    df["mom_20d"] = df["ret_20d"] # this is not included into dump to bin.
    df["mom_60d"] = df["close"] / df["close"].shift(60) - 1

    df["mom_5d_z"]  = rolling_zscore(df["ret_5d"], 60)
    df["mom_20d_z"] = rolling_zscore(df["ret_20d"], 60)

    df["price_above_ma20"] = df["close"] / df["close"].rolling(20).mean()
    df["price_above_ma60"] = df["close"] / df["close"].rolling(60).mean()
    df["trend_5_20"] = df["close"].rolling(5).mean() / df["close"].rolling(20).mean()

    pct = df["close"].pct_change()
    df["vol_5d"]  = pct.rolling(5).std()
    df["vol_10d"] = pct.rolling(10).std()
    df["vol_20d"] = pct.rolling(20).std()

    df["vol_5_20"]  = df["vol_5d"]  / (df["vol_20d"] + eps)
    df["vol_10_20"] = df["vol_10d"] / (df["vol_20d"] + eps)

    df["vol_20_60"] = df["vol_20d"] / (pct.rolling(60).std() + eps)
    df["vol_5_60"]  = df["vol_5d"]  / (pct.rolling(60).std() + eps)

    df["intraday_range"] = (df["high"] - df["low"]) / (df["close"] + eps)
    df["intraday_body"]  = (df["close"] - df["open"]) / ((df["high"] - df["low"]) + eps)

    df["range_ma5"]  = df["intraday_range"].rolling(5).mean()
    df["range_ma20"] = df["intraday_range"].rolling(20).mean()

    df["volume_log"] = np.log1p(df["volume"])
    df["volume_shock"] = df["volume_log"] / (df["volume_log"].rolling(20).mean() + eps)
    df["volume_z"] = rolling_zscore(df["volume_log"], 60)
    df["volume_vol"] = df["volume_log"].rolling(20).std()

    df["fwd_ret_5d"]  = df["close"].shift(-5)  / df["close"] - 1
    df["fwd_ret_10d"] = df["close"].shift(-10) / df["close"] - 1
    df["fwd_ret_20d"] = df["close"].shift(-20) / df["close"] - 1

    df["ensemble_label"] = (
        0.5 * df["fwd_ret_5d"] +
        0.3 * df["fwd_ret_10d"] +
        0.2 * df["fwd_ret_20d"]
    )

    df["days_since_ipo"] = (
        pd.to_datetime(df["date"]) -
        pd.to_datetime(df["date"].min())
    ).dt.days

    df["days_since_ipo_cont"] = df["days_since_ipo"].clip(0, 500) / 500.0

    bins = [0, 30, 70, 130, 250, 600, np.inf]
    labels = list(range(len(bins) - 1))
    df["ipo_bucket"] = pd.cut(df["days_since_ipo"], bins=bins, labels=labels, include_lowest=True)
    df["ipo_bucket"] = df["ipo_bucket"].astype(float).fillna(0).astype(int)

    df = add_institutional_features(df)

    enriched_dfs.append(df)

full_df = pd.concat(enriched_dfs)

xs_cols = [
    "vol_5d", "vol_10d", "vol_20d",
    "mom_20d", "mom_60d",
    "intraday_range", "volume_log", "volume_shock"
]

for col in xs_cols:
    full_df[f"rank_{col}"] = full_df.groupby("date")[col].rank()

for symbol, df_symbol in full_df.groupby("symbol"):
    df_symbol.to_csv(os.path.join(target_dir, f"{symbol}.csv"), index=False)