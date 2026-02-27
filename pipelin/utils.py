from qlib.data import D
import os
import pandas as pd
import numpy as np
from datetime import datetime
from colorama import Fore, Style, init
import re
from typing import Optional
from pathlib import Path



SECTOR_MAP = {"AAPL": "Technology", "MSFT": "Technology", "GOOG": "Technology", "NVDA": "Technology", "AMD": "Technology", "INTC": "Technology", "QCOM": "Technology", "AVGO": "Technology", "ADBE": "Technology", "CRM": "Technology", "ORCL": "Technology", "CSCO": "Technology", "DDOG": "Technology", "PANW": "Technology", "SHOP": "Technology", "META": "Communication Services", "NFLX": "Communication Services", "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD":   "Consumer Discretionary", "LOW":  "Consumer Discretionary", "MCD":  "Consumer Discretionary", "SBUX": "Consumer Discretionary", "NKE":  "Consumer Discretionary", "WMT": "Consumer Staples", "COST": "Consumer Staples", "PG":   "Consumer Staples", "KO":   "Consumer Staples", "PEP":  "Consumer Staples", "JPM": "Financials", "BAC": "Financials", "WFC": "Financials", "GS":  "Financials", "MS":  "Financials", "UNH": "Healthcare", "JNJ": "Healthcare", "PFE": "Healthcare", "ABBV": "Healthcare", "LLY":  "Healthcare", "XOM": "Energy", "CVX": "Energy", "SPY":  "ETF", "SPUU": "ETF", "SSO":  "ETF", "QQQ":  "ETF", "VWO":  "ETF", "IBIT": "ETF",}
EPS = 1e-8
SOURCE_DIR = Path(r"C:\Users\harve\.qlib\stock_data\source\us_data")
ENRICH_DIR = r"C:\Users\harve\.qlib\stock_data\normalize\us_data_enriched"

# Initialize colorama for Windows terminals
init(autoreset=True)

# Default log file (can be overridden via init_log_file())
_log_file = "anything_else_log.txt"

# Regex to strip ANSI escape sequences (colors) for clean log files
_ANSI_ESCAPE = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')


# ============================================================
#   Qlib Utilities
# ============================================================
def g_safe_features():
    return [
        "$open", "$high", "$low", "$close",
        "$vol_5d", "$vol_10d", "$vol_20d",
        "$vol_10_20",   # "$vol_5_20",
        "$vol_20_60", "$vol_5_60",
        "$ret_5d", "$ret_10d", "$ret_20d",
        "$mom_60d", # "$mom_20d",
        "$mom_5d_z", "$mom_20d_z",
        "$price_above_ma20", "$price_above_ma60",
        "$trend_5_20",
        # "$intraday_range", # "$intraday_body",
        "$range_ma5", "$range_ma20",
        "$volume_log", "$volume_shock", "$volume_z", "$volume_vol",
        "$rank_vol_5d", "$rank_vol_10d", "$rank_vol_20d",
        "$rank_mom_20d", "$rank_mom_60d",
        "$rank_intraday_range", "$rank_volume_log", # "$rank_volume_shock",
        "$days_since_ipo_cont", "$ipo_bucket", "$ret_5d_vol_scaled", 
        "$trend_persist_ma20", "$trend_persist_ma60", # "$ret_10d_vol_scaled"
        "$vol_20d_resid_liq", # "$intraday_range_vol_norm", "$intraday_vol_ratio",
        "$ret_20d_vol_scaled", "$micro_imbalance_z_20", "$overnight_ret", # "$micro_imbalance",
        "$eps_actual_lag3", "$eps_est_lag3", "$eps_surprise_lag3", "$eps_ttm", "$eps_growth_yoy",
        "$surprise_std", "$surprise_pct", "$beat_streak", "$revision_trend", "$eps_momentum", "$earnings_yield",
    ]
# ============================================================
# Enrich data utilities
# ============================================================
def load_enriched_data():
    dfs = []
    for fname in os.listdir(ENRICH_DIR):
        if fname.endswith(".csv"):
            df = pd.read_csv(os.path.join(ENRICH_DIR, fname))
            dfs.append(df)
    full = pd.concat(dfs, ignore_index=True)
    return full

def attach_scores_to_enriched(full_df, X_train, model):
    idx = X_train.index

    score_df = pd.DataFrame({
        "symbol": idx.get_level_values(0),
        "date": idx.get_level_values(1).astype(str),
        "score": model.predict(X_train)
    })

    merged = full_df.merge(score_df, on=["date", "symbol"], how="left")
    return merged
def build_calibration_table(df, horizon=5, threshold=0.008, n_bins=20):
    col = f"fwd_ret_{horizon}d"

    d = df[['score', col]].dropna()

    # Create quantile bins
    d['score_bin'] = pd.qcut(d['score'], n_bins, duplicates='drop')

    # Group by the categorical bin directly
    calib = d.groupby('score_bin').agg(
        prob_up   = (col, lambda x: (x > threshold).mean()),
        avg_ret   = (col, 'mean'),
        med_ret   = (col, 'median'),
        count     = (col, 'count'),
        score_min = ('score', 'min'),
        score_max = ('score', 'max')
    ).reset_index()

    # Extract bin edges AFTER grouping
    calib['bin_low'] = calib['score_bin'].apply(lambda x: float(x.left))
    calib['bin_high'] = calib['score_bin'].apply(lambda x: float(x.right))

    calib = calib.drop(columns=['score_bin'])

    return calib

def save_calibration(calib, horizon):
    out_path = f"artifacts/calibration_{horizon}d.parquet"
    os.makedirs("artifacts", exist_ok=True)
    calib.to_parquet(out_path)
    print(f"[CALIB] Saved calibration table: {out_path}")

# called by training code
def load_merge_and_save_calibration(model, X):
    prints("[CALIB] Loading enriched data...")
    df_enriched = load_enriched_data()

    prints("[CALIB] Attaching model scores...")
    merged_df_enriched_with_score = attach_scores_to_enriched(df_enriched, X, model)
    thresholds = {5: 0.008, 10: 0.012, 20: 0.015}
    for horizon in [5, 10, 20]:
        prints(f"[CALIB] Building calibration for {horizon}d...")
        calib = build_calibration_table(
            merged_df_enriched_with_score,
            horizon=horizon,
            threshold=thresholds[horizon],
            n_bins=20
        )
        save_calibration(calib, horizon)

# used by daily inference code:
def load_calibrations():
    calib = {}
    for horizon in [5, 10, 20]:
        path = f"artifacts/calibration_{horizon}d.parquet"
        calib[horizon] = pd.read_parquet(path)
    return calib

def lookup_calibration(calib_df, score):
    row = calib_df[(score >= calib_df["bin_low"]) & (score <= calib_df["bin_high"])]
    if len(row) == 0:
        return None

    r = row.iloc[0]
    return {
        "prob_up": float(r["prob_up"]),
        "avg_ret": float(r["avg_ret"]),
        "med_ret": float(r["med_ret"]),
        "count": int(r["count"])
    }
# ============================================================

def get_last_trading_day() -> str:
    """
    Return the most recent trading day according to Qlib's calendar.
    """
    cal = D.calendar(start_time="2020-01-01", end_time=pd.Timestamp.today())
    return cal[-1].strftime("%Y-%m-%d")


# ============================================================
#   Logging Utilities
# ============================================================

def strip_colors(text: str) -> str:
    """Remove ANSI color codes from a string."""
    return _ANSI_ESCAPE.sub('', text)


def init_log_file(filename: str) -> None:
    """
    Set the log file path for all subsequent prints().
    """
    global _log_file
    _log_file = filename


def prints(message: str, level: Optional[str] = None) -> None:
    """
    Print a message to console (with optional color) and append it to the log file.

    Parameters
    ----------
    message : str
        The message to print.
    level : None | 'info' | 'warning' | 'error'
        Optional log level that controls console color.
    """

    # Color mapping for console output
    level_colors = {
        "info": Fore.GREEN,
        "warning": Fore.YELLOW,
        "error": Fore.RED + Style.BRIGHT,
    }

    # Apply color if level is valid
    if level in level_colors:
        console_message = f"{level_colors[level]}{message}"
    else:
        console_message = message

    # Print to console
    print(console_message)

    # Strip colors for log file
    clean_message = strip_colors(message)

    # Timestamp for log entry
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {clean_message}"

    # Append to log file
    with open(_log_file, "a", encoding="utf-8") as f:
        f.write(full_message + "\n")


def validate_prices(df):
    # 1. Check for non-positive prices
    for col in ["open", "high", "low", "close", "prev_close"]:
        if (df[col] <= 0).any():
            raise ValueError(f"Non-positive values found in {col}")

    # 2. Only enforce high >= low
    if not (df["high"] >= df["low"]).all():
        bad = df[df["high"] < df["low"]]
        raise ValueError(f"High < low for some rows:\n{bad}")

    # 3. Warn (not error) if low > open/close
    tol = 1e-6
    bad_low = df[df["low"] > df[["open", "close"]].min(axis=1) + tol]
    if len(bad_low) > 0:
        print("⚠️ Warning: low > open/close for some rows (likely adjustment noise)")
        print(bad_low[["instrument", "open", "high", "low", "close"]])

    return True


def load_real_close(symbol: str) -> float:
    """
    Loads the most recent REAL (unadjusted) close price from your local Yahoo CSV.
    """
    path = SOURCE_DIR / f"{symbol}.csv"
    df = pd.read_csv(path)

    last_row = df.iloc[-1]  # most recent trading day
    return float(last_row["close"])  # REAL Yahoo close


def calculate_vwap(df_today: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
        - vwap_yesterday (adjusted VWAP approximation)
        - split_factor (real_close / qlib_adjusted_close)
        - vwap (human-readable VWAP)
    """

    # 1. Improved VWAP approximation using weighted OHLC
    df_today["vwap_yesterday"] = (
        0.30 * df_today["$open"] +
        0.35 * df_today["$high"] +
        0.20 * df_today["$low"] +
        0.15 * df_today["$close"]
    )

    # 2. Compute split factors using REAL close / Qlib adjusted close
    split_factors = {}
    for sym in df_today["instrument"]:
        real_close = load_real_close(sym)
        qlib_adj_close = df_today.loc[df_today["instrument"] == sym, "$close"].iloc[0]
        split_factors[sym] = real_close / qlib_adj_close

    df_today["split_factor"] = df_today["instrument"].map(split_factors)

    # 3. Convert adjusted VWAP → real VWAP
    df_today["vwap"] = df_today["vwap_yesterday"] * df_today["split_factor"]

    return df_today

def cs_rank(s):
    return s.rank(pct=True)

def residualize(y, X):
    X_ = X.copy()
    X_['const'] = 1.0
    beta = np.linalg.lstsq(X_.values, y.values, rcond=None)[0]
    y_hat = X_.values @ beta
    return y - y_hat

def add_xs_ranks_1(X):
    # ============================================================
    # Cross-sectional ranks (must be computed at training time)
    # ============================================================
    # These features MUST be computed at training time because they
    # require cross-sectional context across all symbols for each date.

    xs_cols = [
        "$ret_5d", "$ret_10d", "$ret_20d",
        # "$vol_5_20",
        "$vol_10_20"
    ]

    for col in xs_cols:
        X[f"{col}_rank_xs"] = (
            X[col]
            .groupby(level=0)   # group by date
            .rank(pct=True)
        )

    # ============================================================
    # NaN-safe filling for new XS features ONLY
    # ============================================================
    new_cols = [
        "$ret_5d_rank_xs", "$ret_10d_rank_xs", "$ret_20d_rank_xs",
        # "$vol_5_20_rank_xs", 
        "$vol_10_20_rank_xs"
    ]

    for col in new_cols:
        X[col] = X[col].fillna(0)
    return X

def add_xs_ranks(df):
    xs_cols = [
        '$ret_20d_vol_scaled',
        # '$intraday_vol_ratio',
        '$trend_persist_ma20',
        '$trend_persist_ma60',
        '$vol_20d_resid_liq',
        #'$overnight_ret',
        #'$intraday_range_vol_norm', (the rank as feature importance of 10) may want to consider to put it back and delete the original column
    ]

    for col in xs_cols:
        rank_col = f"{col}_rank"
        df[rank_col] = df.groupby(level='datetime')[col].transform(cs_rank)

    return df

def add_sector_neutral_momentum(df):
    # 1. Inject sector
    symbols = df.index.get_level_values("instrument")
    df["sector"] = symbols.map(SECTOR_MAP).fillna("Other")

    def resid_mom(group):
        # Use ret_20d instead of removed mom_20d
        y = group['$ret_20d'].astype("float64")

        # Convert sector to category for stable encoding
        sectors = group['sector'].astype("category")

        # If only one sector → fallback
        if len(sectors.cat.categories) <= 1:
            return y - y.mean()

        # Build dummy matrix manually as float64
        X = pd.get_dummies(sectors, drop_first=True, dtype="float64")

        # Add constant
        X["const"] = 1.0

        # Solve OLS
        beta = np.linalg.lstsq(X.values, y.values, rcond=None)[0]
        y_hat = X.values @ beta
        return y - y_hat

    # Apply residualization
    df['$mom_20d_resid_sector'] = (
        df.groupby(level='datetime')
          .apply(lambda g: resid_mom(g))
          .reset_index(level=0, drop=True)
    )

    # Rank it cross-sectionally
    df['$mom_20d_resid_sector_rank'] = (
        df.groupby(level='datetime')['$mom_20d_resid_sector']
          .transform(cs_rank)
    )

    # Cleanup
    df = df.drop(columns=["sector"])
    return df

def add_beta_neutral_returns(df):
    if 'beta' not in df.columns or 'mkt_ret' not in df.columns:
        return df

    def resid_beta(group):
        y = group['ret_20d']
        X = pd.DataFrame({'beta_mkt': group['beta'] * group['mkt_ret']})
        return residualize(y, X)

    df['ret_20d_beta_neutral'] = (
        df.groupby(level='datetime')
          .apply(resid_beta)
          .reset_index(level=0, drop=True)
    )

    df['ret_20d_beta_neutral_rank'] = (
        df.groupby(level='datetime')['ret_20d_beta_neutral']
          .transform(cs_rank)
    )

    return df

def add_overnight_volatility(df):
    df['overnight_vol'] = (
        df.groupby(level='instrument')['$overnight_ret']
          .rolling(20)
          .std()
          .reset_index(level=0, drop=True)
    )
    # drop overnight_ret column after it has been used
    df = df.drop(columns=["$overnight_ret"])

    df['overnight_vol_z'] = (
        df.groupby(level='instrument')['overnight_vol']
          .transform(lambda s: (s - s.mean()) / (s.std() + EPS))
    )

    df['overnight_vol_rank'] = (
        df.groupby(level='datetime')['overnight_vol']
          .transform(cs_rank)
    )

    return df

def add_cross_sectional_features(df):
    df = add_xs_ranks_1(df)
    df = add_xs_ranks(df)
    df = add_sector_neutral_momentum(df)
    df = add_beta_neutral_returns(df)
    df = add_overnight_volatility(df)
    return df