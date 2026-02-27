# pipeline/eps.py
import time
import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# CONFIG
# ============================================================

CACHE_DIR = os.path.join("data", "eps_cache")
MAX_AGE_HOURS = 39
MAX_CALLS_PER_DAY = 25


# ============================================================
# 1. Cache utilities
# ============================================================

def cache_path(symbol):
    return os.path.join(CACHE_DIR, f"{symbol}_eps.json")


def is_cache_fresh(path):
    if not os.path.exists(path):
        return False
    age_hours = (time.time() - os.path.getmtime(path)) / 3600
    return age_hours < MAX_AGE_HOURS


def load_cached_eps(symbol):
    path = cache_path(symbol)
    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        data = json.load(f)

    # AlphaVantage format
    if isinstance(data, dict) and "quarterlyEarnings" in data:
        q = data["quarterlyEarnings"]
        df = pd.DataFrame(q)
        df = coerce_eps_types(df)
        return df

    print(f"[EPS] Unknown cache format for {symbol}")
    return None

def coerce_eps_types(df):
    numeric_cols = ["reportedEPS", "estimatedEPS", "surprise"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def save_cache(symbol, full_json):
    """
    Save the entire JSON response exactly as returned by AlphaVantage.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(symbol)
    with open(path, "w") as f:
        json.dump(full_json, f, indent=2)

# ============================================================
# 2. API fetch
# ============================================================
def fetch_eps_from_api(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={symbol}&apikey={api_key}"
    r = requests.get(url)
    data = r.json()
    time.sleep(1)  # AlphaVantage rate limit pacing

    # Quota reached
    if "Information" in data and data["Information"].startswith("We have detected your API key"):
        return None

    if "Note" in data and data["Note"].startswith("We have detected your API key as "):
        return None

    print(f"[EPS] Fetched from API for {symbol}")
    save_cache(symbol, data)

    # Extract quarterly earnings for DataFrame use
    if isinstance(data, dict) and "quarterlyEarnings" in data:
        q = data["quarterlyEarnings"]
    else:
        return None

    df = pd.DataFrame(q)
    df = coerce_eps_types(df)
    return df


# ============================================================
# 3. Public loader (quota-aware)
# ============================================================

def load_eps(symbol, api_key, api_counter):
    """
    Returns (eps_df, api_counter)
    - Uses cache if fresh
    - Refreshes if stale and quota allows
    - Always returns a DataFrame or None
    """

    path = cache_path(symbol)

    # Fresh cache → use it
    if is_cache_fresh(path):
        return load_cached_eps(symbol), api_counter

    # Quota reached → fallback to stale cache
    if api_counter >= MAX_CALLS_PER_DAY:
        print(f"[EPS] Quota reached — using stale cache for {symbol}")
        return load_cached_eps(symbol), api_counter

    # Fetch from API
    eps_df = fetch_eps_from_api(symbol, api_key)
    if eps_df is not None and not eps_df.empty:
        api_counter += 1
        return eps_df, api_counter

    # API returned nothing → fallback to stale cache
    return load_cached_eps(symbol), api_counter


# ============================================================
# 4. Align EPS to next trading day (PIT-correct)
# ============================================================

def align_eps(daily_df, eps_df):
    """
    Shifts EPS to the correct trading day:
    - post-market → next trading day
    - pre-market / unknown → same day
    """

    eps_df = eps_df.copy()
    eps_df["reportedDate"] = pd.to_datetime(eps_df["reportedDate"]).dt.normalize()

    trading_days = daily_df["date"].sort_values().unique()

    aligned = []
    for rd, rt in zip(eps_df["reportedDate"], eps_df["reportTime"]):
        if isinstance(rt, str) and "post" in rt.lower():
            td = trading_days[trading_days > rd]
        else:
            td = trading_days[trading_days >= rd]

        aligned.append(td[0] if len(td) else None)

    eps_df["aligned_date"] = aligned
    eps_df = eps_df.dropna(subset=["aligned_date"])
    eps_df["aligned_date"] = pd.to_datetime(eps_df["aligned_date"]).dt.normalize()

    return eps_df


# ============================================================
# 5. Merge EPS into daily data (lag3, ffill, PIT-safe)
# ============================================================
def merge_eps(daily_df, eps_df):
    # 0) Clean daily data
    daily_df.columns = daily_df.columns.str.strip()
    daily_df["date"] = pd.to_datetime(daily_df["date"]).dt.normalize()
    daily_df = daily_df.drop_duplicates(subset=["date"]).sort_values("date")

    # 1) Remove ALL EPS columns (robust)
    eps_cols = [c for c in daily_df.columns if c.startswith("eps_")]
    daily_df = daily_df.drop(columns=eps_cols, errors="ignore")

    # 2) Prepare EPS data
    eps_df = eps_df.sort_values("aligned_date").reset_index(drop=True)
    eps_df["eps_actual_lag3"] = eps_df["reportedEPS"].shift(3)
    eps_df["eps_est_lag3"] = eps_df["estimatedEPS"].shift(3)
    eps_df["eps_surprise_lag3"] = eps_df["surprise"].shift(3)

    # 3) Safe merge (never deletes date)
    merged = daily_df.merge(
        eps_df[[
            "aligned_date",
            "eps_actual_lag3",
            "eps_est_lag3",
            "eps_surprise_lag3"
        ]],
        left_on="date",
        right_on="aligned_date",
        how="left"
    )

    merged = merged.drop(columns=["aligned_date"], errors="ignore")

    # 4) Forward-fill
    for col in ["eps_actual_lag3", "eps_est_lag3", "eps_surprise_lag3"]:
        merged[col] = merged[col].ffill()

    return merged
# ============================================================
# 6. EPS-derived features
# ============================================================
def add_eps_features(df):
    # Ensure date is a column
    if "date" not in df.columns:
        df = df.reset_index()

    # Ensure date exists now
    if "date" not in df.columns:
        raise ValueError("ERROR: 'date' column missing before EPS feature generation")

    # Clean daily data
    df = df.drop_duplicates(subset=["date"]).sort_values("date")

    # Remove old EPS feature columns to avoid suffix conflicts
    EPS_FEATURE_COLS = [
        "eps_ttm", "eps_growth_yoy", "surprise_std", "surprise_pct",
        "beat_streak", "revision_trend", "eps_momentum", "earnings_yield"
    ]
    df = df.drop(columns=EPS_FEATURE_COLS, errors="ignore")

    # ---------------------------------------------------------
    # 1) Extract true EPS event days
    # ---------------------------------------------------------
    events = df.loc[
        df["eps_actual_lag3"].notna() &
        (df["eps_actual_lag3"] != df["eps_actual_lag3"].shift())
    ][[
        "date", "eps_actual_lag3", "eps_est_lag3", "eps_surprise_lag3"
    ]].copy()

    events = (
        events.drop_duplicates(subset=["date"])
              .sort_values("date")
              .reset_index(drop=True)
    )

    # ---------------------------------------------------------
    # 2) Compute event-level features (PIT-safe)
    # ---------------------------------------------------------

    # TTM EPS
    events["eps_ttm"] = (
        events["eps_actual_lag3"]
        + events["eps_actual_lag3"].shift(1)
        + events["eps_actual_lag3"].shift(2)
        + events["eps_actual_lag3"].shift(3)
    )

    # YoY EPS growth
    events["eps_growth_yoy"] = events["eps_actual_lag3"].pct_change(4)

    # Surprise volatility
    events["surprise_std"] = (
        events["eps_surprise_lag3"]
        .rolling(4, min_periods=2)
        .std()
    )

    # Surprise magnitude
    events["surprise_pct"] = (
        events["eps_surprise_lag3"] /
        events["eps_est_lag3"].abs()
    )

    # Beat streak (institutional)
    events["beat"] = (events["eps_surprise_lag3"] > 0).astype(int)
    events["beat_streak"] = (
        events["beat"]
        * (
            events["beat"]
            .groupby((events["beat"] == 0).cumsum())
            .cumcount() + 1
        )
    )

    # Revision trend
    events["revision_trend"] = (
        events["eps_est_lag3"]
        .diff()
        .rolling(4, min_periods=1)
        .mean()
    )

    # EPS momentum
    events["eps_momentum"] = (
        events["eps_growth_yoy"]
        .rolling(4, min_periods=1)
        .mean()
    )

    # ---------------------------------------------------------
    # 3) Safe merge using index join (no explosion)
    # ---------------------------------------------------------
    # Safe merge — never deletes date, never creates suffixes
    df = df.merge(
        events[[
            "date",
            "eps_ttm",
            "eps_growth_yoy",
            "surprise_std",
            "surprise_pct",
            "beat_streak",
            "revision_trend",
            "eps_momentum"
        ]],
        on="date",
        how="left"
    )

    # ---------------------------------------------------------
    # 4) Forward-fill event features
    # ---------------------------------------------------------
    EVENT_FEATURE_COLS = [
        "eps_ttm", "eps_growth_yoy", "surprise_std", "surprise_pct",
        "beat_streak", "revision_trend", "eps_momentum"
    ]

    for col in EVENT_FEATURE_COLS:
        df[col] = df[col].ffill()

    # ---------------------------------------------------------
    # 5) Earnings yield (TTM / price) with outlier control
    # ---------------------------------------------------------
    df["earnings_yield"] = df["eps_ttm"] / df["close"].replace(0, np.nan)
    df["earnings_yield"] = df["earnings_yield"].clip(-5, 5)

    return df