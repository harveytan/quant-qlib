# pipeline/entry_logger.py

import os
import pandas as pd

from .config import ENTRY_LOG_PATH, FORWARD_HORIZONS


def log_entries(today_date, today_entries, price_df):
    """
    Log today's SAFE entries in a duplicate-safe way.

    Parameters
    ----------
    today_date : str or datetime-like
        Date of the entries (trading date).
    today_entries : list[dict]
        Output from your SAFE classifier. Each dict should include:
        - symbol
        - direction ("LONG" or "SHORT")
        - score
        - momentum
        - crash
        - reason
    price_df : DataFrame
        MultiIndex DataFrame indexed by (symbol, date) with a 'close' column.
        Must include (symbol, today_date) for all symbols in today_entries.
    """
    today_date = pd.to_datetime(today_date)

    if ENTRY_LOG_PATH.exists():
        log = pd.read_parquet(ENTRY_LOG_PATH)
        log["date"] = pd.to_datetime(log["date"])
    else:
        log = pd.DataFrame()

    # Build today's entries
    df = pd.DataFrame(today_entries)
    if df.empty:
        return log  # nothing to log

    df["date"] = today_date

    # Ensure required columns exist
    required_cols = ["symbol", "direction", "score", "momentum", "crash", "reason"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in today_entries: {missing}")

    # Attach entry_price
    entry_prices = []
    for row in df.itertuples():
        key = (row.symbol, today_date)
        try:
            entry_prices.append(price_df.loc[key, "close"])
        except KeyError:
            raise KeyError(f"Missing price for (symbol={row.symbol}, date={today_date}) in price_df")

    df["entry_price"] = entry_prices

    # Initialize forward return columns
    for h in FORWARD_HORIZONS:
        df[f"ret_{h}d"] = pd.NA
        df[f"filled_{h}d"] = False

    # If log empty, just save df
    if log.empty:
        df.to_parquet(ENTRY_LOG_PATH, index=False)
        return df

    # Remove any existing entries for the same (date, symbol)
    log = log[~log.set_index(["date", "symbol"]).index.isin(
        df.set_index(["date", "symbol"]).index
    )].reset_index(drop=True)

    # Append refreshed entries
    updated = pd.concat([log, df], ignore_index=True)

    # Enforce uniqueness (keep newest)
    updated = updated.drop_duplicates(subset=["date", "symbol"], keep="last")

    updated.to_parquet(ENTRY_LOG_PATH, index=False)
    return updated