# pipeline/forward_returns.py

import pandas as pd

from .config import ENTRY_LOG_PATH, FORWARD_HORIZONS


def update_forward_returns(price_df, today_date):
    """
    Update forward returns for all logged entries where enough days have passed.

    Parameters
    ----------
    price_df : DataFrame
        MultiIndex DataFrame indexed by (symbol, date) with 'close' column.
        Should cover the full date span needed for all horizons.
    today_date : str or datetime-like
        "Current" date at which you are updating (typically latest trading day).
    """
    today_date = pd.to_datetime(today_date)

    if not ENTRY_LOG_PATH.exists():
        print("No entry log found at", ENTRY_LOG_PATH)
        return None

    log = pd.read_parquet(ENTRY_LOG_PATH)
    if log.empty:
        return log

    log["date"] = pd.to_datetime(log["date"])

    # Iterate rows – you can vectorize later if needed
    for idx, row in log.iterrows():
        entry_date = row["date"]
        symbol = row["symbol"]
        entry_price = row["entry_price"]

        for h in FORWARD_HORIZONS:
            col_ret = f"ret_{h}d"
            col_flag = f"filled_{h}d"

            # If already filled or column missing, skip
            if col_flag not in log.columns or bool(row.get(col_flag, False)):
                continue

            target_date = entry_date + pd.Timedelta(days=h)

            # Not enough days have passed yet
            if target_date > today_date:
                continue

            key = (symbol, target_date)
            try:
                future_price = price_df.loc[key, "close"]
            except KeyError:
                # No price available for that symbol/date
                continue

            if row["direction"] == "LONG":
                ret = (future_price - entry_price) / entry_price
            else:  # SHORT
                ret = (entry_price - future_price) / entry_price

            log.at[idx, col_ret] = float(ret)
            log.at[idx, col_flag] = True

    log.to_parquet(ENTRY_LOG_PATH, index=False)
    return log