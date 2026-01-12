# daily_inference.py

import qlib
from qlib.config import REG_US  # or REG_CN depending on your setup

qlib.init(provider_uri="~/.qlib/qlib_data/us_data", region=REG_US)

from pathlib import Path
import pandas as pd

from pipeline.entry_logger import log_entries
from pipeline.forward_returns import update_forward_returns
from pipeline.prices import load_price_df


SAFE_DF_PATH = Path("artifacts/safe_entries.parquet")


def run_daily_logging(today):
    print(f"\n=== DAILY INFERENCE ({today}) ===")

    # 1. Load SAFE entries
    if not SAFE_DF_PATH.exists():
        print("❌ No SAFE entries found. Did you run top_long_short.py?")
        return

    safe_df = pd.read_parquet(SAFE_DF_PATH)
    print(f"Loaded SAFE entries: {len(safe_df)} symbols")

    today_entries = safe_df.to_dict(orient="records")
    symbols = sorted(safe_df["symbol"].unique())

    # 2. Load today's prices
    print("Loading today's prices…")
    price_today_df = load_price_df(symbols, today, today)

    # 3. Log entries
    print("Logging entries (duplicate-safe)…")
    updated_log = log_entries(today, today_entries, price_today_df)
    print(f"Entry log now contains {len(updated_log)} total rows")

    # 4. Load full price history
    print("Loading full price history for forward returns…")
    price_full_df = load_price_df(symbols, "2024-01-01", today)

    # 5. Update forward returns
    print("Updating forward returns…")
    updated_log = update_forward_returns(price_full_df, today)

    print("Done.")
    print(f"Forward returns updated for {today}.\n")


if __name__ == "__main__":
    from pipeline.utils import get_last_trading_day
    today = get_last_trading_day()
    run_daily_logging(today)