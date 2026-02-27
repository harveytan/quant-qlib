# pipeline/daily_logger.py

from pathlib import Path
import pandas as pd

from pipeline.entry_logger import log_entries
from pipeline.forward_returns import update_forward_returns
from pipeline.prices import load_price_df
from pipeline.utils import prints

SAFE_DF_PATH = Path("artifacts/safe_entries.parquet")


def run_daily_logging(today):
    prints(f"\n=== DAILY LOGGING ({today}) ===")

    # 1. Load SAFE entries
    if not SAFE_DF_PATH.exists():
        prints("❌ No SAFE entries found. Did you run top_long_short.py?")
        return

    safe_df = pd.read_parquet(SAFE_DF_PATH)
    prints(f"Loaded SAFE entries: {len(safe_df)} symbols")

    today_entries = safe_df.to_dict(orient="records")
    symbols = sorted(safe_df["symbol"].unique())

    # 2. Load today's prices
    prints("Loading today's prices…")
    price_today_df = load_price_df(symbols, today, today)

    # 3. Log entries
    prints("Logging entries (duplicate-safe)…")
    updated_log = log_entries(today, today_entries, price_today_df)
    prints(f"Entry log now contains {len(updated_log)} total rows")

    # 4. Load full price history
    prints("Loading full price history for forward returns…")
    price_full_df = load_price_df(symbols, "2024-01-01", today)

    # 5. Update forward returns
    prints("Updating forward returns…")
    updated_log = update_forward_returns(price_full_df, today)

    prints("Done.")
    prints(f"Forward returns updated for {today}.\n")