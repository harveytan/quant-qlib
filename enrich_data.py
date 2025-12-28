import pandas as pd
import os

# Source and target directories
source_dir = r"C:\Users\harve\.qlib\stock_data\normalize\us_data"
target_dir = r"C:\Users\harve\.qlib\stock_data\normalize\us_data_enriched"
os.makedirs(target_dir, exist_ok=True)

# Collect all dataframes for cross-sectional ranking
enriched_dfs = []

for fname in os.listdir(source_dir):
    if not fname.endswith(".csv"):
        continue

    df = pd.read_csv(os.path.join(source_dir, fname))
    df["symbol"] = fname.replace(".csv", "")

    # Compute forward returns
    df["ret_5d"] = df["close"].shift(-5) / df["close"] - 1
    df["ret_10d"] = df["close"].shift(-10) / df["close"] - 1
    df["ret_20d"] = df["close"].shift(-20) / df["close"] - 1

    # Compute volatility
    df["vol_5d"] = df["close"].pct_change().rolling(5).std()
    df["vol_10d"] = df["close"].pct_change().rolling(10).std()
    df["vol_20d"] = df["close"].pct_change().rolling(20).std()

    # Ensemble label: weighted blend of multi-horizon returns
    df["ensemble_label"] = (
        0.5 * df["ret_5d"] +
        0.3 * df["ret_10d"] +
        0.2 * df["ret_20d"]
    )
    # # Count trading days since the first available date for each symbol
    # df["days_since_ipo"] = (pd.to_datetime(df["date"]) - pd.to_datetime(df["date"].min())).dt.days

    # ✅ New Feature: Days since IPO
    # Count trading days since the first available date for each symbol
    df["days_since_ipo"] = (
        pd.to_datetime(df["date"]) - pd.to_datetime(df["date"].min())
    ).dt.days

    enriched_dfs.append(df)

# Concatenate for cross-sectional ranking
full_df = pd.concat(enriched_dfs)

# Rank per date
#full_df["rank_ret_5d"] = full_df.groupby("date")["ret_5d"].rank()
full_df["rank_vol_5d"] = full_df.groupby("date")["vol_5d"].rank()

#full_df["rank_ret_10d"] = full_df.groupby("date")["ret_10d"].rank()
full_df["rank_vol_10d"] = full_df.groupby("date")["vol_10d"].rank()

#full_df["rank_ret_20d"] = full_df.groupby("date")["ret_20d"].rank()
full_df["rank_vol_20d"] = full_df.groupby("date")["vol_20d"].rank()

# Save back to individual files
for symbol, df_symbol in full_df.groupby("symbol"):
    df_symbol.to_csv(os.path.join(target_dir, f"{symbol}.csv"), index=False)