import qlib
import pickle
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# ----------------------------
# Config
# ----------------------------
START_DATE = "2025-01-01"
INSTRUMENTS = "all"    # uses your 40-stock bundle
HOLDING_DAYS = 20      # ~1 month; set 5 for weekly
TOP_N = 10             # top 10 buys
BOT_N = 10             # bottom 10 sells
NUM_COHORTS = 10       # how many past cohorts to show (plus today)

# ----------------------------
# Main
# ----------------------------
def main():
    # Reload trained model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    handler = Alpha158(
        instruments=INSTRUMENTS,
        start_time=START_DATE,
        end_time=None,
        label=["Ref($close, -5) / Ref($close, 0) - 1"]  # keep label for Qlib, ignore later
    )
    dataset = DatasetH(handler, {"test": (START_DATE, None)})

    # Prepare features only
    X_test = dataset.prepare("test", col_set="feature")
    X_test = X_test.dropna(axis=1, how="all")
    X_test = X_test.dropna(axis=0, how="any")

    # Predict across all available dates
    preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration", None))
    df_pred = pd.DataFrame({"pred": preds}, index=X_test.index)

    # Get all available trading dates
    all_dates = sorted(df_pred.index.get_level_values("datetime").unique())
    latest_date = all_dates[-1]
    print(f"✅ Latest available date in features: {latest_date.date()}")

    # Step through dates in increments of HOLDING_DAYS
    rebalance_dates = all_dates[::HOLDING_DAYS]

    # Ensure we include the very latest date even if it's not aligned to HOLDING_DAYS
    if rebalance_dates[-1] != latest_date:
        rebalance_dates.append(latest_date)

    # Take the last N cohorts
    recent_rebalances = rebalance_dates[-NUM_COHORTS:]

    # Print trade lists for each rebalance date
    for dt in recent_rebalances:
        day_slice = df_pred.loc[dt]
        ranked = day_slice.sort_values("pred", ascending=False)

        buys = ranked.head(TOP_N).index.tolist()
        sells = ranked.tail(BOT_N).index.tolist()

        print(f"\n📅 {dt.date()} — Trade List")
        print("  Buys:", ", ".join(buys))
        print("  Sells:", ", ".join(sells))

if __name__ == "__main__":
    main()