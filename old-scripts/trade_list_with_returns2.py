import qlib
import pickle
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data import D
import csv

# ----------------------------
# Config
# ----------------------------
START_DATE = "2025-01-01"
INSTRUMENTS = "all"
HOLDING_DAYS = 10
TOP_N = 10
BOT_N = 10
NUM_COHORTS = 10
HORIZONS = [5, 10, 20]
CSV_PATH = "cohort_returns.csv"

# ----------------------------
# Main
# ----------------------------
def main():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    handler = Alpha158(
        instruments=INSTRUMENTS,
        start_time=START_DATE,
        end_time=None,
        label=["Ref($close, -5) / Ref($close, 0) - 1"]
    )
    dataset = DatasetH(handler, {"test": (START_DATE, None)})

    X_test = dataset.prepare("test", col_set="feature").dropna(axis=1, how="all").dropna(axis=0, how="any")
    preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration", None))
    df_pred = pd.DataFrame({"pred": preds}, index=X_test.index)

    instruments = D.instruments(market='all')
    close = D.features(instruments, ["$close"], start_time=START_DATE, end_time=None)["$close"]
    close = close.swaplevel().sort_index()  # make index (datetime, instrument)
    close = close.unstack().ffill().bfill().stack()  # fill missing prices
    close = close.loc[df_pred.index]  # restrict to prediction index

    all_dates = sorted(df_pred.index.get_level_values("datetime").unique())
    latest_date = all_dates[-1]
    print(f"✅ Latest available date in features: {latest_date.date()}")

    # Step backward in HOLDING_DAYS increments
    recent_rebalances = []
    latest_idx = all_dates.index(latest_date)
    for i in range(NUM_COHORTS):
        idx = latest_idx - i * HOLDING_DAYS
        if idx >= 0:
            recent_rebalances.append(all_dates[idx])
    recent_rebalances = sorted(recent_rebalances)

    # Prepare CSV writer
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Horizon", "BuyReturn", "SellReturn", "Spread", "BuyCoverage", "SellCoverage"])

        for dt in recent_rebalances:
            day_slice = df_pred.loc[dt]
            ranked = day_slice.sort_values("pred", ascending=False)
            buys = ranked.head(TOP_N).index.tolist()
            sells = ranked.tail(BOT_N).index.tolist()

            print(f"\n📅 {dt.date()} — Trade List — {HOLDING_DAYS} holding days")
            print("  Buys:", ", ".join(buys))
            print("  Sells:", ", ".join(sells))

            valid_horizons = [h for h in HORIZONS if all_dates.index(dt) + h < len(all_dates)]
            if not valid_horizons:
                continue

            for horizon in valid_horizons:
                exit_idx = all_dates.index(dt) + horizon
                exit_date = all_dates[exit_idx]
                buy_returns, sell_returns = [], []
                for inst in buys:
                    try:
                        entry = close.loc[(dt, inst)]
                        exitp = close.loc[(exit_date, inst)]
                        buy_returns.append(exitp / entry - 1)
                    except Exception:
                        pass
                for inst in sells:
                    try:
                        entry = close.loc[(dt, inst)]
                        exitp = close.loc[(exit_date, inst)]
                        sell_returns.append(exitp / entry - 1)
                    except Exception:
                        pass
                if len(buy_returns) >= 5 and len(sell_returns) >= 5:
                    avg_buy = pd.Series(buy_returns).mean()
                    avg_sell = pd.Series(sell_returns).mean()
                    spread = avg_buy - avg_sell
                    print(f"    ⏱ {horizon}d → Buy: {avg_buy:+.1%} ({len(buy_returns)}), Sell: {avg_sell:+.1%} ({len(sell_returns)}), Spread: {spread:+.1%}")
                    writer.writerow([
                        dt.date(), horizon,
                        f"{avg_buy:.6f}", f"{avg_sell:.6f}", f"{spread:.6f}",
                        len(buy_returns), len(sell_returns)
                    ])
                else:
                    print(f"    ⏱ {horizon}d → insufficient data for realized return")

if __name__ == "__main__":
    main()