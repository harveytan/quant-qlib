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
TOP_N = 2
BOT_N = 2
NUM_COHORTS = 10
HORIZONS = [5, 10, 20]
CSV_PATH = "cohort_returns.csv"

# ----------------------------
# Main
# ----------------------------
def main():
    with open("trained_model_2.pkl", "rb") as f:
        model = pickle.load(f)

    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    handler = Alpha158(
        instruments=INSTRUMENTS,
        start_time=START_DATE,
        end_time=None,
        label=["Ref($close, -5) / Ref($close, 0) - 1"]
    )
    dataset = DatasetH(handler, {"test": (START_DATE, None)})

    # X_test = dataset.prepare("test", col_set="feature").dropna(axis=1, how="all").dropna(axis=0, how="any")
    # preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration", None))
    # df_pred = pd.DataFrame({"pred": preds}, index=X_test.index)
    preds = model.predict(dataset, "test")
    X_test = dataset.prepare("test", col_set="feature")
    df_pred = pd.DataFrame({"pred": preds}, index=X_test.index)




    instruments = D.instruments(market='all')
    close = D.features(instruments, ["$close"], start_time=START_DATE, end_time=None)["$close"]
    close = close.swaplevel().sort_index()
    close = close.unstack().ffill().bfill().stack()
    close = close.loc[df_pred.index]

    all_dates = sorted(df_pred.index.get_level_values("datetime").unique())
    latest_date = all_dates[-1]
    print(f"✅ Latest available date in features: {latest_date.date()}")

    recent_rebalances = []
    latest_idx = all_dates.index(latest_date)
    for i in range(NUM_COHORTS):
        idx = latest_idx - i * HOLDING_DAYS
        if idx >= 0:
            recent_rebalances.append(all_dates[idx])
    recent_rebalances = sorted(recent_rebalances)

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Date", "Horizon", "BuyReturn", "SellReturn", "Spread", "BuyCoverage", "SellCoverage", "RR_Ratio"])
        for dt in recent_rebalances:
            day_slice = df_pred.loc[dt]
            ranked = day_slice.sort_values("pred", ascending=False)

            buys = ranked.head(TOP_N)
            sells = ranked.tail(BOT_N)

            # ✅ Format with scores
            buys_fmt = [f"{inst}({score:+.4f})" for inst, score in buys["pred"].items()]
            sells_fmt = [f"{inst}({score:+.4f})" for inst, score in sells["pred"].items()]

            print(f"\n📅 {dt.date()} — Trade List — {HOLDING_DAYS} holding days")
            print("  Buys:", ", ".join(buys_fmt))
            print("  Sells:", ", ".join(sells_fmt))

            valid_horizons = [h for h in HORIZONS if all_dates.index(dt) + h < len(all_dates)]
            if not valid_horizons:
                continue

            for horizon in valid_horizons:
                exit_idx = all_dates.index(dt) + horizon
                exit_date = all_dates[exit_idx]
                buy_returns, sell_returns = [], []
                for inst in buys.index:
                    try:
                        entry = close.loc[(dt, inst)]
                        exitp = close.loc[(exit_date, inst)]
                        buy_returns.append(exitp / entry - 1)
                    except Exception:
                        pass
                for inst in sells.index:
                    try:
                        entry = close.loc[(dt, inst)]
                        exitp = close.loc[(exit_date, inst)]
                        sell_returns.append(exitp / entry - 1)
                    except Exception:
                        pass

                if len(buys) == 0 or len(sells) == 0:
                    print("⚠️ Skipping attribution — empty cohort")
                    continue

                if len(buy_returns) >= TOP_N and len(sell_returns) >= BOT_N:
                    avg_buy = pd.Series(buy_returns).mean()
                    avg_sell = pd.Series(sell_returns).mean()
                    spread = avg_buy - avg_sell

                    # ✅ Calculate realized volatility and reward-to-risk ratio
                    cohort_returns = buy_returns + [-r for r in sell_returns]
                    volatility = pd.Series(cohort_returns).std()
                    rr_ratio = abs(spread) / volatility if volatility > 0 else float("inf")

                    flag = "✅" if rr_ratio >= 2 else "⚠️"
                    print(f"    ⏱ {horizon}d → Buy: {avg_buy:+.1%} ({len(buy_returns)}), Sell: {avg_sell:+.1%} ({len(sell_returns)}), Spread: {spread:+.1%}, Vol: {volatility:.2%}, R/R: {rr_ratio:.2f} {flag}")

                    writer.writerow([
                        dt.date(), horizon,
                        f"{avg_buy:.6f}", f"{avg_sell:.6f}", f"{spread:.6f}",
                        len(buy_returns), len(sell_returns),
                        f"{rr_ratio:.2f}"
                    ])
                else:
                    print(f"    ⏱ {horizon}d → insufficient data for realized return")

if __name__ == "__main__":
    main()