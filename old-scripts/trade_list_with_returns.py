import qlib
import pickle
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

START_DATE = "2025-01-01"
INSTRUMENTS = "all"
HOLDING_DAYS = 20
TOP_N = 10
BOT_N = 10
NUM_COHORTS = 10

def main():
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    handler = Alpha158(instruments=INSTRUMENTS,
                       start_time=START_DATE,
                       end_time=None,
                       label=["Ref($close, -5) / Ref($close, 0) - 1"])
    dataset = DatasetH(handler, {"test": (START_DATE, None)})

    X_test = dataset.prepare("test", col_set="feature").dropna(axis=1, how="all").dropna(axis=0, how="any")
    preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration", None))
    df_pred = pd.DataFrame({"pred": preds}, index=X_test.index)

    # Get close prices for realized return calc
    close = dataset.prepare("test", col_set="label")  # or use handler.fetch_col("close")
    close = close.droplevel(0, axis=1) if isinstance(close.columns, pd.MultiIndex) else close

    all_dates = sorted(df_pred.index.get_level_values("datetime").unique())
    rebalance_dates = all_dates[::HOLDING_DAYS]
    if rebalance_dates[-1] != all_dates[-1]:
        rebalance_dates.append(all_dates[-1])
    recent_rebalances = rebalance_dates[-NUM_COHORTS:]

    for dt in recent_rebalances:
        day_slice = df_pred.loc[dt]
        ranked = day_slice.sort_values("pred", ascending=False)
        buys = ranked.head(TOP_N).index.tolist()
        sells = ranked.tail(BOT_N).index.tolist()

        # Compute realized returns
        exit_idx = all_dates.index(dt) + HOLDING_DAYS
        if exit_idx < len(all_dates):
            exit_date = all_dates[exit_idx]
            buy_returns = []
            sell_returns = []
            for inst in buys:
                try:
                    r = close.loc[exit_date, inst] / close.loc[dt, inst] - 1
                    buy_returns.append(r)
                except KeyError:
                    pass
            for inst in sells:
                try:
                    r = close.loc[exit_date, inst] / close.loc[dt, inst] - 1
                    sell_returns.append(r)
                except KeyError:
                    pass
            avg_buy = pd.Series(buy_returns).mean() if buy_returns else None
            avg_sell = pd.Series(sell_returns).mean() if sell_returns else None
        else:
            avg_buy = avg_sell = None

        print(f"\n📅 {dt.date()} — Trade List — {HOLDING_DAYS} holding days")
        print("  Buys:", ", ".join(buys))
        print("  Sells:", ", ".join(sells))
        if avg_buy is not None:
            print(f"  📈 Avg Buy Return over {HOLDING_DAYS}d: {avg_buy:.2%}")
            print(f"  📉 Avg Sell Return over {HOLDING_DAYS}d: {avg_sell:.2%}")
            print(f"  Spread (Buy - Sell): {(avg_buy - avg_sell):.2%}")

if __name__ == "__main__":
    main()