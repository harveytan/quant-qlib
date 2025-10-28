import qlib
import pickle
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.data import D
import numpy as np

# ----------------------------
# Config
# ----------------------------
START_DATE = "2025-01-01"
INSTRUMENTS = "all"
HOLDING_DAYS = 10
TOP_N = 10

# ✅ Your tradable universe
MY_PORTFOLIO = ["APP", "PLTR", "MSFT", "NVDA", "HOOD", "GOOG", "NFLX", "INTC", "COST", "UNH", "RDDT", "AMZN", "RDDT", "META", "MSTR"]

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
    close = close.swaplevel().sort_index()
    close = close.unstack().ffill().bfill().stack()
    close = close.loc[df_pred.index]

    all_dates = sorted(df_pred.index.get_level_values("datetime").unique())
    latest_date = all_dates[-1]
    print(f"✅ Latest available date in features: {latest_date.date()}")

    portfolio_returns = []

    for dt in all_dates:
        if all_dates.index(dt) + HOLDING_DAYS >= len(all_dates):
            continue
        exit_date = all_dates[all_dates.index(dt) + HOLDING_DAYS]
        day_slice = df_pred.loc[dt]

        # ✅ Filter to your portfolio only
        ranked = day_slice.sort_values("pred", ascending=False)
        ranked = ranked[ranked.index.isin(MY_PORTFOLIO)]
        buys = ranked.head(TOP_N)

        returns = []
        for inst in buys.index:
            try:
                entry = close.loc[(dt, inst)]
                exitp = close.loc[(exit_date, inst)]
                ret = exitp / entry - 1
                returns.append(ret)
            except Exception:
                pass

        if len(returns) >= 5:
            avg_ret = np.mean(returns)
            volatility = np.std(returns)
            rr_ratio = abs(avg_ret) / volatility if volatility > 0 else float("inf")

            if rr_ratio >= 2:
                portfolio_returns.append((dt, avg_ret))
            else:
                if len(returns) >= 5:
                    top_score = ranked["pred"].iloc[0]
                    if top_score >= 0.01:
                        print(f"⚠️ Fallback {dt.date()} — R/R: {rr_ratio:.2f}, Score: {top_score:+.4f} ≥ 0.01 — trading anyway")
                        portfolio_returns.append((dt, avg_ret))
                    else:
                        print(f"❌ Skipping {dt.date()} — R/R: {rr_ratio:.2f}, Score: {top_score:+.4f} < 0.01")
                else:
                    print(f"❌ Skipping {dt.date()} — insufficient coverage")

    # ----------------------------
    # Performance Summary
    # ----------------------------
    df_pnl = pd.DataFrame(portfolio_returns, columns=["Date", "Return"])
    df_pnl["Cumulative"] = (1 + df_pnl["Return"]).cumprod()
    df_pnl["Drawdown"] = df_pnl["Cumulative"] / df_pnl["Cumulative"].cummax() - 1

    total_return = df_pnl["Cumulative"].iloc[-1] - 1
    volatility = df_pnl["Return"].std()
    sharpe = df_pnl["Return"].mean() / volatility * np.sqrt(252) if volatility > 0 else 0
    max_dd = df_pnl["Drawdown"].min()

    print("\n📊 Strategy Performance — Long Only (Filtered)")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Volatility: {volatility * np.sqrt(252):.2%}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2%}")
    print(f"Trade Cohorts: {len(df_pnl)}")

if __name__ == "__main__":
    main()