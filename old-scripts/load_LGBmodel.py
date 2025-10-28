import qlib
from qlib.config import REG_US
import pickle
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from datetime import datetime, timedelta
import pandas as pd
from qlib.data import D
import json
import matplotlib.pyplot as plt
from qlib.workflow import R

def main():
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region=REG_US)

    # Step 1: load trained model
    with open("manual_artifacts/model.pkl", "rb") as f:
         booster = pickle.load(f)

    model = LGBModel()
    model.model = booster  # Inject trained booster

    with open("my_symbols.txt", "r") as f:
         my_stocks = [line.strip() for line in f if line.strip()]



    # Step 2: prepare dataset for september 24, 2025
    target_date = "2025-09-24"
    day_before = (datetime.strptime(target_date, "%Y-%m-%d") - timedelta(days=5)).strftime("%Y-%m-%d")

    handler = Alpha158(instruments="all", start_time="2016-01-01", end_time="2025-09-24")
    dataset = DatasetH(handler=handler, segments={
        "predict": (day_before, target_date)
    })

    #Step 3: predict and rank instruments
    preds = model.predict(dataset, segment="predict")
    print(list(preds.items())[:5])

    # Convert to DataFrame
    ranked_df = pd.DataFrame([
        {"datetime": dt, "instrument": inst, "prediction": score}
        for (dt, inst), score in preds.items()
    ])

    # Filter to your 40-stock universe
    ranked_df = ranked_df[ranked_df["instrument"].isin(my_stocks)]

    # Sort and select top/bottom
    ranked_df = ranked_df.sort_values(by=["datetime", "prediction"], ascending=[True, False])

    top10 = ranked_df.groupby("datetime").head(10)
    bottom10 = ranked_df.groupby("datetime").tail(10)
    print("Top10 shape:", top10.shape)
    print("Bottom10 shape:", bottom10.shape)

    # Step 4: simulate long-short strategy:
    def simulate_returns(df, long=True):
        returns = []
        calendar = D.calendar(start_time="2025-01-01", end_time="2025-12-31")
        calendar = pd.Series(calendar)

        def get_prev_trading_day(date):
            date = pd.to_datetime(date)
            prev_dates = calendar[calendar < date]
            return prev_dates.iloc[-1] if not prev_dates.empty else None

        for dt, group in df.groupby("datetime"):
            instruments = group["instrument"].tolist()
            start = get_prev_trading_day(dt)
            end = pd.to_datetime(dt)

            if start is None:
                print(f"⚠️ No prior trading day for {dt}")
                continue

            prev_prices = D.features(instruments, ["$close"], start_time=start, end_time=start)
            curr_prices = D.features(instruments, ["$close"], start_time=end, end_time=end)

            rows = []
            for inst in instruments:
                try:
                    prev = prev_prices.loc[(inst, start), "$close"]
                    curr = curr_prices.loc[(inst, end), "$close"]
                    rows.append({
                        "instrument": inst,
                        "prev_close": float(prev),
                        "close": float(curr)
                    })
                except KeyError:
                    print(f"⚠️ Missing data for {inst} on {start} or {end}")

            df = pd.DataFrame(rows)

            if not df.empty:
                df["return"] = (df["close"] - df["prev_close"]) / df["prev_close"]
                df["prev_close"] = df["prev_close"].astype(float)
                df["close"] = df["close"].astype(float)
                df["return"] = (df["close"] - df["prev_close"]) / df["prev_close"]                
                ret = df["return"].mean() if long else -df["return"].mean()
                print(f"{dt} return: {ret:.4f}")
            else:
                ret = float("nan")
                print(f"{dt} return: NaN (no valid prices)")

            returns.append(ret)
        return pd.Series(returns)

    long_returns = simulate_returns(top10, long=True)
    short_returns = simulate_returns(bottom10, long=False)
    net_returns = long_returns + short_returns


    print("long_returns", long_returns)
    print("short_returns", short_returns)
    print("net_returns", net_returns)

    # plot cumulative returns
    plt.plot(net_returns.cumsum(), label="Net Strategy Return")
    plt.plot(long_returns.cumsum(), label="Long Only")
    plt.plot(short_returns.cumsum(), label="Short Only")
    plt.legend()
    plt.title("Strategy Performance")
    plt.show()    

    # Step 5: visualize feature importance
    # Load importance scores
    with open("manual_artifacts/feature_importance.json", "r") as f:
         importance = json.load(f)

    # recorder_dict = R.list_recorders()

    # recorder_objs = list(recorder_dict.values())

    # # Filter for FINISHED recorders
    # finished = [r for r in recorder_objs if getattr(r, "info", {}).get("status") == "FINISHED"]

    # # Sort by end_time descending
    # finished_sorted = sorted(finished, key=lambda r: getattr(r, "end_time", ""), reverse=True)

    # # Pick the latest
    # latest_finished = finished_sorted[0] if finished_sorted else None

    # if latest_finished:
    #    print(f"✅ Latest finished recorder: {latest_finished.id}")
    #    recorder = R.get_recorder(recorder_id=latest_finished.id)
    #    print(recorder.list_artifacts())
    #    params = recorder.load_object("params")
    #    feature_names = params.get("feature_names", [])
    # else:
    #    print("⚠️ No finished recorders found.")
    #    feature_names = []

    # Convert and sort
    # sorted_items = sorted(importance.items(), key=lambda x: float(x[1]), reverse=True)
    # top_features = sorted_items[:10]

    # # Unpack for plotting
    # features = [f[0] for f in top_features]
    # scores = [float(f[1]) for f in top_features]

    # # Plot
    # plt.figure(figsize=(10, 6))
    # plt.barh(features, scores, color="steelblue")
    # plt.xlabel("Importance Score")
    # plt.title("Top 10 Feature Importances")
    # plt.gca().invert_yaxis()
    # plt.tight_layout()
    # plt.show()

    # Map columnXX to real names
    feature_names = [
        "$close", "$volume", "Ref($close, -1)", "MA($close, 5)", "MA($volume, 10)",
        "STD($close, 5)", "MACD", "RSI", "BOLL", "ATR", "ADX", "OBV", "CCI", "ROC",
        "WILLR", "TSF", "KDJ_K", "KDJ_D", "KDJ_J", "DMI"
    ]
    feature_names = [
        # Price lags
        *[f"Ref($close, -{i})" for i in range(1, 21)],

        # Volume lags
        *[f"Ref($volume, -{i})" for i in range(1, 21)],

        # Moving averages
        *[f"MA($close, {i})" for i in [5, 10, 20, 30, 60]],
        *[f"MA($volume, {i})" for i in [5, 10, 20, 30, 60]],

        # Standard deviation
        *[f"STD($close, {i})" for i in [5, 10, 20, 30, 60]],

        # Momentum indicators
        "MACD", "RSI", "CCI", "ROC", "WILLR", "ADX", "OBV", "ATR", "TSF",

        # Bollinger Bands
        "BOLL_UPPER", "BOLL_LOWER", "BOLL_MID",

        # KDJ
        "KDJ_K", "KDJ_D", "KDJ_J",

        # DMI
        "DMI_PLUS", "DMI_MINUS", "DMI_ADX",

        # Calendar features
        "day_of_week", "day_of_month", "month", "is_holiday",

        # Binary flags
        "is_earnings_day", "is_dividend_day", "is_split_day",

        # Custom features (examples)
        "close_to_high_ratio", "volume_spike", "price_gap", "volatility_5d", "volatility_20d",

        # Lagged returns
        *[f"Return_{i}d" for i in range(1, 21)],

        # Rolling z-scores
        *[f"ZScore($close, {i})" for i in [5, 10, 20]],

        # Rolling min/max
        *[f"Max($close, {i})" for i in [5, 10, 20]],
        *[f"Min($close, {i})" for i in [5, 10, 20]],

        # Price ratios
        "close/open", "high/low", "close/prev_close", "volume/avg_volume_20d",

        # Technical flags
        "is_breakout", "is_reversal", "is_trending_up", "is_trending_down",

        # Additional lags
        *[f"Ref($high, -{i})" for i in range(1, 11)],
        *[f"Ref($low, -{i})" for i in range(1, 11)],
        *[f"Ref($open, -{i})" for i in range(1, 11)],

        # Final padding (if needed)
        "custom_1", "custom_2", "custom_3", "custom_4", "custom_5", "custom_6"
    ]
    print(feature_names)
    mapped = {
        feature_names[int(k.replace("Column_", ""))]: v
        for k, v in importance.items()
        if k.startswith("Column_")
    }


    # Convert to Series and sort
    fi = pd.Series(mapped).sort_values(ascending=False)
    top10 = fi.head(10)
    print(top10)

    # Plot top 10
    top10 = fi.head(10)
    top10.plot(kind="barh", title="Top 10 Feature Importance", figsize=(10, 6), color="skyblue")
    plt.gca().invert_yaxis()
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()