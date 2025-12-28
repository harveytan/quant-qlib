import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from utils import prints
import qlib
from qlib.data import D

def main():

    # ✅ Initialize Qlib
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    # ✅ Load trained model
    with open("trained_model_2.pkl", "rb") as f:
        obj = pickle.load(f)
    model = obj["model"]
    training_columns = obj["columns"]
    # training columns: ['$open', '$high', '$low', '$close', '$ret_5d', '$vol_5d', '$rank_ret_5d', '$rank_vol_5d', '$ret_10d', '$vol_10d', '$rank_ret_10d',
    #                    '$rank_vol_10d', '$ret_20d', '$vol_20d', '$rank_ret_20d', '$rank_vol_20d', '$volume_log']


    # ✅ Display feature importance
    # importances = model.feature_importances_
    # feat = model.feature_name_
    # importance_dict = dict(zip(feat, importances))
    # sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    # printx("\n📊 Feature Importance (Descending):")
    # for name, score in sorted_importance:
    #     printx(f"  {name:<20} {score:>8}")


    # ✅ Define date range and instruments
    START_DATE = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d") #START_DATE = "2025-01-01"
    END_DATE = (datetime.today() + timedelta(days=1)).strftime("%Y-%m-%d")    #END_DATE = "2030-10-18"
    prints(f"Using date range: {START_DATE} to {END_DATE}", "trade_list_ensemble_log.txt")
    prints(f"Using training columns: {training_columns}", "trade_list_ensemble_log.txt")

    instrument_path = r"C:/Users/harve/.qlib/qlib_data/us_data/instruments/all.txt"
    with open(instrument_path, "r") as f:
        instrumentx = [line.strip().split("\t")[0] for line in f if line.strip()]
    # ✅ Load features and realized returns
    raw_fields = ["$open", "$high", "$low", "$close", "$volume",
                "$ret_5d", "$vol_5d", "$rank_vol_5d", #"$rank_ret_5d",
                "$ret_10d", "$vol_10d", "$rank_ret_10d", "$rank_vol_10d",
                "$ret_20d", "$vol_20d", "$rank_ret_20d", "$rank_vol_20d","$days_since_ipo"]

    features = D.features(instruments=instrumentx, fields=raw_fields,
                        start_time=START_DATE, end_time=END_DATE)


    features["$volume_log"] = np.log1p(features["$volume"])
    features.drop(columns=["$volume"], inplace=True)

    diagnostic_cols = ["$ret_5d", "$days_since_ipo"]  # add more if needed
    cols_to_keep = training_columns + [c for c in diagnostic_cols if c in features.columns]

    features = features[cols_to_keep]

    # Slice only training columns for prediction
    X_for_model = features[training_columns]

    labels = D.features(instruments=instrumentx,
                        fields=["$ret_5d", "$ret_10d", "$ret_20d"],
                        start_time=START_DATE, end_time=END_DATE)


    # ✅ Score features
    # features = features.dropna(subset=training_columns)
    features["score"] = model.predict(X_for_model)

    df_combined = pd.concat([features, labels], axis=1)
    df_combined = df_combined.dropna(subset=["score"])

    # ✅ Attribution function
    def attribution(cohort, horizon="5"):
        col = f"$ret_{horizon}d"
        rr = cohort[col].dropna().values.mean()
        vol = cohort[col].dropna().values.std()
        rr_ratio = rr / vol if vol > 0 else 0
        return rr, vol, rr_ratio

    # ✅ Trade list loop
    latest_date = features.index.get_level_values("datetime").max()
    trade_dates = df_combined.index.get_level_values("datetime").unique()

    if latest_date not in trade_dates:
        trade_dates = trade_dates.append(pd.Index([latest_date]))

    for date in trade_dates.sort_values():

        if date not in df_combined.index.get_level_values("datetime"):
            continue

        df_day = df_combined.xs(date, level="datetime", drop_level=False)
        if df_day.empty or df_day.shape[0] < 4:
            continue

        df_day_sorted = df_day.sort_values("score", ascending=False)
        top = df_day_sorted.head(2)
        bottom = df_day_sorted.tail(2)

        prints(f"\n📅 {date.date()} — Trade List (Ensemble Model)", "trade_list_ensemble_log.txt")
        for horizon in ["5", "10", "20"]:
            col = f"$ret_{horizon}d"
            col_data = df_day[col]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]  # Take the first column explicitly
            if col_data.isna().all():
                continue
            buy_rr, buy_vol, buy_rrr = attribution(top, horizon)
            sell_rr, _, _ = attribution(bottom, horizon)
            spread = buy_rr - sell_rr
            prints(f"  ⏱ {horizon}d → Buy: {buy_rr:.2%}, Sell: {sell_rr:.2%}, Spread: {spread:.2%}, Vol: {buy_vol:.2%}, R/R: {buy_rrr:.2f}", "trade_list_ensemble_log.txt")

        #prints(f"  Buys: {top.index.get_level_values('instrument').tolist()} — Scores: {top['score'].tolist()}", "trade_list_ensemble_log.txt")
        #prints(f"  Sells: {bottom.index.get_level_values('instrument').tolist()} — Scores: {bottom['score'].tolist()}", "trade_list_ensemble_log.txt")
        def icon_score(score, is_buy=True):
            if is_buy and score >= 0.2:
                return f"{score:.4f} ✅"  # Strong buy
            elif is_buy and score >= 0.15:
                return f"{score:.4f} ✔️"  # buy
            elif not is_buy and score <= -0.2:
                return f"{score:.4f} ❌"  # Strong sell
            else:
                return score     # Neutral

        # Print buys with ✅ for strong buy
        buy_instruments = top.index.get_level_values("instrument").tolist()
        buy_scores = [icon_score(s, is_buy=True) for s in top["score"].tolist()]
        prints(f"  Buys: {buy_instruments} — Scores: {buy_scores}", "trade_list_ensemble_log.txt")

        # Print sells with ❌ for strong sell
        sell_instruments = bottom.index.get_level_values("instrument").tolist()
        sell_scores = [icon_score(s, is_buy=False) for s in bottom["score"].tolist()]
        prints(f"  Sells: {sell_instruments} — Scores: {sell_scores}", "trade_list_ensemble_log.txt")

    labels_renamed = labels.rename(columns={
        "$ret_5d": "ret_5d_label",
        "$ret_10d": "ret_10d_label",
        "$ret_20d": "ret_20d_label",
    })

    df_combined = pd.concat([features, labels_renamed], axis=1)
    # Step 1: Reset index for easier slicing
    df_valid = df_combined.reset_index()

    # Step 2: Filter for dates with valid realized returns
    df_valid = df_valid[df_valid["$ret_5d"].notna()]
    # Keep forward returns for evaluation diagnostics

    # Step 3: Rename columns for clarity (optional)
    df_valid = df_valid.rename(columns={"score": "score", "$ret_5d": "label"})

    # Step 4: Create score buckets
    df_valid["bucket"] = pd.qcut(df_valid["score"], q=5, labels=False)

    # Step 5: Attribution by bucket
    bucket_returns = df_valid.groupby("bucket")["label"].mean()
    prints("📊 Average 5d return per score bucket:", "trade_list_ensemble_log.txt")
    prints(bucket_returns, "trade_list_ensemble_log.txt")
    hit_rate = df_valid.groupby("bucket")["label"].apply(lambda x: (x > 0).mean())
    prints("✅ Hit rate per bucket:", "trade_list_ensemble_log.txt")
    prints(hit_rate, "trade_list_ensemble_log.txt")

    IPO_CUTOFF = 600  # ~1 year of trading days

    df_valid["ipo_cohort"] = (df_valid["$days_since_ipo"] < IPO_CUTOFF).astype(int)

    # Attribution by cohort
    for cohort, name in [(0, "Core (≥250 days)"), (1, "IPO (<250 days)")]:
        df_c = df_valid[df_valid["ipo_cohort"] == cohort]
        prints(f"\n📊 {name} Attribution")
        prints(df_c.groupby("bucket")["label"].mean())
        prints(df_c.groupby("bucket")["label"].apply(lambda x: (x > 0).mean()))
if __name__ == "__main__":
    main()