import qlib
import pickle
import pandas as pd
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

def main():
    # ----------------------------
    # Reload trained model
    # ----------------------------
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    print("✅ Model reloaded from disk")

    # ----------------------------
    # Reload metrics from Recorder
    # ----------------------------
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    exp = R.get_exp(experiment_name="dump_bin_lightgbm")

    # List all recorders (returns dict keyed by recorder_id)
    recorders = exp.list_recorders()
    for rec_id in recorders:
        rec = exp.get_recorder(rec_id)
        info = rec.list_metrics()
        print(f"Recorder {rec_id} | Metrics: {info}")
    # Pick the first recorder_id
    rec_id = next(iter(recorders))
    rec = exp.get_recorder(rec_id)

    print("Metrics:", rec.list_metrics())
    metrics = rec.list_metrics()
    print("📊 Metrics from Recorder:", metrics)

    # ----------------------------
    # Optional: run inference again
    # ----------------------------
    handler = Alpha158(instruments=["AAPL","MSFT"], start_time="2025-01-01", end_time="2025-09-30")
    segments = {"test": ("2025-01-01", "2025-09-30")}
    dataset = DatasetH(handler, segments)

    X_test = dataset.prepare("test", col_set="feature")
    y_test = dataset.prepare("test", col_set="label").squeeze("columns")

    # Drop NaN labels only
    mask = y_test.notna()
    X_test, y_test = X_test[mask], y_test[mask]

    # Drop all-NaN columns
    X_test = X_test.dropna(axis=1, how="all")
    print(f"Test slice: {X_test.shape[0]} rows, {X_test.shape[1]} features")
    if X_test.empty:
        raise RuntimeError("Empty test features after cleaning; adjust instruments/segments.")



    preds = model.predict(X_test, num_iteration=model.best_iteration)
    df_eval = pd.DataFrame({"pred": preds, "label": y_test}).dropna()
    # Aggregate IC (Pearson)
    agg_ic = df_eval["pred"].corr(df_eval["label"])

    # Per-day IC (Pearson)
    if hasattr(df_eval.index, "names") and "datetime" in df_eval.index.names:
        ic_by_day = df_eval.groupby(level="datetime").apply(lambda g: g["pred"].corr(g["label"]))
    else:
        # If index is flat, try to use a 'datetime' column if available; otherwise skip
        ic_by_day = pd.Series(dtype=float)

    mean_ic = ic_by_day.mean() if not ic_by_day.empty else np.nan
    std_ic = ic_by_day.std() if not ic_by_day.empty else np.nan
    coverage_days = ic_by_day.count() if not ic_by_day.empty else 0

    # Rank IC (Spearman) — more robust to monotonic relationships
    rank_ic = df_eval["pred"].rank().corr(df_eval["label"].rank(), method="pearson")

    print(f"🔁 Aggregate IC (Pearson): {agg_ic}")
    print(f"📅 Per-day IC mean: {mean_ic}, std: {std_ic}, coverage days: {coverage_days}")
    print(f"🏁 Rank IC (Spearman via rank corr): {rank_ic}")

    # ----------------------------
    # 5) Quick feature sanity check (optional)
    # ----------------------------
    stds = X_test.std(numeric_only=True)
    all_nan_cols = stds[stds.isna()].index.tolist()
    near_const_cols = stds[(stds <= 1e-9) & (~stds.isna())].index.tolist()
    print(f"🧪 All-NaN columns in test: {len(all_nan_cols)}")
    print(f"🧪 Near-constant columns in test: {len(near_const_cols)}")



if __name__ == "__main__":
    main()