import qlib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH


def main():
    """
    This gives you a visual diagnostic toolkit. If the equity curve is jagged but trending up, and the IC histogram is skewed positive, 
    you know the model is learning something. If it’s flat or symmetric around zero, it’s noise.
    """
    # ----------------------------
    # 1) Reload trained model
    # ----------------------------
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model reloaded from disk")

    # ----------------------------
    # 2) Initialize Qlib & load Recorder metrics
    # ----------------------------
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")
    exp = R.get_exp(experiment_name="dump_bin_lightgbm")

    recorders = exp.list_recorders()
    rec_id = next(iter(recorders)) if isinstance(recorders, dict) else recorders.iloc[0]["id"]
    rec = exp.get_recorder(rec_id)
    print("📊 Recorder metrics:", rec.list_metrics())

    # ----------------------------
    # 3) Prepare test slice
    # ----------------------------
    instruments = ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]  # swap in your 40-stock list
    handler = Alpha158(instruments=instruments, start_time="2025-01-01", end_time="2025-09-30")
    segments = {"test": ("2025-01-01", "2025-09-30")}
    dataset = DatasetH(handler, segments)

    X_test = dataset.prepare("test", col_set="feature")
    y_test = dataset.prepare("test", col_set="label").squeeze("columns")

    mask = y_test.notna()
    X_test, y_test = X_test[mask], y_test[mask]
    X_test = X_test.dropna(axis=1, how="all")

    print(f"Test slice: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    # ----------------------------
    # 4) Inference and IC metrics
    # ----------------------------
    preds = model.predict(X_test, num_iteration=getattr(model, "best_iteration", None))
    df_eval = pd.DataFrame({"pred": preds, "label": y_test}).dropna()

    agg_ic = df_eval["pred"].corr(df_eval["label"])
    ic_by_day = df_eval.groupby(level="datetime").apply(lambda g: g["pred"].corr(g["label"]))
    mean_ic, std_ic, coverage_days = ic_by_day.mean(), ic_by_day.std(), ic_by_day.count()
    rank_ic = df_eval["pred"].rank().corr(df_eval["label"].rank(), method="pearson")

    print(f"🔁 Aggregate IC (Pearson): {agg_ic}")
    print(f"📅 Per-day IC mean: {mean_ic}, std: {std_ic}, coverage days: {coverage_days}")
    print(f"🏁 Rank IC (Spearman via rank corr): {rank_ic}")

    # ----------------------------
    # 5) Simple daily long-short backtest
    # ----------------------------
    daily_returns = []

    for dt, g in df_eval.groupby(level="datetime"):
        if g.empty:
            continue

        ranks = g["pred"].rank(method="first")
        top = ranks >= ranks.quantile(0.9)
        bottom = ranks <= ranks.quantile(0.1)

        w = (g["pred"] - g["pred"].mean()) / (g["pred"].std(ddof=0) + 1e-9)
        w = w.where(top, 0.0) + (-w).where(bottom, 0.0)

        if w.abs().sum() > 0:
            w = w / w.abs().sum()
        else:
            continue

        day_ret = (w * g["label"]).sum()
        daily_returns.append((dt, day_ret))

    rets = pd.Series([r for _, r in daily_returns],
                     index=[dt for dt, _ in daily_returns]).sort_index()

    if not rets.empty:
        cagr = (1 + rets.mean())**252 - 1
        sharpe = np.sqrt(252) * rets.mean() / (rets.std(ddof=0) + 1e-9)
        max_dd = (rets.cumsum() - rets.cumsum().cummax()).min()

        print(f"📈 Backtest results: CAGR={cagr:.2%}, Sharpe={sharpe:.2f}, "
              f"MaxDD={max_dd:.2%}, Mean daily={rets.mean():.4%}")

        # ----------------------------
        # 6) Plots
        # ----------------------------
        cumrets = rets.cumsum()
        drawdown = cumrets - cumrets.cummax()

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        # Equity curve
        axes[0].plot(cumrets.index, cumrets.values, label="Equity Curve")
        axes[0].set_title("Equity Curve (Cumulative Returns)")
        axes[0].legend()

        # Drawdown
        axes[1].fill_between(drawdown.index, drawdown.values, 0, color="red", alpha=0.4)
        axes[1].set_title("Drawdown")

        # IC histogram
        axes[2].hist(ic_by_day.dropna(), bins=20, alpha=0.7, color="blue")
        axes[2].set_title("Per-day IC Distribution")

        plt.tight_layout()
        plt.show()

    else:
        print("⚠️ No valid daily returns computed — check data slice or instrument list.")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()