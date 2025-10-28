import qlib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH

# ----------------------------
# Config
# ----------------------------
START_DATE = "2025-01-01"
END_DATE   = "2025-09-30"
INSTRUMENTS = "all"          # use same as training
EXPERIMENT_NAME = "dump_bin_lightgbm"

REBALANCE = "weekly"         # "daily" or "weekly"
LONG_SHORT = True            # True = long-short; False = long-only
TOP_Q = 0.9
BOT_Q = 0.1
TURNOVER_COST_BPS = 10
MAX_NAME_WEIGHT = 0.05

# ----------------------------
# Helpers
# ----------------------------
def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / (s.std(ddof=0) + 1e-9)

def to_week_key(idx):
    return idx.to_period("W").astype(str)

def rebalance_groups(df, freq="weekly"):
    if freq == "weekly":
        idx = df.index.get_level_values("datetime")
        keys = to_week_key(idx)
        return [(k, g) for k, g in df.groupby(keys)]
    else:
        return list(df.groupby(level="datetime"))

def compute_turnover_cost(prev_w: pd.Series, w: pd.Series, bps: float) -> float:
    if prev_w is not None:
        aligned = prev_w.reindex(w.index).fillna(0.0)
        turnover = (w - aligned).abs().sum()
    else:
        turnover = w.abs().sum()
    return (bps / 10000.0) * turnover

# ----------------------------
# Backtest
# ----------------------------
def run_backtest(df_eval: pd.DataFrame) -> pd.Series:
    prev_weights = None
    daily_rets = []

    for key, g in rebalance_groups(df_eval, REBALANCE):
        if g.empty:
            continue

        ranks = g["pred"].rank(method="first")
        top = ranks >= ranks.quantile(TOP_Q)
        bottom = ranks <= ranks.quantile(BOT_Q)

        w = zscore(g["pred"])
        if LONG_SHORT:
            w = w.where(top, 0.0) + (-w).where(bottom, 0.0)
        else:
            w = w.where(top, 0.0)

        if w.abs().sum() == 0:
            continue
        w = w / w.abs().sum()
        w = w.clip(lower=-MAX_NAME_WEIGHT, upper=MAX_NAME_WEIGHT)

        period_cost = compute_turnover_cost(prev_weights, w, TURNOVER_COST_BPS)

        for dt, day in g.groupby(level="datetime"):
            day_ret = (w.reindex(day.index).fillna(0.0) * day["label"]).sum() - period_cost
            daily_rets.append((dt, day_ret))

        prev_weights = w

    rets = pd.Series([r for _, r in daily_rets],
                     index=[dt for dt, _ in daily_rets]).sort_index()
    return rets

def summarize_performance(rets: pd.Series) -> dict:
    if rets.empty:
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0, "MeanDaily": 0.0, "Days": 0}
    cagr = (1 + rets.mean())**252 - 1
    sharpe = np.sqrt(252) * rets.mean() / (rets.std(ddof=0) + 1e-9)
    drawdown = (rets.cumsum() - rets.cumsum().cummax())
    max_dd = drawdown.min()
    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "MeanDaily": rets.mean(),
        "Days": rets.shape[0],
    }

def plot_results(rets: pd.Series, ic_by_day: pd.Series):
    if rets.empty:
        print("⚠️ No returns to plot.")
        return

    cumrets = rets.cumsum()
    drawdown = cumrets - cumrets.cummax()

    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    axes[0].plot(cumrets.index, cumrets.values, label="Equity Curve")
    axes[0].set_title("Equity Curve (Cumulative Returns)")
    axes[0].legend()

    axes[1].fill_between(drawdown.index, drawdown.values.astype(float), 0.0,
                         color="red", alpha=0.4)
    axes[1].set_title("Drawdown")

    if not ic_by_day.empty:
        axes[2].hist(ic_by_day.dropna().astype(float), bins=20, alpha=0.7, color="blue")
        axes[2].set_title("Per-day IC Distribution")
    else:
        axes[2].text(0.5, 0.5, "No IC data", ha="center")

    plt.tight_layout()
    plt.show()

# ----------------------------
# Main
# ----------------------------
def main():
    # Reload model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    print("✅ Model reloaded from disk")

    # Init Qlib
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    # Prepare test slice
    handler = Alpha158(
        instruments=INSTRUMENTS,
        start_time=START_DATE,
        end_time=END_DATE,
        label=["Ref($close, -5) / Ref($close, 0) - 1"]  # match training label
    )
    dataset = DatasetH(handler, {"test": (START_DATE, END_DATE)})

    X_test = dataset.prepare("test", col_set="feature")
    y_test = dataset.prepare("test", col_set="label").squeeze("columns")

    mask = y_test.notna()
    X_test, y_test = X_test[mask], y_test[mask]
    X_test = X_test.dropna(axis=1, how="all")

    print(f"Test slice: {X_test.shape[0]} rows, {X_test.shape[1]} features")

    # Predictions + IC
    preds = model.predict(X_test, num_iteration=model.best_iteration)
    df_eval = pd.DataFrame({"pred": preds, "label": y_test}).dropna()

    agg_ic = df_eval["pred"].corr(df_eval["label"])
    ic_by_day = df_eval.groupby(level="datetime").apply(lambda g: g["pred"].corr(g["label"]))
    print(f"🔁 Aggregate IC: {agg_ic:.4f}, Per-day IC mean: {ic_by_day.mean():.4f}")

    # Backtest
    rets = run_backtest(df_eval)
    perf = summarize_performance(rets)
    print(
        f"📈 Backtest (rebalance={REBALANCE}, long_short={LONG_SHORT}, "
        f"cost={TURNOVER_COST_BPS}bps, cap={MAX_NAME_WEIGHT:.0%}): "
        f"CAGR={perf['CAGR']:.2%}, Sharpe={perf['Sharpe']:.2f}, "
        f"MaxDD={perf['MaxDD']:.2%}, MeanDaily={perf['MeanDaily']:.4%}, Days={perf['Days']}"
    )

    # Plots
    plot_results(rets, ic_by_day)

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()