import qlib
import optuna
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from qlib.data import D
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler
import lightgbm as lgb
from pipeline.utils import prints, init_log_file
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from pathlib import Path
import shutil
from stability.rolling_ic import compute_daily_ic
from stability import run_rolling_ic_monitor_training, run_model_stability_check

# ============================================================
# CONFIG
# ============================================================

# Rolling window: last 4 years
TODAY = datetime.today()
START_DATE = (TODAY - timedelta(days=365 * 4)).strftime("%Y-%m-%d")

# 15-day buffer to avoid label leakage
END_DATE = (TODAY - timedelta(days=15)).strftime("%Y-%m-%d")

MODEL_PATH = "trained_model_15_20260122.pkl"

# Toggle: normalize features per instrument
NORMALIZE = True

SAFE_FEATURES = [
    "$open", "$high", "$low", "$close",
    "$volume",
    "$vol_5d", "$vol_10d", "$vol_20d",
    "$rank_vol_5d", "$rank_vol_10d", "$rank_vol_20d",
    "$days_since_ipo",
]

LABEL = "$ensemble_label"

init_log_file("logs/train_ensemble_15_20260122.log")

np.random.seed(42)

# ============================================================
# WRAPPER FOR STATIC DATA
# ============================================================
class LoaderWrapper(DataHandler):
    def __init__(self, loader):
        feature_df = loader._config.get("feature")
        label_df = loader._config.get("label")

        if not isinstance(feature_df, pd.DataFrame) or not isinstance(label_df, pd.DataFrame):
            raise TypeError("Expected DataFrames for 'feature' and 'label'")

        self.data_loader = loader
        self._data = pd.concat({"feature": feature_df, "label": label_df}, axis=1)

        self.instruments = sorted(set(feature_df.index.get_level_values("instrument")))
        self.start_time = str(feature_df.index.get_level_values("datetime").min().date())
        self.end_time = str(feature_df.index.get_level_values("datetime").max().date())
        self.fetch_orig = True

    def fetch(self, instruments=None, start_time=None, end_time=None,
              freq="day", col_set="__all", data_key=None):

        if col_set in ["__all", None]:
            return self._data

        if isinstance(col_set, (list, tuple)):
            return self._data.loc[:, col_set]

        if col_set in self._data.columns.levels[0]:
            return self._data.xs(col_set, axis=1, level=0)

        return self._data


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================
def main():

    # -----------------------------
    # Initialize Qlib
    # -----------------------------
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    # Load instruments
    instrument_path = r"C:\Users\harve\.qlib\qlib_data\us_data\instruments\all.txt"
    with open(instrument_path, "r") as f:
        instruments = [line.strip().split("\t")[0] for line in f if line.strip()]

    # -----------------------------
    # Load features + labels
    # -----------------------------
    features = D.features(
        instruments=instruments,
        fields=SAFE_FEATURES,
        start_time=START_DATE,
        end_time=END_DATE
    )

    labels = D.features(
        instruments=instruments,
        fields=[LABEL],
        start_time=START_DATE,
        end_time=END_DATE
    )

    from qlib.data.dataset.loader import StaticDataLoader
    loader = StaticDataLoader(config={"feature": features, "label": labels})
    handler = LoaderWrapper(loader)

    dataset = DatasetH(handler=handler, segments={"train": (START_DATE, END_DATE)})
    df = dataset.prepare("train")

    X = df.xs("feature", axis=1, level=0)
    y = df.xs("label", axis=1, level=0).squeeze()

    # -----------------------------
    # Feature engineering
    # -----------------------------
    X["$volume_log"] = np.log1p(X["$volume"])
    X.drop(columns=["$volume"], inplace=True)

    # -----------------------------
    # Clean NaN/Inf
    # -----------------------------
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    prints(f"Training rows after cleaning: {len(X)}")

    # -----------------------------
    # Optional per-instrument normalization
    # -----------------------------
    if NORMALIZE:
        prints("Applying per-instrument normalization...")
        X = X.groupby("instrument").transform(lambda df: (df - df.mean()) / df.std())

    # -----------------------------
    # Save drift sample
    # -----------------------------
    X.sample(n=min(50000, len(X)), random_state=42).to_parquet(
        "artifacts/train_features_sample.parquet"
    )
    prints("Saved training feature sample for drift monitoring.")

    # ============================================================
    # Chronological split helper
    # ============================================================
    dates = pd.Series(X.index.get_level_values("datetime"), index=X.index)

    cutoff_80 = dates.quantile(0.80)
    cutoff_85 = dates.quantile(0.85)

    def chronological_split(X, y, cutoff):
        mask_train = dates <= cutoff
        mask_val = dates > cutoff
        return X[mask_train], X[mask_val], y[mask_train], y[mask_val]

    # ============================================================
    # OPTUNA OBJECTIVE
    # ============================================================
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": trial.suggest_int("num_leaves", 32, 256),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1200),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "random_state": 42,
        }

        X_train, X_val, y_train, y_val = chronological_split(X, y, cutoff_80)

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=55)]
        )
        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return mse

    # ============================================================
    # RUN OPTUNA
    # ============================================================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=55)

    prints(f"Best MSE: {study.best_value:.6f}")
    prints(f"Best params: {study.best_trial.params}")

    # ============================================================
    # TRAIN FINAL MODEL
    # ============================================================
    best_params = study.best_trial.params
    best_params["random_state"] = 42

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "columns": X.columns.tolist()}, f)

    prints(f"\n📦 Tuned model saved to {MODEL_PATH}")

    # ============================================================
    # VALIDATION EVALUATION (OUT-OF-SAMPLE)
    # ============================================================
    X_train_final, X_valid, y_train_final, y_valid = chronological_split(X, y, cutoff_85)

    preds_valid = model.predict(X_valid)
    mse_valid = mean_squared_error(y_valid, preds_valid)
    ic = spearmanr(preds_valid, y_valid.values).correlation

    prints(f"\nValidation MSE: {mse_valid}")
    prints(f"Validation IC: {ic}")

    df_eval = pd.DataFrame({
        "date": X_valid.index.get_level_values(0),
        "symbol": X_valid.index.get_level_values(1),
        "pred": preds_valid,
        "label": y_valid.values,
    })

    # ============================================================
    # Stability checks
    # ============================================================


    STABILITY_DIR = Path("stability_outputs")
    STABILITY_DIR.mkdir(exist_ok=True, parents=True)

    ic_series = compute_daily_ic(df_eval)

    rolling_summary = run_rolling_ic_monitor_training(
        ic_series,
        out_dir=STABILITY_DIR / "rolling_ic",
    )
    prints(f"\n📈 Rolling IC summary: {rolling_summary}")

    # Save artifacts
    prev_dir = Path("models") / "last_model"
    curr_dir = Path("models") / "current_model"
    curr_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).to_csv(curr_dir / "feature_importance.csv", index=False)

    df_eval.to_csv(curr_dir / "predictions.csv", index=False)

    # Stability comparison
    if prev_dir.exists():
        old_importance = pd.read_csv(prev_dir / "feature_importance.csv").set_index("feature")["importance"]
        new_importance = pd.read_csv(curr_dir / "feature_importance.csv").set_index("feature")["importance"]

        old_pred = pd.read_csv(prev_dir / "predictions.csv").set_index(["date", "symbol"])["pred"]
        new_pred = df_eval.set_index(["date", "symbol"])["pred"]

        stability_summary = run_model_stability_check(
            old_importance=old_importance,
            new_importance=new_importance,
            old_pred=old_pred,
            new_pred=new_pred,
            out_dir=STABILITY_DIR / "model_stability",
        )
        prints(f"\n🧱 Model stability summary: {stability_summary}")
    else:
        prints("\nℹ️ No previous model found — skipping model stability check.")

    # Move current_model → last_model
    if curr_dir.exists():
        shutil.rmtree(prev_dir, ignore_errors=True)
        shutil.copytree(curr_dir, prev_dir)


if __name__ == "__main__":
    main()