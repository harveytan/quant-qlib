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
from lightgbm import early_stopping
from sklearn.model_selection import TimeSeriesSplit
from pipeline.utils import prints, init_log_file, add_cross_sectional_features, g_safe_features, load_merge_and_save_calibration
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================
START_DATE = "2018-01-01"
END_DATE = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
MODEL_PATH = "trained_model_2.pkl"

# FIXED SAFE FEATURES (only change you requested)
SAFE_FEATURES = g_safe_features()

LABEL = "$ensemble_label"

init_log_file("logs/train_ensemble.log")
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

    prints('=== Starting Ensemble Training Pipeline ===')
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

    # -----------------------------
    # Date-based split (same as baselines)
    # -----------------------------
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (START_DATE, END_DATE),  # We'll split manually inside Optuna
        }
    )

    # ============================================================
    # Load prepared dataset
    # ============================================================
    df = dataset.prepare("train")
    X = df.xs("feature", axis=1, level=0)
    y = df.xs("label", axis=1, level=0).iloc[:, 0]

    X = add_cross_sectional_features(X)

    X = X.sort_index()
    y = y.sort_index()

    # 1. Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    # 2. Drop rows with NaN labels
    valid = y.notna()
    X = X.loc[valid]
    y = y.loc[valid]

    # 3. Fill NaNs in features only
    X = X.fillna(0)

    dates = X.index.get_level_values("datetime").unique().sort_values()
    date_index = X.index.get_level_values("datetime")

    tscv = TimeSeriesSplit(n_splits=5)
    date_folds = []

    for train_date_idx, val_date_idx in tscv.split(dates):
        train_dates = dates[train_date_idx]
        val_dates = dates[val_date_idx]
        train_mask = date_index.isin(train_dates)
        val_mask = date_index.isin(val_dates)
        date_folds.append((train_mask, val_mask))

    prints(f"Training rows after cleaning: {len(X)}")

    # Save a sample of training features for drift monitoring
    train_sample = X.sample(n=50000, random_state=42)  # or use full X if small
    train_sample.to_parquet("artifacts/train_features_sample.parquet")
    prints("Saved training feature sample for drift monitoring.")

    # ============================================================
    # OPTUNA OBJECTIVE — uses internal validation split
    # ============================================================
    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": 42,

            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.03, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "max_depth": -1,

            "min_child_samples": trial.suggest_int("min_child_samples", 10, 150),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-4, 20.0, log=True),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 1e-8, 5.0, log=True),

            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),

            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 100.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 100.0, log=True),

            "max_bin": trial.suggest_int("max_bin", 128, 1024),
            "n_estimators": trial.suggest_int("n_estimators", 500, 8000),
        }
        fold_mse = []
        best_iterations = []

        for train_mask, val_mask in date_folds:

            X_train = X.loc[train_mask]
            y_train = y.loc[train_mask]

            X_val = X.loc[val_mask]
            y_val = y.loc[val_mask]

            model = lgb.LGBMRegressor(**params)

            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="mse",
                callbacks=[
                    early_stopping(100, verbose=False)
                ],
            )

            preds = model.predict(X_val, num_iteration=model.best_iteration_)

            mse = mean_squared_error(y_val, preds)
            fold_mse.append(mse)

            best_iterations.append(model.best_iteration_)

        # Store best iteration for final training
        trial.set_user_attr(
            "best_iteration",
            int(np.mean(best_iterations))
        )
        return float(np.mean(fold_mse))

    # ============================================================
    # RUN OPTUNA
    # ============================================================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

    prints(f"Best MSE: {study.best_value:.6f}")
    prints(f"Best params: {study.best_trial.params}")

    # ============================================================
    # TRAIN FINAL MODEL
    # ============================================================
    best_params = study.best_trial.params
    if "best_iteration" in study.best_trial.user_attrs:
        best_params["n_estimators"] = study.best_trial.user_attrs["best_iteration"]

    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "columns": X.columns.tolist()}, f)

    prints(f"📦 Tuned model saved to {MODEL_PATH}")

    # ============================================================
    # FEATURE IMPORTANCE
    # ============================================================
    importances = model.feature_importances_
    features = model.feature_name_
    for name, score in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        prints(f"Feature: {name:<20} Importance: {score}")

    # ============================================================
    # VALIDATION EVALUATION (OUT-OF-SAMPLE)
    # ============================================================
    split_idx = int(len(X) * 0.85)
    X_train_final, X_valid = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train_final, y_valid = y.iloc[:split_idx], y.iloc[split_idx:]

    preds_valid = model.predict(X_valid)

    # ============================================================
    # load, merge enriched data with scores and save
    # ============================================================
    load_merge_and_save_calibration(model, X)


    mse_valid = mean_squared_error(y_valid, preds_valid)
    ic = spearmanr(preds_valid, y_valid.values).correlation

    prints(f"Validation MSE: {mse_valid}")
    prints(f"Validation IC: {ic}")

    # ============================================================
    # BUILD df_eval FOR STABILITY MODULES
    # ============================================================
    # Your X_valid index is MultiIndex: (date, instrument)
    df_eval = pd.DataFrame({
        "date": X_valid.index.get_level_values(0),
        "symbol": X_valid.index.get_level_values(1),
        "pred": preds_valid,
        "label": y_valid.values,
    })

    # ============================================================
    # RUN STABILITY CHECKS
    # ============================================================
    from pathlib import Path
    from stability.rolling_ic import compute_daily_ic
    from stability import (
        run_rolling_ic_monitor_training,
        run_model_stability_check,
    )

    STABILITY_DIR = Path("stability_outputs")
    STABILITY_DIR.mkdir(exist_ok=True, parents=True)

    # ---------------------------
    # 1) Rolling IC Stability
    # ---------------------------
    ic_series = compute_daily_ic(df_eval)
    rolling_summary = run_rolling_ic_monitor_training(
        ic_series,
        out_dir=STABILITY_DIR / "rolling_ic",
    )
    prints(f"📈 Rolling IC summary: {rolling_summary}")

    # ---------------------------
    # 2) Model Stability vs Previous Model
    # ---------------------------
    prev_dir = Path("models") / "last_model"
    curr_dir = Path("models") / "current_model"

    # Save current model artifacts for comparison
    Path(curr_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": features, "importance": importances}).to_csv(
        Path(curr_dir) / "feature_importance.csv", index=False
    )
    df_eval.to_csv(Path(curr_dir) / "predictions.csv", index=False)

    if prev_dir.exists():
        old_importance = pd.read_csv(prev_dir / "feature_importance.csv").set_index("feature")["importance"]
        new_importance = pd.read_csv(curr_dir / "feature_importance.csv").set_index("feature")["importance"]

        old_pred = (pd.read_csv(prev_dir / "predictions.csv").set_index(["date", "symbol"])["pred"])
        new_pred = (df_eval.set_index(["date", "symbol"])["pred"])

        # ---------------------------
        # ALIGN PREDICTIONS HERE
        # ---------------------------
        aligned = (pd.DataFrame({"old": old_pred}).join(pd.DataFrame({"new": new_pred}), how="inner").dropna())

        # Extract aligned vectors
        old_pred_aligned = aligned["old"]
        new_pred_aligned = aligned["new"]

        # Now run stability check with aligned predictions
        stability_summary = run_model_stability_check(
            old_importance=old_importance,
            new_importance=new_importance,
            old_pred=old_pred_aligned,
            new_pred=new_pred_aligned,
            out_dir=STABILITY_DIR / "model_stability",
        )
        prints(f"🧱 Model stability summary: {stability_summary}")
    else:
        prints("ℹ️ No previous model found — skipping model stability check.")

    # After finishing, move current_model → last_model for next run
    import shutil
    if Path(curr_dir).exists():
        shutil.rmtree(prev_dir, ignore_errors=True)
        shutil.copytree(curr_dir, prev_dir)

if __name__ == "__main__":
    main()