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
from utils import prints
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr


# ============================================================
# CONFIG
# ============================================================
START_DATE = "2018-01-01"
END_DATE = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
MODEL_PATH = "trained_model_2.pkl"

# FIXED SAFE FEATURES (only change you requested)
SAFE_FEATURES = [
    "$open", "$high", "$low", "$close",
    "$volume",
    "$vol_5d", "$vol_10d", "$vol_20d",
    "$rank_vol_5d", "$rank_vol_10d", "$rank_vol_20d",
    "$days_since_ipo",
]

LABEL = "$ensemble_label"


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

    # -----------------------------
    # Date-based split (same as baselines)
    # -----------------------------
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (START_DATE, END_DATE),  # We'll split manually inside Optuna
        }
    )

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

    # ============================================================
    # OPTUNA OBJECTIVE — uses internal validation split
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
        }

        # -----------------------------
        # Internal row-based split (unchanged)
        # -----------------------------
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mse = mean_squared_error(y_val, preds)
        return mse

    # ============================================================
    # RUN OPTUNA
    # ============================================================
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=55)

    prints(f"Best MSE: {study.best_value:.6f}")
    prints(study.best_trial.params)

    # ============================================================
    # TRAIN FINAL MODEL
    # ============================================================
    best_params = study.best_trial.params
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "columns": X.columns.tolist()}, f)

    prints(f"\n📦 Tuned model saved to {MODEL_PATH}")

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
    mse_valid = mean_squared_error(y_valid, preds_valid)
    ic = spearmanr(preds_valid, y_valid.values).correlation

    prints(f"\nValidation MSE: {mse_valid}")
    prints(f"Validation IC: {ic}")


if __name__ == "__main__":
    main()