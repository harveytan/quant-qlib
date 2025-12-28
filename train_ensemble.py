import qlib
import optuna
import pickle
import pandas as pd
from utils import prints
from qlib.data.dataset import DatasetH
from datetime import datetime, timedelta
from qlib.data import D
import numpy as np
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandler
import lightgbm as lgb


# ----------------------------
# Config
# ----------------------------
START_DATE = "2018-01-01"
END_DATE = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")    #END_DATE = "2030-10-18" Adjust this if it caused error

INSTRUMENTS = "all"
MAX_TRIALS = 50
MODEL_PATH = "trained_model_2.pkl"


def main():
    class LoaderWrapper(DataHandler):
        def __init__(self, loader):
            # Defensive extraction
            feature_df = loader._config.get("feature")
            label_df = loader._config.get("label")

            if not isinstance(feature_df, pd.DataFrame) or not isinstance(label_df, pd.DataFrame):
                raise TypeError("Expected DataFrames for 'feature' and 'label'")

            self.data_loader = loader
            self._data = pd.concat({"feature": feature_df, "label": label_df}, axis=1)

            # Required attributes for DatasetH
            prints(feature_df.index.names)
            prints(feature_df.head())
            self.instruments = sorted(set(feature_df.index.get_level_values("instrument")))
            self.start_time = str(feature_df.index.get_level_values("datetime").min().date())
            self.end_time = str(feature_df.index.get_level_values("datetime").max().date())
            self.fetch_orig = True

        def fetch(self, instruments=None, start_time=None, end_time=None, freq="day", col_set="__all", data_key=None):
            if col_set == "__all":
                return self._data
            return self._data.xs(col_set, axis=1, level=0)


    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    instrument_path = r"C:\Users\harve\.qlib\qlib_data\us_data\instruments\all.txt"

    with open(instrument_path, "r") as f:
        instrumentx = [line.strip().split("\t")[0] for line in f if line.strip()]

    fields = ["$open", "$high", "$low", "$close", "$volume",
        "$vol_5d", "$rank_vol_5d",
        # "$ret_5d", "$rank_ret_5d",
        # "$ret_10d", "$vol_10d", "$rank_ret_10d", "$rank_vol_10d",
        # "$ret_20d", "$vol_20d", "$rank_ret_20d", "$rank_vol_20d",
        ]
    features = D.features(
        instruments=instrumentx,
        fields=fields,
        start_time=START_DATE,
        end_time=END_DATE
    )

    labels = D.features(
        instruments=instrumentx,
        fields=["$ensemble_label"],
        start_time=START_DATE,
        end_time=END_DATE
    )
    from qlib.data.dataset.loader import StaticDataLoader

    loader = StaticDataLoader(config={
        "feature": features,
        "label": labels
    })


    handler = LoaderWrapper(loader)

    END_TRAIN_DATE = (datetime.today() - timedelta(days=95)).strftime("%Y-%m-%d")
    START_VALID_DATE = (datetime.today() - timedelta(days=90)).strftime("%Y-%m-%d")
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": (START_DATE, END_TRAIN_DATE),
            "valid": (START_VALID_DATE, END_DATE)
        }
    )

    df = dataset.prepare("train")

    X = df.xs("feature", axis=1, level=0)
    y = df.xs("label", axis=1, level=0)

    X["$volume_log"] = np.log1p(X["$volume"])
    X.drop(columns=["$volume"], inplace=True)

    y_flat = y.squeeze()
    y_flat.index = X.index
    y_flat = y_flat.loc[X.index]

    # Drop rows with NaN labels
    mask = ~y_flat.isna()
    X = X.loc[mask]
    y_flat = y_flat.loc[mask]

    # Optionally also drop rows with NaN in critical features
    X = X.dropna(subset=["$vol_5d", "$rank_vol_5d"])
    y_flat = y_flat.loc[X.index]

    prints(f"Training rows after cleaning: {len(X)}")
    prints(f"Remaining NaN labels: {y_flat.isna().sum()}")
    prints(f"Remaining NaN features: {X.isna().sum().sum()}")

    # 📊 Correlation diagnostics
    for col in ["$ret_5d", "$ret_10d", "$ret_20d"]:
        if col in X.columns:
            corr = X[col].corr(y_flat)
            prints(f"Correlation with ensemble_label: {col:<10} → {corr:.4f}")

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "mse",
            "num_leaves": trial.suggest_int("num_leaves", 32, 256),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 24),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = lgb.LGBMRegressor(**params)
        model.fit(X, y_flat)

        preds = model.predict(X)
        mse = np.mean((preds - y_flat.values.flatten())**2)
        return mse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=55)

    prints(f"Best value: {study.best_trial.value}")
    prints(f"  MSE: {study.best_value:.6f}")
    prints(study.best_trial.params)

    best_params = study.best_trial.params
    model = lgb.LGBMRegressor(**best_params)
    model.fit(X, y_flat)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "columns": X.columns.tolist()}, f)

    prints(f"\n📦 Tuned model saved to {MODEL_PATH}")

    importances = model.feature_importances_
    features = model.feature_name_
    for name, score in sorted(zip(features, importances), key=lambda x: x[1], reverse=True):
        prints(f"Feature: {name:<20} Importance: {score}")

    # Step 1: Prepare validation features
    X_valid = dataset.prepare("valid", col_set="feature")
    y_valid = dataset.prepare("valid", col_set="label")

    # Step 2: Apply same feature engineering
    X_valid["$volume_log"] = np.log1p(X_valid["$volume"])
    X_valid.drop(columns=["$volume"], inplace=True)

    # Step 3: Drop rows with NaN labels
    y_valid_flat = y_valid.squeeze()
    mask = ~y_valid_flat.isna()
    X_valid = X_valid.loc[mask]
    y_valid_flat = y_valid_flat.loc[mask]

    # Step 4: Predict
    preds_valid = model.predict(X_valid)

    # Step 5: Evaluate MSE
    from sklearn.metrics import mean_squared_error
    mse_valid = mean_squared_error(y_valid_flat, preds_valid)
    prints(f"Validation MSE: {mse_valid}")

    # Step 6: Optional — log IC
    from scipy.stats import spearmanr
    ic = spearmanr(preds_valid, y_valid_flat.values).correlation
    prints(f"Validation IC: {ic}")

if __name__ == "__main__":
    main()