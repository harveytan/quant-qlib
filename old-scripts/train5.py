import qlib
from qlib.workflow import R
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation
import pandas as pd
import pickle

# ----------------------------
# Load dataset with Alpha158
# ----------------------------
def load_dataset():
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    # Use all instruments for cross-sectional learning
    handler = Alpha158(
        instruments="all",
        start_time="2019-01-01",
        end_time="2026-12-31",
        # 5-day forward return label
        label=["Ref($close, -5) / Ref($close, 0) - 1"]
    )

    segments = {
        "train": ("2019-06-01", "2023-12-31"),  # skip early NaNs
        "valid": ("2024-01-01", "2024-12-31"),
        "test":  ("2025-01-01", "2025-09-30"),
    }

    dataset = DatasetH(handler, segments)
    return dataset

# ----------------------------
# Prepare split (features + labels)
# ----------------------------
def prepare_split(dataset, segment):
    X = dataset.prepare(segment, col_set="feature")
    y = dataset.prepare(segment, col_set="label").squeeze("columns")

    # Drop NaN labels only
    mask = y.notna()
    X, y = X[mask], y[mask]

    # Drop all-NaN and near-constant features
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.std() > 1e-6]

    print(f"{segment} after cleaning: {X.shape[0]} rows, {X.shape[1]} features")
    return X, y

# ----------------------------
# Main training loop
# ----------------------------
def main():
    dataset = load_dataset()

    X_train, y_train = prepare_split(dataset, "train")
    X_valid, y_valid = prepare_split(dataset, "valid")
    X_test, y_test   = prepare_split(dataset, "test")

    print("Train shape:", X_train.shape, "Valid shape:", X_valid.shape, "Test shape:", X_test.shape)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    # Stronger parameters for cross-sectional alpha
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "max_depth": -1,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-3,
        "reg_lambda": 1e-3,
    }

    with R.start(experiment_name="dump_bin_lightgbm"):
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,  # allow longer training
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[early_stopping(stopping_rounds=100), log_evaluation(50)]
        )

        # Save model
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # ----------------------------
        # Validation diagnostics
        # ----------------------------
        preds_valid = model.predict(X_valid, num_iteration=model.best_iteration)
        unique_per_day = (
            pd.Series(preds_valid, index=X_valid.index)
            .groupby(level="datetime")
            .nunique()
        )
        print("Unique preds per day (validation):")
        print(unique_per_day.head())

        # ----------------------------
        # Test evaluation
        # ----------------------------
        preds_test = model.predict(X_test, num_iteration=model.best_iteration)
        df_eval = pd.concat(
            [pd.Series(preds_test, index=y_test.index, name="pred"), y_test.rename("label")],
            axis=1
        ).dropna()

        if not df_eval.empty and df_eval["pred"].std() > 0:
            ic = df_eval["pred"].corr(df_eval["label"])
        else:
            ic = float("nan")

        R.log_metrics(test_ic=float(ic))
        print("Logged test IC:", ic)

if __name__ == "__main__":
    main()