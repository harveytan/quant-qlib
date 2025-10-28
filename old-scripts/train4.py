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
    handler = Alpha158(instruments="all", start_time="2019-01-01", end_time="2026-12-31")

    segments = {
        "train": ("2019-06-01", "2023-12-31"),  # start later to skip lookback NaNs
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

    # Drop all-NaN columns (e.g. VWAP0)
    X = X.dropna(axis=1, how="all")

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

    # Looser parameters for small/medium data
    params = {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": -1,
        "min_child_samples": 5,
        "reg_alpha": 0.0,
        "reg_lambda": 0.0,
    }

    with R.start(experiment_name="dump_bin_lightgbm"):
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=[early_stopping(stopping_rounds=50), log_evaluation(20)]
        )

        # Save model
        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Evaluate on test set
        preds = model.predict(X_test, num_iteration=model.best_iteration)
        df_eval = pd.concat([pd.Series(preds, index=y_test.index), y_test], axis=1).dropna()

        if not df_eval.empty and df_eval.iloc[:,0].std() > 0:
            ic = df_eval.iloc[:,0].corr(df_eval.iloc[:,1])
        else:
            ic = float("nan")

        R.log_metrics(test_ic=float(ic))
        print("Logged test IC:", ic)

if __name__ == "__main__":
    main()