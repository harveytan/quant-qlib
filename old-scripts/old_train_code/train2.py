# 1. Initialize Qlib with your dumped bundle
import qlib
from qlib.config import C


# 2. Build a Dataset from Alpha158 (or Alpha360 if you prefer)
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from qlib.contrib.model.gbdt import LGBModel
from qlib.workflow import R
import pandas as pd
import multiprocessing



def main():
    # Point to your dump_bin output directory
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")
    # Replace with your instrument universe (e.g., S&P500, NASDAQ100, or custom list)
    handler = Alpha158(instruments="all", start_time="2019-01-01", end_time="2026-12-31")

    # 3. Split into train/valid/test
    segments = {
        "train":("2019-01-01", "2023-12-31"),
        "valid":("2022-01-01", "2024-12-31"),
        "test":("2024-01-01", "2025-12-31"),
    }

    dataset = DatasetH(handler, segments)

    # is this the same as:

    model = LGBModel(
        loss="mse",
        colsample_bytree=0.8879,
        learning_rate=0.05,
        subsample=0.8789,
        num_leaves=200,
        reg_alpha=130.0,
        reg_lambda=80.0,
        max_depth=8,
        n_estimators=500,
        min_child_weight=0.1,
    )

    model.fit(dataset)

    # 5. Evaluate on test set
    # Predictions: Series with MultiIndex
    preds = model.predict(dataset, segment="test")

    # Labels: DataFrame -> convert to Series
    test_label = dataset.prepare("test", col_set="label")

    # If it's a DataFrame with one column, squeeze it down
    if isinstance(test_label, pd.DataFrame):
        test_label = test_label.iloc[:, 0]

    # Align indices
    preds, test_label = preds.align(test_label, join="inner")

    # Now correlation works
    ic = preds.corr(test_label)
    print(f"Test IC: {ic:.4f}")
    print("Sample predictions:")
    print(preds.head())

    rec = R.get_recorder()  # grab the current run
    if rec is not None:
        rec.save_objects(trained_model=model)  # explicitly save the model
        rec.log_params(model.params)
        rec.log_metrics({"test_ic": ic})
        preds.to_csv("test_preds.csv")
        rec.log_artifact("test_preds.csv")
        
        print(f"Test IC logged: {ic:.4f}")
    else:
        print("No active recorder found — skipping logging.")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed only if you plan to freeze into an exe
    main()
