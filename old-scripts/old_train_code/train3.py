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
import pickle



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

    with R.start(experiment_name="dump_bin_lightgbm") as recorder:
        model.fit(dataset)  # model + params saved automatically

        with open("trained_model.pkl", "wb") as f:
            pickle.dump(model, f)

        recorder.log_artifact("trained_model.pkl")
        preds = model.predict(dataset, segment="test")

        # compute IC
        test_label = dataset.prepare("test", col_set="label")
        if isinstance(test_label, pd.DataFrame):
            test_label = test_label.iloc[:, 0]
        preds, test_label = preds.align(test_label, join="inner")
        ic = preds.corr(test_label)

        print(f"Test IC: {ic:.4f}")
        print("Sample predictions:")
        print(preds.head())

        # log metrics & artifacts
        recorder.log_metrics({"test_ic": ic})
        preds.to_csv("test_preds.csv")
        # recorder.log_artifact("test_preds.csv")
        print("Recorder ID:", recorder.id)



if __name__ == "__main__":
    multiprocessing.freeze_support()  # Needed only if you plan to freeze into an exe
    main()
