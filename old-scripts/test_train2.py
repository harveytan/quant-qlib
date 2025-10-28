import multiprocessing
import qlib

def main():
    qlib.init(
        provider_uri="C:/Users/harve/.qlib/qlib_data/us_data",  # your custom bundle
        region="us"
    )

    from qlib.contrib.data.handler import Alpha158
    from qlib.data.dataset import DatasetH
    from qlib.contrib.model.gbdt import LGBModel

    handler = Alpha158(
        instruments="all",
        start_time="2016-01-01",
        end_time="2023-12-31"
    )

    dataset = DatasetH(handler, segments={
        "train": ("2017-01-01", "2023-12-31"),
        "valid": ("2022-01-01", "2022-12-31"),
        "test": ("2023-01-01", "2024-12-31")
    })

    model = LGBModel(loss="mse", num_leaves=128, learning_rate=0.02, n_estimators=500, max_depth=8)
    model.fit(dataset)
    preds = model.predict(dataset)
    print(preds)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # optional unless you're freezing to .exe
    main()