import qlib
import optuna
import pickle
import pandas as pd
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.contrib.model.gbdt import LGBModel
from sklearn.metrics import mean_squared_error
from qlib.data import D


# ----------------------------
# Config
# ----------------------------
START_DATE = "2018-01-02"
END_DATE = "2025-09-01"
INSTRUMENTS = "all"
MAX_TRIALS = 50
MODEL_PATH = "trained_model_2.pkl" # enriched features

# ----------------------------
# Objective Factory
# ----------------------------
def make_objective(dataset):
    def objective(trial):
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "learning_rate": 0.05,
            "verbose": -1
        }

        model = LGBModel(loss="mse", **params)

        print("fitting model")
        model.fit(dataset)

        print("predicting")
        preds = model.predict(dataset, "valid")
        y_valid = dataset.prepare("valid", col_set="label")

        # Choose one:
        # return mean_squared_error(y_valid, preds)  # for MSE
        # return preds.corr(y_valid)  # for IC
        return mean_squared_error(y_valid, preds)  # for MSE

    return objective

class CustomAlpha158(Alpha158):
    def get_loader(self):
        self.loader_config["kwargs"]["feature"] = [
            "$close / Ref($close, 1) - 1",
            "Std($close, 5)",
            "Std($close, 20)",
            "Std($close, 5) / Std($close, 20)",
            "($ask - $bid) / $mid",
            "Rank(($ask - $bid) / $mid)"
        ]
        print("loader:", self.loader_config)
        return super().get_loader()
# ----------------------------
# Main
# ----------------------------
def main():
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region="us")

    handler = CustomAlpha158(
        instruments="all",
        start_time=START_DATE,
        end_time="2025-09-01",
        fit_start_time=START_DATE,
        fit_end_time="2025-09-01",
        infer_processors=[
            {"class": "Fillna"},
            {"class": "ZScoreNorm", "kwargs": {
                "fit_start_time": START_DATE,
                "fit_end_time": "2025-09-01"
            }}
        ],
        label=["(Ref($close, -5) / Ref($close, 0) - 1) / Std($close, 5)"]
    )


    dataset = DatasetH(handler=handler, segments={
        "train": (START_DATE, "2023-12-31"),
        "valid": ("2024-01-01", "2025-09-01")
    })
    print("Dataset initialized.", dataset)

    # # Fetch raw features and labels
    # features = handler.fetch(col_set="feature").dropna(axis=1, how="all").dropna(axis=0, how="any")
    # labels = handler.fetch(col_set="label").loc[features.index]

    # # Get valid close index
    # instruments = D.instruments(market="all")
    # close = D.features(instruments, ["$close"], start_time=START_DATE, end_time=END_DATE)["$close"]
    # close = close.swaplevel().sort_index()
    # close = close.unstack().ffill().bfill().stack()
    # valid_index = close.index

    # # Intersect all three
    # common_index = features.index.intersection(labels.index).intersection(valid_index)
    # features = features.loc[common_index]
    # labels = labels.loc[common_index]

    # # Split into train/valid
    # train_dates = features.index.get_level_values(0).unique().sort_values()
    # split_date = train_dates[int(len(train_dates) * 0.8)]

    # segments = {
    #     "train": (START_DATE, split_date.strftime("%Y-%m-%d")),
    #     "valid": (split_date.strftime("%Y-%m-%d"), END_DATE)
    # }
    # print("Train segment:", segments["train"])
    # print("Valid segment:", segments["valid"])

    # # Patch handler with filtered data
    # handler._data = {"feature": features, "label": labels}
    # dataset = DatasetH(handler, segments)

    # Run Optuna
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(dataset), n_trials=MAX_TRIALS)

    print("\n✅ Best Trial:")
    print(f"  MSE: {study.best_value:.6f}")
    print(f"  Params: {study.best_params}")

    # Retrain final model
    final_model = LGBModel(loss="mse", **study.best_params)
    final_model.fit(dataset)
    booster = final_model.model
    feature_names = booster.feature_name()
    importance = booster.feature_importance()
    fi_series = pd.Series(importance, index=feature_names)
    print("\nFeature Importance:", fi_series.sort_values(ascending=False).head(7))

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(final_model, f)

    print(f"\n📦 Tuned model saved to {MODEL_PATH}")
    # BEST PARAMETER FOUND:
    # 1. MSE: 59.617
    # {
    # 'num_leaves': 110,
    # 'max_depth': 8,
    # 'min_data_in_leaf': 86,
    # 'feature_fraction': 0.8659,
    # 'bagging_fraction': 0.5846,
    # 'lambda_l1': 2.6198,
    # 'lambda_l2': 4.6805
    # }
    # 2. MSE: 2.845586 (FOUND a better one!)
    # Params: 
    # {'num_leaves': 46, 
    # 'max_depth': 6, 
    # 'min_data_in_leaf': 66, 
    # 'feature_fraction': 0.8564443776160324, 
    # 'bagging_fraction': 0.6728790578919412, 
    # 'lambda_l1': 3.536557382242193, 
    # 'lambda_l2': 1.1294727016279218
    # }
    # 3. MSE: 2.860894 (the current one)
    # Params: 
    # {'num_leaves': 18, 
    # 'max_depth': 3, 
    # 'min_data_in_leaf': 75, 
    # 'feature_fraction': 0.5073647945063973, 
    # 'bagging_fraction': 0.917932921564762, 
    # 'lambda_l1': 1.8744800160665798, 
    # 'lambda_l2': 4.227264606948985
    #}
if __name__ == "__main__":
    main()