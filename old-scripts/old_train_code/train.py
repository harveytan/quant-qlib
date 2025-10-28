import qlib
#import mlflow
from qlib.config import REG_US
from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
#from qlib.workflow import R
import os
import pickle
import json

def main():
    qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region=REG_US)

    handler = Alpha158(instruments="all", start_time="2016-01-01", end_time="2023-12-31")
    dataset = DatasetH(handler=handler, segments={
        "train": ("2017-01-01", "2023-12-31"),
        "valid": ("2022-01-01", "2022-12-31"),
        "test": ("2023-01-01", "2024-12-31")
    })

    model = LGBModel(loss="mse", num_leaves=128, learning_rate=0.02, n_estimators=500, max_depth=8)
    model.fit(dataset)
    preds = model.predict(dataset)

	# Define your fixed output directory
    output_dir = "manual_artifacts"
    os.makedirs(output_dir, exist_ok=True)

    # Save model
    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model.model, f)

    # Save predictions
    with open(os.path.join(output_dir, "pred.pkl"), "wb") as f:
        pickle.dump(preds, f)


    # importance_dict = dict(zip(model.model.feature_name(), model.model.feature_importance()))
    importance_dict = {
       str(k): float(v) if hasattr(v, "__float__") else int(v)
       for k, v in zip(model.model.feature_name(), model.model.feature_importance())
    }
    with open(os.path.join(output_dir, "feature_importance.json"), "w") as f:
        json.dump(importance_dict, f, indent=4)

    # # Save feature importance
    # with open(os.path.join(output_dir, "feature_importance.json"), "w") as f:
    #     json.dump(model.model.feature_importance(), f)

    # importance_dict = dict(zip(model.model.feature_name(), model.model.feature_importance()))
    # prediction_summary = {
    #    "num_predictions": len(preds),
    #    "top_prediction": max(preds.values()) if isinstance(preds, dict) else None,
    #    "bottom_prediction": min(preds.values()) if isinstance(preds, dict) else None
    # }

    # log_data = {
    #    "timestamp": "2025-09-23",
    #    "model_type": "LightGBM",
    #    "hyperparameters": model.model.get_params(),
    #    "feature_importance": importance_dict,
    #    "prediction_summary": prediction_summary
    # }

    # with open(os.path.join(output_dir, "log.json"), "w") as f:
    #    json.dump(log_data, f, indent=4)


if __name__ == "__main__":
    main()