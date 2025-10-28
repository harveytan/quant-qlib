from qlib.data.dataset import DatasetH
from qlib.data import D
from qlib.config import REG_US
import qlib
import json

qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region=REG_US)
#D.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data")

print(D.features(["META"], ["$close"], start_time="2025-09-18", end_time="2025-09-22"))


# from qlib.workflow import R

# recorder_dict = R.list_recorders()

# recorder_objs = list(recorder_dict.values())

# # Filter for FINISHED recorders
# finished = [r for r in recorder_objs if getattr(r, "info", {}).get("status") == "FINISHED"]

# # Sort by end_time descending
# finished_sorted = sorted(finished, key=lambda r: getattr(r, "end_time", ""), reverse=True)

# # Pick the latest
# latest_finished = finished_sorted[0] if finished_sorted else None

# if latest_finished:
#     print(f"✅ Latest finished recorder: {latest_finished.id}")
#     recorder = R.get_recorder(recorder_id=latest_finished.id)
#     # print(recorder.list_artifacts())
#     # params = recorder.load_object("params")
#     # feature_names = params.get("feature_names", [])
# else:
#     print("⚠️ No finished recorders found.")


with open("manual_artifacts/feature_importance.json") as f:
     importance = json.load(f)

print("Raw keys:", list(importance.keys()))

from qlib.data.dataset.handler import DataHandlerLP

handler = DataHandlerLP(config="workflow_config_lightgbm_us.yaml")
feature_names = handler.get_feature_names()
print(len(feature_names))  # Should be 158