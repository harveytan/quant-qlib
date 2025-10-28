from qlib.workflow import R
import pandas as pd
import matplotlib.pyplot as plt
import qlib
from qlib.config import REG_CN, REG_US

# Use your custom US dataset path
qlib.init(provider_uri="C:/Users/harve/.qlib/qlib_data/us_data", region=REG_US)

# Replace with your actual experiment and recorder IDs
recorder = R.get_recorder(experiment_id="731969603871975595", recorder_id="df1da9c6dfdd4bdb882f1c201b5102b5")

# Load predictions
preds = recorder.load_object("pred.pkl")
print("Sample predictions:\n", preds.head())

# Load feature importance
importance = recorder.load_object("feature_importance")
importance_df = pd.DataFrame(importance.items(), columns=["feature", "importance"]).sort_values(by="importance", ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(importance_df["feature"], importance_df["importance"])
plt.gca().invert_yaxis()
plt.title("Feature Importance - LightGBM")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.show()