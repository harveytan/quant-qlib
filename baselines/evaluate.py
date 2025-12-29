from .naive import naive_predict
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def evaluate_model(model, X, y, df, name="model", label_col="ensemble_label"):
    if model == "naive":
        # Compute naive predictions per ticker
        df = df.copy()
        df["naive_pred"] = df.groupby("symbol")[label_col].shift(1)

        # Extract true and predicted values
        y_true = df[label_col].values
        y_pred = df["naive_pred"].values

    else:
        # Normal model prediction path
        y_pred = model.predict(X)
        y_true = y

    # Convert to numpy arrays
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Build a mask for valid values (no NaN, no Inf)
    mask = (
        np.isfinite(y_true) &
        np.isfinite(y_pred)
    )

    # Apply mask
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    # Safety check
    if len(y_true) == 0:
        print(f"\n{name}: No valid samples after filtering NaN/Inf.")
        return {"mse": None, "mae": None, "r2": None}

    metrics = {
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }

    print(f"\n{name} performance:")
    for k, v in metrics.items():
        print(f"{k:10s}: {v:.6f}")

    return metrics