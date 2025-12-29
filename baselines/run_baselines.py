import os
import glob
import pandas as pd
import numpy as np
from baselines.linear import train_linear_baseline
from baselines.lightgbm_small import train_small_lgbm
from baselines.evaluate import evaluate_model
from baselines.naive import naive_predict


DATA_DIR = r"C:\Users\harve\.qlib\stock_data\normalize\us_data_enriched"

def load_data_from_csv():
    """
    Loads all enriched ticker CSVs into a single DataFrame.
    Assumes each CSV has columns including:
        date, symbol, <features>, label
    """

    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    dfs = []
    for f in all_files:
        df = pd.read_csv(f)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)

    # Ensure proper sorting
    df = df.sort_values(["date", "symbol"]).reset_index(drop=True)

    return df

def prepare_splits(df):
    """
    Convert the full DataFrame into X_train, y_train, X_val, y_val, X_test, y_test.
    Uses 'ensemble_label' as the target.
    """
    label_col = "ensemble_label"

    feature_cols = [
        c for c in df.columns
        if c not in ["date", "symbol", label_col]
    ]

    dates = sorted(df["date"].unique())
    n = len(dates)

    train_cut = int(n * 0.7)
    val_cut = int(n * 0.85)

    train_dates = dates[:train_cut]
    val_dates = dates[train_cut:val_cut]
    test_dates = dates[val_cut:]

    train_df = df[df["date"].isin(train_dates)].copy()
    val_df   = df[df["date"].isin(val_dates)].copy()
    test_df  = df[df["date"].isin(test_dates)].copy()

    # Clean NaN/Inf in label
    for d in (train_df, val_df, test_df):
        d.replace([np.inf, -np.inf], np.nan, inplace=True)
        d.dropna(inplace=True)

    X_train = train_df[feature_cols].values
    y_train = train_df[label_col].values

    X_val = val_df[feature_cols].values
    y_val = val_df[label_col].values

    X_test = test_df[feature_cols].values
    y_test = test_df[label_col].values

    return X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df


def load_data():
    df = load_data_from_csv()
    return prepare_splits(df)

def run_baselines():
    print("Loading enriched dataset...")
    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = load_data()

    results = {}

    # -------------------------
    # Naive baseline
    # -------------------------
    results["naive_train"] = evaluate_model("naive", None, None, train_df, "Naive Train")
    results["naive_val"]   = evaluate_model("naive", None, None, val_df,   "Naive Val")
    results["naive_test"]  = evaluate_model("naive", None, None, test_df,  "Naive Test")    

    # -------------------------
    # Linear baseline
    # -------------------------
    print("\nTraining linear baseline...")
    linear = train_linear_baseline(X_train, y_train)

    results["linear_train"] = evaluate_model(linear, X_train, y_train, "Linear Train")
    results["linear_val"]   = evaluate_model(linear, X_val,   y_val,   "Linear Val")
    results["linear_test"]  = evaluate_model(linear, X_test,  y_test,  "Linear Test")

    # -------------------------
    # LightGBM baseline
    # -------------------------
    print("\nTraining small LightGBM baseline...")
    lgbm = train_small_lgbm(X_train, y_train)

    results["lgbm_train"] = evaluate_model(lgbm, X_train, y_train, "LGBM Train")
    results["lgbm_val"]   = evaluate_model(lgbm, X_val,   y_val,   "LGBM Val")
    results["lgbm_test"]  = evaluate_model(lgbm, X_test,  y_test,  "LGBM Test")

    print("\nBaseline suite complete.")
    return results


if __name__ == "__main__":
    run_baselines()