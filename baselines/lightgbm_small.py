import lightgbm as lgb

def train_small_lgbm(X_train, y_train):
    """
    Small LightGBM baseline.
    Classical ML anchor: low variance, hard to beat.
    """
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves": 7,
        "max_depth": 3,
        "learning_rate": 0.05,
        "n_estimators": 50,
        "verbosity": -1,
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    return model