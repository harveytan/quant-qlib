from sklearn.linear_model import Ridge

def train_linear_baseline(X_train, y_train, alpha=1.0):
    """
    Ridge regression baseline.
    Interpretable, stable, and fast.
    """
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model