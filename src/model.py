from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor


def create_linear_model():
    return LinearRegression()


def create_gradient_boosting_model():
    return HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model