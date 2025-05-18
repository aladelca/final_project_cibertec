import joblib
from sklearn.ensemble import RandomForestRegressor

class Model:
    def __init__(self, params: dict):
        self.model = None
        self.params = params

    def fit(self, X, y):
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, path):
        joblib.dump(self.model, path)