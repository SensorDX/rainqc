from sklearn.ensemble.forest import RandomForestRegressor
from absmodel import Module

class QuantileForestRegression(Module):


    def __init__(self, n_estimator):
        super(QuantileForestRegression, self).__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimator)

    def fit(self, x, y, verbose=False, load=False):
        self.fit(x,y)

    def predict(self, x, y, label=None):
        value = self.model.predict(x)
        return value

