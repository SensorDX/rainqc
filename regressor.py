
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble.forest import RandomForestRegressor
class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = LinearRegression()
    def fit(self, X,y):
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)


import pandas as pd

def random_forest_regressor():

    rf = RandomForestRegressor(n_estimators=100)
    rf.fit()




