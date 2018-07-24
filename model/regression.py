import numpy as np
import statsmodels.api as st
import statsmodels.formula.api as sfa
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.base import BaseEstimator
from sklearn.linear_model.logistic import LogisticRegression

class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = LinearRegression()
    def fit(self, X,y):
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)




kde = KernelDensity()

class MixLinearModel(object):
    """
    Mixture of linear model.
     - 0/1 Hurdle model.
    """

    def __init__(self):
        self.reg_model = LinearRegression()
        self.hurdle_model = sfa.GLM()
        self.log_reg = LogisticRegression()
    def train(self, y, x):
        """

        Args:
            y: observed value
            x: features

        Returns:

        """
        zero_one = np.int(y>0)
        self.log_reg.fit(x,zero_one)
        self.reg_model.fit(X=x,y=zero_one, sample_weight=self.log_reg.fit_intercept)

        ## Train the hurdle model and use the fitted value to train the regression model.
        # - train reg_model
    def save(self):
        """
        save the reg model.
        Returns:

        """

    def predict(self, x):
        p = self.hurdle_model.fit(x)
        linear_pred = self.reg_model.predict(x)
        return self.__mixl(x, p, linear_pred)

    def __mixl(self, x, p, linear_predictions):
        """
         - if RAIN = 0, $ -log (1-p_1)$
         - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$



        Args:
         observations: ground observation.
         p1: 0/1 prediction model.
         predictions: fitted values for the log(rain + epsilon) model

        """
        # Reshape for 1-D format.
        eps = 0.001
        p = p.reshape([-1, 1])
        observations = x.reshape([-1, 1])
        predictions = linear_predictions.reshape([-1, 1])

        zero_rain = np.multiply((1 - p), (observations == 0))
        non_zero = np.divide(np.multiply(p,
                                         np.exp(kde.score_samples(predictions - np.log(observations + eps))).reshape(
                                             [-1, 1])),
                             abs(observations + eps))

        result = zero_rain + non_zero
        return result