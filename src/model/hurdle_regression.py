from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
import pickle
import os
import matplotlib.pylab as plt
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sbn
import logging
from absmodel import Module
logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
logging.basicConfig(filename="logfile.log",
                    level=logging.DEBUG, format=logger_format,
                    filemode='w')  # use filemode='a' for APPEND
logger = logging.getLogger(__name__)


def grid_fit_kde(residual):
    """
    Grid search for best bandwidth of KDE
    Args:
        residual: residual value.

    Returns:

    """
    grid = GridSearchCV(KernelDensity(), {'bandwidth':np.linspace(0.1,1.0,20)}, cv=20)
    grid.fit(residual)
    return grid.best_params_




class MixLinearModel(Module):
    """
        Mixture of linear src.
        Train logistic regression for 0/1 prediction. And fit weighted linear regression, 
        with weight from output of the logistic regression. 
        Fit mixture of linear-src for rainy and non-rainy events.

    """

    def __init__(self, linear_reg=LinearRegression(), log_reg=LogisticRegression(),
                 kde=KernelDensity(kernel="gaussian"), eps=0.0001, offset = -.05):
        super(MixLinearModel, self).__init__()
        self.linear_reg = linear_reg
        self.eps = eps 
        self.log_reg = log_reg
        self.kde = kde
        self.fitted = False
        self.residual = False
        self.offset= offset

    @staticmethod
    def residual_plot(observed, true_value, fitted):
        plt.scatter(true_value, np.log(observed))
        plt.plot(true_value, fitted, '-r')
        plt.xlabel('Log (predictor + eps)')
        plt.ylabel('Log (response + eps)')
        plt.show()

    @staticmethod
    def residual_density_plot(residual):
        plt.subplot(211)
        sbn.distplot(residual,hist=True )
        plt.subplot(212)
        sbn.kdeplot(residual)

    @staticmethod
    def grid_fit_kde(residual):
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(KernelDensity(), {'bandwidth':np.linspace(0.1,1.0,20)}, cv=20)
        grid.fit(residual)
        return grid.best_params_

    def _fit(self, x, y, verbose=False, load=False):
        """
        Args:
            y: Nx1 ndarray observed value.
            x: NxD ndarry features.

        Returns:

        """


        l_x, l_y = np.log(x + self.eps), np.log(y + self.eps)
        y_zero_one = (y > 0.0).astype(int)

        if y_zero_one.max() == y_zero_one.min():
            raise NotFittedError("Logistic model couldn't fit, because the number of classes is <2")

        self.log_reg.fit(x, y_zero_one)
        sample_weight = self.log_reg.predict_proba(x)[:, 1]

        # Linear regression under log mode.
        self.linear_reg.fit(X=l_x, y=l_y, sample_weight=sample_weight)
        self.fitted = self.linear_reg.predict(l_x)
        self.residual = (self.fitted - l_y)

        # Grid fit for bandwidth.
        if load is False:

            param = grid_fit_kde(self.residual)
            self.kde = KernelDensity(bandwidth=param["bandwidth"])
            self.kde.fit(self.residual)

        else:
            self.kde = pickle.load(open("all_kde.kd","rb"))
        self.fitted = True
        #logger.debug("KDE bandwidth %s"%self.kde.bandwidth)
        return self

    def predict(self, x, y, label=None):
        """
        Predict log-likelihood of given observation under the trained src.
        Args:
            y: ndarray Ground truth observation.
            x: ndarray matrix Features.
            label: None,

        Returns:
        """
        if self.fitted is False:
            raise NotFittedError("Call fit before prediction")
        
        log_pred = self.log_reg.predict_proba(x)[:, 1]
        linear_pred = self.linear_reg.predict(np.log(x + self.eps))
        return self.mixl(y, log_pred, linear_pred)
    def decision_function(self, score):
        """
        Return decision based on the anomaly score.
        Args:
            x:
            y:
            label:

        Returns:

        """
        return score - self.offset


    def mixl(self, y, logreg_prediction, linear_predictions):

        """
         - if RAIN = 0, $ -log (1-p_1)$
         - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$
        Args:

         y: (np.array) observations.
         logreg_prediction:(np.array) fitted values from logistic regression (0/1 src).
         linear_predictions:(np.array) fitted values from linear regression on log scale.

        """
        # Reshape the data
        p = logreg_prediction.reshape([-1, 1])
        observations = y.reshape([-1, 1])
        predictions = linear_predictions.reshape([-1, 1])

        zero_rain = np.multiply((1 - p), (observations == 0))
        # density of residual and convert to non-log value.
        residual = predictions - np.log(observations + self.eps)
        residual_density = np.exp(self.kde.score_samples(residual)).reshape(-1,1)

        non_zero_rain = np.divide(np.multiply(p, residual_density),
                                         (observations + self.eps))
        result = zero_rain + non_zero_rain
        return -np.log(result + np.max(result))

    def to_json(self):
        if not self.fitted:
            raise NotFittedError("Fit method should be called before save operation.")
        model_config = {
            "kde_model": self.kde,
            "logistic_model": self.log_reg,
            "linear_model": self.linear_reg
        }
        return model_config

    @classmethod
    def from_json(cls, model_config):

        mlm = MixLinearModel(linear_reg=model_config['linear_model'], log_reg=model_config['logistic_model'],
                             kde=model_config['kde_model'])
        mlm.fitted = True

        return mlm

    def save(self, model_id="001", model_path="rainqc_model"):
        """
        save the reg src.
        Returns:

        """
        # model_config = {"model_id":model_id,
        #                 "kde":self.kde,
        #                 "logistic_reg":self.log_reg,
        #                 "linear_regression":self.linear_reg}
        # localdatasource.dump(model_config,open(model_id+".localdatasource","wb"))
        current_model = os.path.join(model_path, model_id)
        if not os.path.exists(current_model):
            os.makedirs(current_model)
        joblib.dump(self.kde, os.path.join(current_model, "kde_model.pk"))
        joblib.dump(self.linear_reg, os.path.join(current_model, "linear_model.pk"))
        joblib.dump(self.log_reg, os.path.join(current_model, "logistic_model.pk"))

    @classmethod
    def load(cls, model_id="001", model_path="rainqc_model"):
        loaded_model = os.path.join(model_path, model_id)
        # model_config = localdatasource.load(open(model_id+".localdatasource","rb"))
        if not os.path.exists(loaded_model):
            return ValueError("Directory for saved models don't exist")

        reg_model = joblib.load(os.path.join(loaded_model, "linear_model.pk"))
        kde = joblib.load(os.path.join(loaded_model, "kde_model.pk"))
        log_reg = joblib.load(os.path.join(loaded_model, "logistic_model.pk"))  # pickle.load(model_config['zerone'])
        mxll = MixLinearModel(linear_reg=reg_model, log_reg=log_reg, kde=kde)
        mxll.fitted = True
        return mxll
