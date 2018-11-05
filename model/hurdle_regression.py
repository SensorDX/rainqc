from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
import json
import pickle
import os
import matplotlib.pylab as plt
from sklearn.externals import joblib
import numpy as np
from sklearn.model_selection import GridSearchCV
import seaborn as sbn
import common.utils as utils
import logging

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


class Module(object):
    def __init__(self):
        self.name = "Regression_module"
    def fit(self, x, y, verbose=False, load=False):
        return NotImplementedError
    def predict(self, x, y, label=None):
        return NotImplementedError

class MixLinearModel(Module):
    """
        Mixture of linear model.
        Train logistic regression for 0/1 prediction. And fit weighted linear regression, 
        with weight from output of the logistic regression. 
        Fit mixture of linear-model for rainy and non-rainy events. 

    """

    def __init__(self, linear_reg=LinearRegression(), log_reg=LogisticRegression(),
                 kde=KernelDensity(kernel="gaussian"), eps=0.0001):
        super(MixLinearModel, self).__init__()
        self.linear_reg = linear_reg
        self.eps = eps 
        self.log_reg = log_reg
        self.kde = kde
        self.fitted = False
        self.residual = False


    def residual_plot(self, observed, true_value, fitted):
        plt.scatter(true_value, np.log(observed))
        plt.plot(true_value, fitted, '-r')
        plt.xlabel('Log (predictor + eps)')
        plt.ylabel('Log (response + eps)')
        plt.show()
    def residual_density_plot(self, residual):
        plt.subplot(211)
        sbn.distplot(residual,hist=True )
        plt.subplot(212)
        sbn.kdeplot(residual)
        #plt.show()
    def grid_fit_kde(self, residual):
        from sklearn.model_selection import GridSearchCV
        grid = GridSearchCV(KernelDensity(), {'bandwidth':np.linspace(0.1,1.0,20)}, cv=20)
        grid.fit(residual)
        return grid.best_params_

    def fit(self, x, y, verbose=False, load=False):
        """
        Args:
            y: Nx1 ndarray observed value.
            x: NxD ndarry features.

        Returns:

        """
        if verbose:
            print (type(x), type(y), x.shape, y.shape)

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return NameError("The input should be given as ndarray")

        l_x, l_y = np.log(x + self.eps), np.log(y + self.eps)
        y_zero_one = (y > 0.0).astype(int)
        sample_weight = None
        if y_zero_one.max() != y_zero_one.min():
            self.log_reg.fit(x, y_zero_one)
            sample_weight = self.log_reg.predict_proba(x)[:, 1]

        # Linear regression under log mode.
        self.linear_reg.fit(X=l_x, y=l_y, sample_weight=sample_weight)
        self.fitted = self.linear_reg.predict(l_x)
        self.residual = (self.fitted - l_y)
        # Grid fit for bandwidith.
        if load is False:

            param = grid_fit_kde(self.residual)
            self.kde = KernelDensity(bandwidth=param["bandwidth"])
            self.kde.fit(self.residual)


        else:
            self.kde = pickle.load(open("all_kde.kd","rb"))
        logger.debug("KDE bandwidth %s"%self.kde.bandwidth)
        return self

    def predict(self, x, y, label=None):
        """
        Predict log-likelihood of given observation under the trained model.
        Args:
            y: ndarray Ground truth observation.
            x: ndarray matrix Features.
            label: None,

        Returns:
        """

        log_pred = self.log_reg.predict_proba(x)[:, 1]
        linear_pred = self.linear_reg.predict(np.log(x + self.eps))
        return self.mixl(y, log_pred, linear_pred)

    def mixl(self, y, logreg_prediction, linear_predictions):

        """
         - if RAIN = 0, $ -log (1-p_1)$
         - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$
        Args:

         y: (np.array) observations.
         logreg_prediction:(np.array) fitted values from logistic regression (0/1 model).
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
        #np.savetxt("debug.txt",np.hstack([observations,zero_rain,residual,residual_density, non_zero_rain, result]),delimiter=',')
        return -np.log(result + np.max(result))

    def to_json(self, model_id="001", model_path="rainqc_model"):

        model_config = {
            "model_id": model_id,
            "kde_model": pickle.dumps(self.kde),
            "logistic_model": pickle.dumps(self.log_reg),
            "linear_model": pickle.dumps(self.linear_reg)
        }
        json.dump(model_config, open(os.path.join(model_path, model_id + ".localdatasource"), "wb"))

    def from_json(self, model_id="001", model_path="rainqc_model"):
        js = json.load(os.path.join(model_path, model_id + ".localdatasource"), "rb")
        self.kde = pickle.loads(js['kde_model'])
        self.linear_reg = pickle.loads(js['linear_model'])
        self.log_reg = pickle.loads(js['logistic_model'])

    def save(self, model_id="001", model_path="rainqc_model"):
        """
        save the reg model.
        Returns:

        """
        # model_config = {"model_id":model_id,"kde":self.kde, "zeroone":self.log_reg,"regression":self.linear_reg}
        # localdatasource.dump(model_config,open(model_id+".localdatasource","wb"))
        current_model = os.path.join(model_path, model_id)
        if not os.path.exists(current_model):
            os.makedirs(current_model)
        joblib.dump(self.kde, os.path.join(current_model, "kde_model.sv"))
        joblib.dump(self.linear_reg, os.path.join(current_model, "linear_model.sv"))
        joblib.dump(self.log_reg, os.path.join(current_model, "logistic_model.sv"))

    @classmethod
    def load(cls, model_id="001", model_path="rainqc_model"):
        loaded_model = os.path.join(model_path, model_id)
        # model_config = localdatasource.load(open(model_id+".localdatasource","rb"))
        if not os.path.exists(loaded_model):
            return ValueError("Directory for saved models don't exist")

        reg_model = joblib.load(os.path.join(loaded_model, "linear_model.sv"))
        kde = joblib.load(os.path.join(loaded_model, "kde_model.sv"))
        log_reg = joblib.load(os.path.join(loaded_model, "logistic_model.sv"))  # pickle.load(model_config['zerone'])
        mxll = MixLinearModel(linear_reg=reg_model, log_reg=log_reg, kde=kde)
        return mxll
