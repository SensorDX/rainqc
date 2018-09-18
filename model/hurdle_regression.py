import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
import json
import pickle
import os
import matplotlib.pylab as plt
from sklearn.externals import joblib
import seaborn as sbn
import numpy as np
class ModelFactory:
    @staticmethod
    def create_model(model_name):
        if model_name=='MixLinear':
            return MixLinearModel()
        if model_name=='Linear':
            return LinearRegression()


class MixLinearModel(object):

    """
        Mixture of linear model.
        0/1 Hurdle model.
    """
    def __init__(self, linear_reg=LinearRegression(), log_reg = LogisticRegression(),
                 kde =KernelDensity(kernel="gaussian")):
        self.reg_model = linear_reg
        self.eps = 0.001
        self.log_reg = log_reg
        self.kde = kde
        self.fitted = False
        self.residual = False

    def residual_plot(self, observed, true_value, fitted ):
        plt.scatter(true_value, np.log(observed))
        plt.plot(true_value, fitted, '-r')
        plt.xlabel('Log (predictor + eps)')
        plt.ylabel('Log (response + eps)')
        plt.show()

    def fit(self, x, y, verbose=False):
        """
        Args:
            y: 1-D Nx1 ndarray observed value.
            x: NxD ndarry features.

        Returns:

        """
        if verbose:
            print type(x), type(y), x.shape, y.shape

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return NameError("The input should be given as ndarray")


        #if y.ndim <2:

        l_X, l_y = np.log(x + self.eps), np.log(y + self.eps)
        y_zero_one = (y>0.0).astype(int)
        sample_weight = None
        if y_zero_one.max()!=y_zero_one.min():
            self.log_reg.fit(x, y_zero_one)
            sample_weight = self.log_reg.predict_proba(x)[:,1]

        # Linear regression under log mode.
        self.reg_model.fit(X=l_X, y=l_y, sample_weight=sample_weight)
        self.fitted = self.reg_model.predict(l_X)
        self.residual = (self.fitted - l_y)
        self.kde.fit(self.residual)
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
        logreg_pred = self.log_reg.predict_proba(x)[:,1]
        linear_pred = self.reg_model.predict(np.log(x+self.eps))
        return self.__mixl(y, logreg_pred, linear_pred)


    def __mixl(self, y, logreg_prediction, linear_predictions):

        """
         - if RAIN = 0, $ -log (1-p_1)$
         - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$
        Args:

         observations: ground observation.
         p1: 0/1 prediction model.
         predictions: fitted values for the log(rain + epsilon) model

        """
        # This
        p = logreg_prediction.reshape([-1, 1])
        observations = y.reshape([-1, 1])
        predictions = linear_predictions.reshape([-1, 1])

        zero_rain = np.multiply((1 - p), (observations == 0))
        non_zero = np.divide(np.multiply(p,
                                         np.exp(self.kde.score_samples(predictions - np.log(observations + self.eps))).reshape(
                                             [-1, 1])),
                             abs(observations + self.eps))

        result = zero_rain + non_zero
        return result

    def to_json(self, model_id="001", model_path="rainqc_model"):

        model_config = {
            "model_id":model_id,
                        "kde_model":pickle.dumps(self.kde),
                        "logistic_model":pickle.dumps(self.log_reg),
                        "linear_model":pickle.dumps(self.reg_model)
                        }
        json.dump(model_config,open(os.path.join(model_path,model_id+".localdatasource"),"wb"))

    def from_json(self,model_id="001", model_path="rainqc_model"):
        js = json.load(os.path.join(model_path,model_id+".localdatasource"),"rb")
        self.kde = pickle.loads(js['kde_model'])
        self.reg_model = pickle.loads(js['linear_model'])
        self.log_reg = pickle.loads(js['logistic_model'])

    def save(self,model_id="001", model_path="rainqc_model"):
        """
        save the reg model.
        Returns:

        """
        #model_config = {"model_id":model_id,"kde":self.kde, "zeroone":self.log_reg,"regression":self.reg_model}
        #localdatasource.dump(model_config,open(model_id+".localdatasource","wb"))
        current_model = os.path.join(model_path,model_id)
        if not os.path.exists(current_model):
            os.makedirs(current_model)
        joblib.dump(self.kde, os.path.join(current_model,"kde_model.sv"))
        joblib.dump(self.reg_model,os.path.join(current_model, "linear_model.sv"))
        joblib.dump(self.log_reg, os.path.join(current_model, "logistic_model.sv"))
    @classmethod
    def load(cls, model_id="001", model_path="rainqc_model"):
        loaded_model = os.path.join(model_path, model_id)
        #model_config = localdatasource.load(open(model_id+".localdatasource","rb"))
        if not os.path.exists(loaded_model):
            return ValueError("Directory for saved models don't exist")

        reg_model = joblib.load(os.path.join(loaded_model,"linear_model.sv"))
        kde = joblib.load(os.path.join(loaded_model,"kde_model.sv"))
        log_reg = joblib.load(os.path.join(loaded_model,"logistic_model.sv")) #pickle.load(model_config['zerone'])
        mxll = MixLinearModel(linear_reg=reg_model, log_reg=log_reg, kde=kde)
        return mxll

