import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression, LogisticRegression
import json
import pickle
import os
from sklearn.externals import joblib


class MixLinearModel(object):

    """
        Mixture of linear model.
        0/1 Hurdle model.
    """

    def __init__(self):
        self.reg_model = LinearRegression()

        self.eps = 0.001
        self.log_reg = LogisticRegression()
        self.kde = KernelDensity(kernel="gaussian")
    def train(self, y, x):
        """

        Args:
            y: observed value
            x: features

        Returns:

        """
        ## fit regression on log scale.
        y = y.reshape(-1,1)

        zero_one = (y>0.5).astype(int)
        self.log_reg.fit(x, zero_one)
        sample_weight = self.log_reg.predict_proba(x)[:,1]
        self.reg_model.fit(X=np.log(x+self.eps),y=np.log(y+self.eps),sample_weight=sample_weight)
        res = (y - self.reg_model.predict(np.log(x+self.eps)))
        self.kde.fit(res)

        ## Train the hurdle model and use the fitted value to train the regression model.
        # - train reg_model
    def to_json(self, model_id="001", model_path="rainqc_model"):

        model_config = {"model_id":model_id,
                        "kde_model":pickle.dumps(self.kde),
                        "logistic_model":pickle.dumps(self.log_reg),
                        "linear_model":pickle.dumps(self.reg_model)
                        }
        json.dump(model_config,open(os.path.join(model_path,model_id+".json"),"wb"))

    def from_json(self,model_id="001", model_path="rainqc_model"):
        js = json.load(os.path.join(model_path,model_id+".json"),"rb")
        self.kde = pickle.loads(js['kde_model'])
        self.reg_model = pickle.loads(js['linear_model'])
        self.log_reg = pickle.loads(js['logistic_model'])

    def save(self,model_id="001", model_path="rainqc_model"):
        """
        save the reg model.
        Returns:

        """
        #model_config = {"model_id":model_id,"kde":self.kde, "zeroone":self.log_reg,"regression":self.reg_model}
        #json.dump(model_config,open(model_id+".json","wb"))
        current_model = os.path.join(model_path,model_id)
        if not os.path.exists(current_model):
            os.makedirs(current_model)
        joblib.dump(self.kde, os.path.join(current_model,"kde_model.sv"))
        joblib.dump(self.reg_model,os.path.join(current_model, "linear_model.sv"))
        joblib.dump(self.log_reg, os.path.join(current_model, "logistic_model.sv"))

    def load(self, model_id="001", model_path="rainqc_model"):
        loaded_model = os.path.join(model_path, model_id)
        #model_config = json.load(open(model_id+".json","rb"))
        if not os.path.exists(loaded_model):
            return ValueError("Directory for saved models don't exist")

        self.reg_model = joblib.load(os.path.join(loaded_model,"linear_model.sv"))
        self.kde = joblib.load(os.path.join(loaded_model,"kde_model.sv"))
        self.log_reg = joblib.load(os.path.join(loaded_model,"logistic_model.sv")) #pickle.load(model_config['zerone'])


    def predict(self, y, x):
        p = self.log_reg.predict(x)
        linear_pred = self.reg_model.predict(np.log(x+self.eps))
        return self.__mixl(y, p, linear_pred)


    def __mixl(self, y, p, linear_predictions):

        """
         - if RAIN = 0, $ -log (1-p_1)$
         - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$



        Args:
         observations: ground observation.
         p1: 0/1 prediction model.
         predictions: fitted values for the log(rain + epsilon) model

        """
        # Reshape for 1-D format.
        #eps = 0.001
        p = p.reshape([-1, 1])
        observations = y.reshape([-1, 1])
        predictions = linear_predictions.reshape([-1, 1])

        zero_rain = np.multiply((1 - p), (observations == 0))
        non_zero = np.divide(np.multiply(p,
                                         np.exp(self.kde.score_samples(predictions - np.log(observations + self.eps))).reshape(
                                             [-1, 1])),
                             abs(observations + self.eps))

        result = zero_rain + non_zero
        return result

