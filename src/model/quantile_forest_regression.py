from sklearn.ensemble.forest import RandomForestRegressor
import numpy as np
from . import absmodel

class QuantileForestRegression(absmodel.Module):


    def __init__(self, n_estimator=500):
        super(QuantileForestRegression, self).__init__()
        self.model = RandomForestRegressor(n_estimators=n_estimator)

        self.fitted = False
    def _fit(self, x, y, verbose=False, load=False):

        return self.model.fit(x,y)

    def predict(self, x, y, label=None):

        d, up = self.pred_ints(model=self.model, x=x)
        return d, up

    def pred_ints(self, model, x, percentile=95):
        err_down = []
        err_up = []
        for i in range(len(x)):
            preds = []
            for pred in model.estimators_:
                preds.append(pred.predict(x[i])[0])
            err_down.append(np.percentile(preds, (100 - percentile) / 2.))
            err_up.append(np.percentile(preds, 100 - (100 - percentile) / 2.))
        return err_down, err_up