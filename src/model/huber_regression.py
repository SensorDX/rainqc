from .absmodel import Module

class HuberRegressionModel(Module):

    def fit(self, x, y, verbose=False, load=False):
        return super(HuberRegressionModel, self).fit(x, y, verbose, load)

    def predict(self, x, y, label=None):
        return super(HuberRegressionModel, self).predict(x, y, label)

    def __init__(self):
        super(HuberRegressionModel, self).__init__()
        pass




