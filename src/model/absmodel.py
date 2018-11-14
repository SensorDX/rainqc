
class Module(object):
    def __init__(self):
        self.name = "Regression_module"

    def fit(self, x, y, verbose=False, load=False):
        return NotImplementedError

    def predict(self, x, y, label=None):
        return NotImplementedError