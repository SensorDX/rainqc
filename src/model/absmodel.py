import numpy as np
class Module(object):
    def __init__(self):
        self.name = "Regression_module"

    def fit(self, x, y, verbose=False, load=False):
        if verbose:
            print (type(x), type(y), x.shape, y.shape)

        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return NameError("The input should be given as ndarray")
        return self._fit(x, y, verbose, load)

    def _fit(self, x, y, verbose=False, load=False):
        return NotImplementedError
    def predict(self, x, y, label=None):
        return NotImplementedError

    @staticmethod
    def validate(x, y):
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return NameError("The input should be given as ndarray")

        x = np.asarray(x)
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.dtype.kind == 'f'
        assert y.shape[1] == x.shape[1]
        assert y.dtype.kind == 'f'

        return x, y
