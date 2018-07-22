import numpy as np
import statsmodels.api as st
import statsmodels.formula.api as sfa
from sklearn.neighbors import KernelDensity


class Regression:
    pass


kde = KernelDensity()

def mixl(observations, p1, predictions):
    """
     - if RAIN = 0, $ -log (1-p_1)$
     - if RAIN > 0, $ -log [p_1 \frac{P(log(RAIN + \epsilon)}{(RAIN + \epsilon)}]$



    Args:
     observations: ground observation.
     p1: 0/1 prediction model.
     predictions: fitted values for the log(rain + epsilon) model

    """
    # Reshape for 1-D format.
    p1 = p1.reshape([-1, 1])
    observations = observations.reshape([-1, 1])
    predictions = predictions.reshape([-1, 1])

    zero_rain = np.multiply((1 - p1), (observations == 0))
    non_zero = np.divide(np.multiply(p1,
                                     np.exp(kde.score_samples(predictions - np.log(observations + eps))).reshape(
                                         [-1, 1])),
                         abs(observations + eps))

    result = zero_rain + non_zero
    return result