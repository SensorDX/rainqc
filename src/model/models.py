from .hurdle_regression import MixLinearModel, LinearRegression
from .quantile_forest_regression import QuantileForestRegression
class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'MixLinearModel':
            return MixLinearModel()
        if model_name == 'LinearRegression':
            return LinearRegression()
        if model_name =='QuantileForestRegression':
            return QuantileForestRegression()




