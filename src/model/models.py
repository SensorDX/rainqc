from hurdle_regression import MixLinearModel, LinearRegression
class ModelFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'MixLinearModel':
            return MixLinearModel()
        if model_name == 'LinearRegression':
            return LinearRegression()




