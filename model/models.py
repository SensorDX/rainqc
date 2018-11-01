from hurdle_regression import MixLinearModel, LinearRegression
class ModelFactory:
    @staticmethod
    def create_model(model_name):
        if model_name == 'MixLinear':
            return MixLinearModel()
        if model_name == 'Linear':
            return LinearRegression()


class Module:
    pass