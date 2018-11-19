from . import TahmoAPILocal, TahmoDataSource, FakeTahmo
class DataSourceFactory:
    @staticmethod
    def get_model(model_name):
        if model_name == 'FakeTahmo':
            return FakeTahmo()
        if model_name == 'TahmoDataSource':
            return TahmoDataSource()
        if model_name =='TahmoAPILocal':
            return TahmoAPILocal()
        return None
