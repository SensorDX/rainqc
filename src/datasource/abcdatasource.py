from abc import ABCMeta, abstractmethod
from datetime import datetime
from dateutil import tz
import pandas as pd
class DataSource(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self.__name__ = "data source"

    @abstractmethod
    def stations(self):
        return NotImplementedError

    @abstractmethod
    def get_data(self, station_name, start_date, end_date, data_format="json"):
        return NotImplementedError

    @abstractmethod
    def daily_data(self, station_name, weather_variable, start_date, end_date):
        return NotImplementedError

    @abstractmethod
    def nearby_stations(self, target_station, k, radius):
        return NotImplementedError
    @abstractmethod
    def active_stations(self, station_list, active_day_range):
        return NotImplementedError

    def to_json(self):
        return NotImplementedError
    @classmethod
    def from_json(cls, json_config):
        return NotImplementedError

    def online_stations(self, active_day_range=datetime.now(tz.tzutc()), threshold=24):
        return NotImplementedError

    def set_modeldb(self, modeldb):
        self.modeldb = modeldb


    def fitted_stations(self, query={}, selector={'station': 1, '_id': 0}):
        """

        Returns:
            fitted model. Which the saved model for the stations.

        """
        try:
            if self.modeldb is None:
                raise Exception("Modeldb is not configured.")

            result = self.modeldb['model'].find(query, selector)  # query_fitted_stations)
            if result is None:
                raise ValueError("There are no fitted saved models.")

            return result
        except Exception as ex:
            return ValueError(str(ex))


    def get_model(self, query, selector=None):
        # query = {'station': target_station, 'weather_variable': weather_variable}
        model_config = self.modeldb['model'].find_one(query)  # self.fitted_stations(selector={}, query=query) #self.modeldb.db['model'].find_one(query)
        if model_config is None:
            raise ValueError("Couldn't find the model")
        return model_config

    def get_fitted_station(self, station_name):
        assert station_name is not None
        query = {'station':station_name}
        return self.get_model(query, None)
    def save_scores(self, scores, model_config, start_date, end_date):
        if len(scores)<1:
            return
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
            score_result = {'model_id': model_config['_id'], 'station': model_config['station'],
                        'weather_variable': model_config['weather_variable'],
                        'scores': {str(date): score for date, score in zip(date_range, scores)}
                        }

            self.modeldb['qc_score'].insert(score_result)

            return True
        except Exception as ex:
            return ValueError(str(ex))
    def save_score_pool(self, score_pool):
        # Save score operation.
        if score_pool is None:
            return
        self.modeldb['score_pool'].insert(score_pool)
    def last_failure_pool(self):

        #last_failure = self.modeldb['score_pool'].find_one({},{'_id':-1}) #find().sort({'_id': -1}).limit(1)
        last_failure = self.modeldb['score_pool'].find().sort('_id',-1).limit(1)
        last_failure = list(last_failure)
        print(last_failure)
        if len(last_failure)>0:
            return last_failure[0]
        else:
            return [{"failure":[]}][0]
class ModelDB(object):


    def __init__(self, mongo):
        #self.config = config
        self.mongo  = mongo

    def save(self, model_json, collection_name):
        try:
            self.mongo.db[collection_name].insert(model_json)
        except Exception as ex:
            raise ValueError(str(ex))

    def fetch(self, collection_name, query, selector):
        self.mongo.db[collection_name].find(query, selector)

    def fetch_one(self, collection_name, query, selector):
        return self.mongo.db[collection_name].find_one(query, selector)
    def delete(self, query):
        pass
