from abc import ABCMeta, abstractmethod
from datetime import datetime
from dateutil import tz
import pandas as pd
from influxdb import InfluxDBClient

import requests

def flag_it(value):
    if value>-7:
        return 1
    else:
        return 0
posturl = "http://localhost:8086/write?db=rainqc"

class DataSource(object):
    __metaclass__ = ABCMeta

    def __init__(self, modeldb=None):

        self.__name__ = "data source"
        if modeldb is not None:
            assert isinstance(modeldb, ModelDB)
        self.modeldb = modeldb
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


    def fitted_stations(self, ):
        """

        Returns:
            fitted model. Which the saved model for the stations.

        """
        query = {}
        selector = {'station': 1, '_id': 0}
        # try:
        #     if self.modeldb is None:
        #         raise Exception("Modeldb is not configured.")
        #
        #     result = self.modeldb['model'].find(query, selector)  # query_fitted_stations)
        #     if result is None:
        #         raise ValueError("There are no fitted saved models.")
        #
        #     return result
        # except Exception as ex:
        #     return ValueError(str(ex))
        return self.modeldb.fetch(collection_name='model', query=query, selector=selector)



    def get_model(self, query, selector=None):
        # query = {'station': target_station, 'weather_variable': weather_variable}
        #model_config = self.modeldb['model'].find_one(query)  # self.fitted_stations(selector={}, query=query) #self.modeldb.db['model'].find_one(query)
        model_config = self.modeldb.fetch_one(collection_name='model', query=query)
        if model_config is None:
            raise ValueError("Couldn't find the model")
        return model_config

    def get_fitted_station(self, station_name):
        assert station_name is not None
        query = {'station':station_name}
        return self.get_model(query, None)
    def save_scores(self, scores, model_config, start_date, end_date, flag_it=flag_it):
        if len(scores)<1:
            return
        try:
            date_range = pd.date_range(start=start_date, end=end_date, freq='1D')
            _id, _station, _weather_variable = model_config['_id'], model_config['station'], model_config['weather_variable']

            score_result = {'model_id': _id , 'station': _station,
                        'weather_variable': _weather_variable,
                        'scores': {str(date): score for date, score in zip(date_range, scores)},

                        }

            #self.modeldb['qc_score'].insert(score_result)

            #self.modeldb.save(score_result, "qc_score")
            # Let's also save into influx.
            client = InfluxDBClient()

            score_json = []
            msr = ""
            for date, score in zip(date_range, scores):
                score_json.append({"measurement": 'testscore',
                                   "tags": {
                                       'model_id': _id,
                                       'station': _station,
                                       'weather_variable': _weather_variable,

                                   },
                                   "time": str(date),
                                   "fields": {
                                       "score": score,
                                       "flag": flag_it(score)
                                   }})
                msr += "testscore, model_id={},station={},weather_variable={},time={} score={} " \
                       "flag={}\n".format(_id, _station, _weather_variable, str(date), score, flag_it(score))

            #r= requests.post(posturl, data=msr)
            client.write_points(score_json, database="rainqc")
            print("result from influx")
            #print(r.json())
            return True
        except Exception as ex:
            return ValueError(str(ex))

    def save_score_pool(self, score_pool):
        # Save score operation.
        if score_pool is None:
            return
        self.modeldb.save(collection_name="score_pool", document=score_pool)
#        self.modeldb['score_pool'].insert(score_pool)
    def last_failure_pool(self):

        #last_failure = self.modeldb['score_pool'].find().sort('_id',-1).limit(1)
        last_failure = self.modeldb.fetch(collection_name="score_pool", query={}, selector=None).sort('_id', -1).limit(1)

        last_failure = list(last_failure)
        print(last_failure)
        if len(last_failure)>0:
            return last_failure[0]
        else:
            return [{"failure":[]}][0]


class ModelDB(object):
    """
    Wrapper for Mongodb
    """
    def __init__(self, mongo):
        #self.config = config
        self.mongo  = mongo

    def save(self, collection_name, document):
        try:
            self.mongo[collection_name].insert(document)
        except Exception as ex:
            raise ValueError(str(ex))

    def fetch(self, collection_name, query={}, selector={}):
        if selector is not None:
            return self.mongo[collection_name].find(query, selector)
        return  self.mongo[collection_name].find(query)
    def fetch_one(self, collection_name, query, selector=None):
        if selector is None:
            return self.mongo[collection_name].find_one(query)
        else:
            return self.mongo[collection_name].find_one(query, selector)
    def delete(self, collection_name, query):
        return self.mongo[collection_name].delete_many(query)

class InfluxAdapter(object):
    def __init__(self, connection_string):
        self.client = InfluxDBClient(connection_string)
    def save(self, collection_name, json_doc):
        self.client.write_points(json_doc, database=collection_name)
    def read(self):
        pass
