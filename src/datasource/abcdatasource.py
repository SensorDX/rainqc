from abc import ABCMeta, abstractmethod

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


