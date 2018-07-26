#from util.stations_tools import nearby_stations
import json
import pandas as pd
import numpy as np
import os
import json


class DataSource(object):
    """
    Abstract class for accessing data source
    """
    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):
        return NotImplementedError
    @staticmethod
    def get_daily_rainfall(station_list, date):
        """
        Return rainfall readings of list of stations, if range of date-given, it returns numpy array of readings.

        Args:
            statioin_list: list(station_name)
            date: date of observation.

        Returns: Daily rainfall data from list of station, with the vector value of station.

        """
        return NotImplementedError

    @staticmethod
    def get_weather_data(station_name, variable, date_range):
        """
        Get weather data for a given variable of a station.
        Args:
            station_name:
            variable:
            date_range:

        Returns: JSON readings of the given query.

        """
        return NotImplementedError

class LocalDataSource(DataSource):
    local_project_path = "/home/tadeze/projects/sensordx/rainqc/"
    @staticmethod
    def get_daily_rainfall(station_list, date):
        #super(LocalDataSource, station_list).get_daily_rainfall(station_list, date)
        pass
    @staticmethod
    def get_weather_data(station_name, variable, date_range):
        #super(LocalDataSource, station_name).get_weather_data(station_name, variable, date_range)
        full_path = os.path.join(LocalDataSource.local_project_path,
                                 "tahmodata2/json","rm_"+station_name+".json")
        jj = json.load(open(full_path, "rb"))
        return jj

    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):

        stations = pd.read_csv(os.path.join(LocalDataSource.local_project_path,"nearest_stations.csv"))  # Pre-computed value.
        k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]
        k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)[0:k]
        return k_nearest.to.tolist()


