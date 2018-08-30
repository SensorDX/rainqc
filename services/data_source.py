#from util.stations_tools import nearby_stations
import json
import pandas as pd
import numpy as np
import os
import json


def json_to_df(json_station_data, weather_variable='pr', group='D', filter_year=2017):
    rows = [{'date': row['date'], weather_variable: row['measurements'][weather_variable]} for row in json_station_data]
    df = pd.DataFrame(rows)
    df.date = pd.to_datetime(df.date)
    if filter_year:
        df = df[df.date.dt.year == filter_year]
    if group:
        df = df.groupby(df.date.dt.dayofyear).agg({weather_variable:np.sum, "date":np.max}) #apply(lambda x: np.sum(x[weather_variable]))  ## take max readings of the hour.
    return df



class DataSource(object):
    """
    Abstract class for accessing data source
    """
    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):
        return NotImplementedError
    @staticmethod
    def station_list():
        return NotImplementedError

    @staticmethod
    def measurements(station_name, variable, date_from, date_to, **kwargs):
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
    local_project_path = "../"
    @staticmethod
    def json_measurements(station_name):
        full_path = os.path.join(LocalDataSource.local_project_path, "tahmodata/"
                                                                     "rm_" + station_name + ".json")
        jj = json.load(open(full_path, "rb"))
        return jj

    @staticmethod
    def measurements(station_name, weather_variable, date_from, date_to, **kwargs):
        json_data = LocalDataSource.json_measurements(station_name)
        df = json_to_df(json_data, group=kwargs.get('group'), filter_year=kwargs.get('filter_year'),
                        weather_variable=weather_variable)
        df = df[((df.date<date_to) & (df.date>date_from))]
        return df

    @staticmethod
    def station_list():
        station_list = [ stn.split('_')[1].split('.json')[0] for stn in os.listdir(
            os.path.join(LocalDataSource.local_project_path,'tahmodata'))]
        return station_list
    @staticmethod
    def __available_station(sorted_nearest_station, num_k_station):

        online_station = LocalDataSource.station_list()
        available_station = []

        for station in sorted_nearest_station:
            if station in online_station:
                available_station.append(station)
                num_k_station -=1
            if num_k_station==0:
                break
        return available_station



    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):

        stations = pd.read_csv(os.path.join(LocalDataSource.local_project_path,"nearest_stations.csv"))  # Pre-computed value.
        k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]

        k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)['to']  #[0:k]
        print k_nearest.tolist()
        available_stations = LocalDataSource.__available_station(k_nearest, k)
        return available_stations


