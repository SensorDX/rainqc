import requests
import json, datetime
from common.weather_variable import *
import requests
import numpy as np
import pandas as pd
import dateutil
from common.sdxutils import average_angular
from collections import OrderedDict

variable_aggregation = {RAIN: np.sum, TEMP: np.mean, WINDR: average_angular, REL: np.mean}


def json_to_df(json_station_data, weather_variable='pr', group='D', filter_year=None):
    rows = json_station_data["timeseries"][weather_variable]
    df = pd.DataFrame.from_dict(rows, orient="index")
    df.rename(columns={0: weather_variable}, inplace=True)
    df['date'] = df.index
    df.index = np.arange(len(rows))
    df.date = pd.to_datetime(df.date)

    if filter_year:
        df = df[df.date.dt.year == filter_year]
    if group:
        df = df.groupby(df.date.dt.dayofyear).agg({weather_variable: variable_aggregation[weather_variable],
                                                   "date": np.max})  # apply(lambda x: np.sum(x[weather_variable]))  ## take max readings of the hour.
    return df


def nearby_stations(site_code, radius=100, data_path="nearest_station.csv"):
    stations = pd.read_csv(data_path)  # Pre-computed value.
    k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]
    k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)  # [0:k]
    return k_nearest


class TahmoDataSource(object):

    def __init__(self, nearby_station_location="nearest_station.csv"):
        # Later will move to config.
        self.header = {
            'authorization': "Basic NldZSFlUMFhWWTdCWFpIWE43SEJLWUFaODpSazdwWnBkSjBnd3hIVkdyM2twYnBIWDZwOGZrMitwSmhoS0F4Mk5yNzdJ",
            'cache-control': "no-cache"
        }

        self.time_series_url = "https://tahmoapi.mybluemix.net/v1/timeseries/%s/rawMeasurements"
        self.station_url = "https://tahmoapi.mybluemix.net/v1/stations"
        self.cm_url = "https://tahmoapi.mybluemix.net/v1/timeseries/%s%"
        self.nearby_station_file = nearby_station_location

    def __get_request(self, url, params={}):
        try:
            response = requests.request("GET", url, headers=self.header,
                                        params=params)
        except requests.HTTPError, err:
            if err.code == 401:
                return ValueError("Error: Invalid API credentials")
            elif err.code == 404:
                return ValueError("Error: The API endpoint is currently unavailable")
            else:
                return ValueError("Error: {}".format(err))
        return response

    def get_data(self, station_name, start_date, end_date, data_format="json"):
        querystring = {"startDate": start_date, "endDate": end_date}
        url = self.time_series_url % station_name
        json_data = self.__get_request(url, querystring).json()

        if data_format == "json":
            return json_data
        elif data_format == "dataframe":
            return json_to_df(json_data, group=None)


    def get_stations(self):
        station_list = self.__get_request(url=self.station_url)
        return station_list.json()


    def active_stations(self, station_list, active_day_range=datetime.datetime.utcnow(), threshold=24):
        """
        Return active station from the given list of stations during the active day
        Args:
            station_list (list): station name list
            active_day_range (str):datetime to check the last reported date. Default current date
            threshold (int): threshold to decide the station as inactive in hours. Default 24 hours

        Returns:

        """
        if len(station_list) < 1:
            return ValueError("The station list is empty")
        all_stations = self.get_stations()["stations"]
        active_stations = []
        for station in station_list:
            for stn_db in all_stations:
                if stn_db["id"] == station:
                    if not stn_db["active"]:
                        continue
                    last_measure = dateutil.parser.parse(stn_db["lastMeasurement"])  # Need to change to utc.
                    wait_time = divmod((active_day_range - last_measure).total_seconds(), 3600)
                    if wait_time[0] < threshold:
                        active_stations.append(station)
                        continue
        return active_stations
    def daily_data(self, station_name, weather_variable, start_date, end_date):
        json_data = self.get_data(station_name,  start_date, end_date)
        df = json_to_df(json_data, weather_variable=weather_variable, group='D')
        return df

    def nearby_stations(self, target_station, k=10, radius=100):
        k_nearby = nearby_stations(target_station, radius, self.nearby_station_file)[:k]
        oo_dict = OrderedDict()
        for _, row in k_nearby.iterrows():
            oo_dict[row['to']] = row['distance']
        #print oo_dict
        return oo_dict
    def get_active_nearby(self, target_station, k=10, radius=100):
        all_k_stations = nearby_stations(target_station, k, radius)
        return self.active_stations(all_k_stations.keys())



if __name__ == '__main__':
    target_station = "TA00021"
    thm = TahmoDataSource("../localdatasource/nearest_stations.csv")
    start_date = datetime.datetime.strftime(datetime.datetime.utcnow() - datetime.timedelta(20), '%Y-%m-%dT%H:%M')
    end_date = datetime.datetime.strftime(datetime.datetime.utcnow() - datetime.timedelta(10), '%Y-%m-%dT%H:%M')
    print start_date
    # print thm.get_stations()
    # print thm.get_data("TA00021", start_date="2017-09-01", end_date="2017-09-05")
    # print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)
    # get active stations
    station_list = ['TA00028', 'TA00068', "TA00108", "TA00187"]
    # print thm.get_active_station(station_list, active_day_range=datetime.datetime.utcnow())
    print thm.nearby_stations(target_station=target_station, k=20, radius=200)
    #thm = TahmoDataSource()
    # print thm.get_stations()
    # print thm.get_data("TA00021", start_date="2017-09-01", end_date="2017-09-05")
    print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)

