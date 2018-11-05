import json
from common.weather_variable import *
import requests
import numpy as np
import pandas as pd
from dateutil import parser, tz
from datetime import datetime, timedelta
from common.sdxutils import average_angular, haversine_distance
from operator import itemgetter



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
        except requests.HTTPError as err:
            if err.errno == 401:
                raise ValueError("Error: Invalid API credentials")
            elif err.errno == 404:
                raise ValueError("Error: The API endpoint is currently unavailable")
            else:
                raise ValueError("Error: {}".format(err))

        return response

    def get_data(self, station_name, start_date, end_date, data_format="json"):
        querystring = {"startDate": start_date, "endDate": end_date}
        url = self.time_series_url % station_name
        json_data = self.__get_request(url, querystring).json()
        if json_data['status']=='error':
            raise ValueError("Request has error %s"%json_data['error'])

        if data_format == "json":
            return json_data
        elif data_format == "dataframe":
            return json_to_df(json_data, group=None)


    def get_stations(self):
        station_list = self.__get_request(url=self.station_url)
        return station_list.json()


    def active_stations(self, station_list, active_day_range=datetime.now(tz.tzutc()), threshold=24):
        """
        Return active station from the given list of stations during the active day
        Args:
            station_list (list): station name list
            active_day_range (datetime):datetime to check the last reported date. Default current date in utc format.
                                    e.g. datetime.datetime.now(dateutil.tz.tzutc())
            threshold (int): threshold to decide the station as inactive in hours. Default 24 hours

        Returns:

        """
        if len(station_list) < 1:
            raise ValueError("The station list is empty")
        all_stations = self.get_stations()["stations"]
        active_stations = []

        for station in station_list:
            for stn_db in all_stations:
                if stn_db["id"] == station:
                    if not stn_db["active"]:
                        continue
                    last_measure = parser.parse(stn_db["lastMeasurement"])  # Need to change to utc.

                    wait_time = divmod((active_day_range - last_measure).total_seconds(), 3600)
                    if wait_time[0] < threshold:
                        active_stations.append(station)
                        continue
        return active_stations

    def daily_data(self, station_name, weather_variable, start_date, end_date):
        json_data = self.get_data(station_name,  start_date, end_date)
        if len(json_data["timeseries"])<1:
            return None
        elif len(json_data["timeseries"][weather_variable])<1:
            return None
        else:
            df = json_to_df(json_data, weather_variable=weather_variable, group='D')
            return df

    def load_nearby_stations(self, target_station, radius=None, k=None):
        load_station_distance = json.load(open(self.nearby_station_file,"r"))
        t_station = sorted(load_station_distance[target_station], key= itemgetter('distance'), reverse=False)

        if radius:
            t_station =[stn for stn in t_station if stn['distance']<radius]

        if k:
            return t_station[:k]
        return t_station

    def nearby_stations(self, target_station, k=10, radius=100):
        get_within_radius = self.load_nearby_stations(target_station, radius=radius, k=k)
        return [stn["site_to"] for stn in get_within_radius]

    def get_active_nearby(self, target_station, k=10, radius=100):
        all_k_stations = self.load_nearby_stations(target_station, k=k, radius=radius)
        station_list = [stn['site_to'] for stn in all_k_stations]
        return self.active_stations(station_list)

    def compute_nearest_stations(self):
        all_stations = self.get_stations()

        if all_stations["status"]!="success":
            raise ValueError("Looks there is a problem %s"%all_stations["error"])
        all_stations = all_stations["stations"]
        metadata = {}
        for ix, stn in enumerate(all_stations):
            loc = stn["location"]
            metadata[stn["id"]] = []
            for n_stn in all_stations:
                if n_stn==stn:
                    continue
                jloc = n_stn["location"]
                dist = haversine_distance(loc["lat"], loc['lng'], jloc["lat"], jloc["lng"])
                metadata[stn["id"]].append({"site_to":n_stn["id"], "distance":dist,"elevation":n_stn["elevation"]})
        json.dump(metadata, open("station_nearby.json","w"))

if __name__ == '__main__':
    target_station = "TA00055"
    thm = TahmoDataSource("station_nearby.json")
    start_date = datetime.strftime(datetime.utcnow() - timedelta(200), '%Y-%m-%dT%H:%M')
    end_date = datetime.strftime(datetime.utcnow() - timedelta(180), '%Y-%m-%dT%H:%M')
    print (start_date)
    # print thm.get_stations()
    #print thm.daily_data(target_station,weather_variable=RAIN, start_date=start_date, end_date=end_date)
    # print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)
    # get active stations
    station_list = ['TA00028', 'TA00068', "TA00108", "TA00187"]
    #thm.compute_nearest_stations()
    #print thm.load_nearby_stations(target_station, k=5)
    current_day = datetime.now(tz.tzutc())
    #print thm.active_stations(station_list, active_day_range=current_day)
    #print thm.nearby_stations(target_station=target_station, k=20, radius=200)
    #thm = TahmoDataSource()
    # print thm.get_stations()
    # print thm.get_data("TA00021", start_date="2017-09-01", end_date="2017-09-05")
    kdd = thm.load_nearby_stations("TA00025", radius=100)
    print (len(kdd), kdd)
    print (thm.active_stations([stn['site_to'] for stn in kdd]))
    #print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)

