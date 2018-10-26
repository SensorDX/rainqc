import pandas as pd
input_dir='/scratch/projects/sensordx/dump/'
import os



class Station(object):
    """Station entity"""

    def __init__(self, station_name, date_range=None, sensors=None, load=False):
        self.name = station_name
        self.date_range = date_range
        self.sensors = sensors
        self.qflag = False
        self.df = None
        if load:
            self.load_data(date_range, sensors)

    def nearby_stations(self, k=10, radius=100, load_stations=False):
        """
        Return k-nearest stations
        """
        site_code = self.name  # target_station
        stations = pd.read_csv("Nearest_station.dt")
        # print pd.unique(stations['siteFrom']).shape
        # print stations.head(5)
        k_nearest = stations[(stations['siteFrom'] == site_code) & (stations['distanceD'] < radius)]
        all_stations = [ff.split('.csv')[0] for ff in os.listdir(input_dir)]
        self.k_nearest = k_nearest[k_nearest['siteTo'].isin(all_stations)]
        self.k_nearest = self.k_nearest.sort_values(by=['distanceD', 'elevD'], ascending=True)[0:k]
        if not load_stations:
            return self.k_nearest
        all_stations = [{'station_name': row['siteTo'],
                         'station': Station(station_name=row['siteTo'], date_range=self.date_range,
                                            sensors=self.sensors,
                                            load=True),
                         'distance': row['distanceD'], 'elev': row['elevD']}
                        for indx, row in self.k_nearest.iterrows()]
        return all_stations

    def load_data(self, date_range=None, sensors=None):
        """ Load stations with sensor variables"""
        if date_range is not None:
            self.date_range = date_range
        if sensors is not None:
            self.sensors = sensors
        station_fname = input_dir + self.name + ".csv"
        df = pd.read_csv(station_fname)
        df.YYYYMMDDhhmm = pd.to_datetime(df.YYYYMMDDhhmm, format="%Y%m%d%H%M")
        df.index = df.YYYYMMDDhhmm
        if self.date_range is not None:
            df = df[self.date_range[0]:self.date_range[1]]
        if self.sensors is not None:
            df = df[self.sensors]
        self.df = df