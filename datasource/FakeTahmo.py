import pandas as pd
import numpy as np
from common.weather_variable import RAIN


class FakeTahmo(object):

    def __init__(self, local_data_source="../experiments/dataset/tahmostation2016.csv",
                 nearby_station="../localdatasource/nearest_stations.csv"):
        self.data = pd.read_csv(local_data_source)
        self.nearby_station_file = nearby_station
        date_range = pd.date_range(start='2016-01-01', end="2016-12-31",freq='1D')
        self.data.index = date_range
        #self.data.rename(columns={0:RAIN}, inplace=True)
    def stations(self):
        stations = self.data.columns.tolist()
        return stations
    def daily_data(self, target_station, target_variable, start_date, end_date):
        ###
        if target_station is None:
            raise ValueError("Station is empty")
        station_data = self.data[target_station][start_date:end_date].values
        return station_data.reshape(-1,1)
    def get_data(self):
        pass

    def nearby_stations(self, target_station, k=None, radius=100):

        stations = pd.read_csv(self.nearby_station_file)  # Pre-computed value.
        k_nearest = stations[(stations['from'] == target_station) & (stations['distance'] < radius)]
        k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)['to'].tolist()  # [0:k]
        if k is not None:
            return k_nearest[:k]
        return k_nearest
    def active_stations(self, station_list, active_day_range="2016-01-01"):
        return station_list
if __name__ == '__main__':
    ft = FakeTahmo()
    print (ft.stations())
    print (ft.nearby_stations('TA00030'))
    #print (ft.data.head(5))
    dayr = ft.daily_data('TA00030',"pr",'2016-02-02','2016-05-01')



