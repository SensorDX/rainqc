from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
from definition import ROOT_DIR
from .abcdatasource import DataSource
import os



class FakeTahmo(DataSource):

    def get_data(self, station_name, start_date, end_date, data_format="json"):
        pass

    def __init__(self, local_data_source="localdatasource/dataset/tahmostation2016.csv",
                 nearby_station="localdatasource/nearest_stations.csv"):
        super(FakeTahmo, self).__init__()
        self.local_data_source = os.path.join(ROOT_DIR, local_data_source)
        self.data = pd.read_csv(self.local_data_source)
        self.nearby_station_file = os.path.join(ROOT_DIR, nearby_station)
        date_range = pd.date_range(start='2016-01-01', end="2016-12-31", freq='1D')
        self.data.index = date_range

        # self.data.rename(columns={0:RAIN}, inplace=True)

    def stations(self):
        stations = self.data.columns.tolist()
        return [{'online':True, 'id': stn, 'active':True} for stn in stations]
    def exists(self, station):
        return station in self.data.columns.tolist()

    def daily_data(self, target_station, target_variable, start_date, end_date):
        ###

        if target_station is None:
            raise ValueError("Station {} is empty or None".format(target_station))
        if not self.exists(target_station):
            raise ValueError('Station {} doesn\'t exist'.format(target_station))

        station_data = self.data[target_station][start_date:end_date].values

        if station_data.shape[0]<1:
            raise ValueError('Station {} doesn\'t have data'.format(target_station))
        return station_data.reshape(-1, 1)

    def nearby_stations(self, target_station, k=None, radius=100):

        stations = pd.read_csv(self.nearby_station_file)  # Pre-computed value.
        k_nearest = stations[(stations['from'] == target_station) & (stations['distance'] < radius)]
        k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)['to'].tolist()  # [0:k]
        if k is not None:
            return k_nearest[:k]
        return k_nearest

    def active_stations(self, station_list, active_day_range="2016-01-01"):
        return station_list

    def online_station(self, threshold=72):
        return self.data.columns.tolist()

    def to_json(self):
        json_config = {"local_data_source": self.local_data_source, "nearby_station_file": self.nearby_station_file}
        return json_config

    @classmethod
    def from_json(cls, json_config):
        fake_tahmo = FakeTahmo(local_data_source=json_config['local_data_source'],
                               nearby_station=json_config['nearby_station_file'])
        return fake_tahmo


if __name__ == '__main__':
    ft = FakeTahmo()
    print (ft.stations())
    print (ft.nearby_stations('TA00030'))
    # print (ft.data.head(5))
    dayr = ft.daily_data('TA00030', "pr", '2016-02-02', '2016-05-01')
    print (ft.to_json())
