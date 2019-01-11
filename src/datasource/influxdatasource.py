from abcdatasource import DataSource
import os
import json
from definition import ROOT_DIR


class InfluxDataSource(DataSource):

    def __init__(self, keepdim=True):
        super(InfluxDataSource, self).__init__()
   # url
    # https://tahmoapibc.eu-gb.mybluemix.net/v1/timeseries/TA00390/rawmeasurements?startDate=2018-01-01&endDate=2018-06-28&variable=pr
        # Later will move to config.
        config_path = os.path.join(ROOT_DIR, 'config/config.json')
        if not os.path.exists(config_path):
            raise ValueError("{} doesn't exist".format(config_path))
        config = json.load(open(config_path, 'r'))
        tahmo_connection = config['tahmo']
        self.header = tahmo_connection['header']
        self.timeseries_url = tahmo_connection['timeseries']
        self.cm_url = tahmo_connection["cm_url"]
        self.nearby_station_file = os.path.join(ROOT_DIR, "config/" + tahmo_connection["nearby_station"])
        self.station_url = tahmo_connection["station_url"]
        self.keepdim = keepdim
        if not os.path.exists(self.nearby_station_file):
            self.compute_nearest_stations()


    def stations(self):
        pass

    def get_data(self, station_name, start_date, end_date, data_format="json"):
        pass

    def daily_data(self, station_name, weather_variable, start_date, end_date):
        pass

    def nearby_stations(self, target_station, k, radius):
        pass

    def active_stations(self, station_list, active_day_range):
        pass

    def to_json(self):
        return super(InfluxDataSource, self).to_json()

    @classmethod
    def from_json(cls, json_config):
        return super(InfluxDataSource, cls).from_json(json_config)