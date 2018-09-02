from data_source import DataSource, json_to_df
import numpy as np
import pandas as pd
import json
def generate_toy(date_from, date_to, variable):
    rng = pd.date_range(date_from, date_to, freq='5Min')
    ts = pd.DataFrame({variable:np.random.exponential(5, len(rng)),'date':rng})
    return ts

class StationData:
    def __init__(self, raw_data, station_name):
        self.raw_data = raw_data
        self.station_name = station_name
    def to_df(self):
        pass
    def to_json(self):
        pass


class ToyDataSource(DataSource):
    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):

        return np.random.choice(ToyDataSource.station_list(), k)


    @staticmethod
    def station_list():
        stations = ['TA000'+str(i) for i in range(1, 20)]
        return stations

    @staticmethod
    def measurements(station_name, variable, date_from, date_to, **kwargs):
        df = generate_toy(date_from, date_to, variable)
        if kwargs.get('group'):
            df = df.groupby(df.date.dt.dayofyear).agg({variable: np.sum,
                                                  "date": np.max})
        return df

if __name__ == '__main__':
    xx = generate_toy('2017-09-10','2017-09-12','pr')
    jxx = pd.DataFrame(xx)
    #print jxx.head(5)
    #print ToyDataSource.nearby_stations('TA0001', k=5)
    print ToyDataSource.measurements('TA0001','pr','2017-09-10','2017-12-12', group='D')
