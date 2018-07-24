import pandas as pd 
import numpy as np 
from util.stations_tools import nearby_stations

class DataSource:

    def __init__(self, src):
        self.tahmo_db = pd.read_csv(src)

    def extract_features(self, target_station, k=5, year=2016):

        """
        Extract percipitation from k nearby stations.
        """
        filter_stn = lambda stn_name, year: self.tahmo_db[(self.tahmo_db.station == stn_name) & (self.tahmo_db.year == year)]
        t_station = filter_stn(target_station, year)
        k_station = nearby_stations(site_code=target_station, k=k, radius=300).to.tolist()
        X = t_station['tahmo'].as_matrix().reshape([-1, 1])
        datetime = t_station.datetime.tolist()

        # print datetime.shape
        for stt in k_station:
            R = filter_stn(stt, year)[['tahmo']].as_matrix().reshape([-1, 1])
            X = np.hstack([X, R])

        df = pd.concat([pd.DataFrame(datetime), pd.DataFrame(X)], axis=1)
        df.columns = ["datetime", target_station] + k_station

        return df
