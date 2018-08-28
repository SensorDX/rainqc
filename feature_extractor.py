import pandas as pd
import numpy as np
from util.stations_tools import nearby_stations
from services.data_source import LocalDataSource as LDS
import json


def json_to_df(json_station_data, weather_variable='pr', group='D', filter_year=2017):
    rows = [{'date': row['date'], weather_variable: row['measurements'][weather_variable]} for row in json_station_data]
    df = pd.DataFrame(rows)
    df.date = pd.to_datetime(df.date)
    if filter_year:
        df = df[df.date.dt.year == filter_year]
    df_formatted = df.groupby(df.date.dt.dayofyear).agg({weather_variable:np.sum, "date":np.max}) #apply(lambda x: np.sum(x[weather_variable]))  ## take max readings of the hour.
    return df_formatted


class PairwiseView:
    def __init__(self, data_source=None, radius=300, num_k_station=5, variable="pr"):
        self.X = None
        self.y = None
        self.label = None
        self.radius =radius
        self.num_k_station = num_k_station
        self.variable = variable
        self.data_source = data_source

    def make_view(self, target_station, date_from, date_to):
        """

        Args:
            target_station: (str) target station code.
            date_range: (list) of date range from, to period.

        Returns:

        """
        t_station = self.data_source.measurements(target_station, self.variable, date_from=date_from, date_to=date_to,
                                                  group='D')

        self.y, self.label = t_station[self.variable].values.reshape(-1, 1), t_station['date']  # .values.reshape(-1, 1)

        k_nearest_station = self.data_source.nearby_stations(target_station, self.num_k_station, self.radius)
        print k_nearest_station
        stn_list = []
        for k_station in k_nearest_station:
             k_station_data = self.data_source.measurements(k_station, self.variable, date_from=date_from, date_to=date_to,
                                                  group='D')
             k_station_data = k_station_data[self.variable].values.reshape(-1,1)

             if k_station_data.shape[0] == self.y.shape[0]:
                stn_list.append(k_station_data)
        if len(stn_list) < 2:
            return NameError("There are less than 2 stations with equal number of observation as the target station.")
        self.X = np.hstack(stn_list)
        df = pd.concat([pd.DataFrame(self.label), pd.DataFrame(self.y), pd.DataFrame(self.X)], axis=1)
        pd.DataFrame.dropna(df, inplace=True)
        self.label, self.y, self.X = df.iloc[:,0], df.iloc[:,1], df.iloc[:,2:]

    def to_csv(self):
        df = pd.concat([pd.DataFrame(self.label), pd.DataFrame(self.y), pd.DataFrame(self.X)], axis=1)
        return df
##TODO: Pairwise view need to be fixed, with $n$ nearby station, there should be $n$ models
