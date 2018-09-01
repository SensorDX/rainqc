

## Qeury data and build views based on the data.

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


## steps for building the system.
## 1. Check availability of the stations,
## 2. Build views:
##  - pairwise view maker,,,
## - Query data and transform to array formats.


def query_transform(data_source, **kwargs):
     """
     Query data from data source
     Args:
         data_source:
         **kwargs:

     Returns:

     """
