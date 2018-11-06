
## Qeury data and build _views based on the data.

import numpy as np
import pandas as pd


def json_to_df(json_station_data, weather_variable='pr', group='D', filter_year=2017):
    rows = [{'date': row['date'], weather_variable: row['measurements'][weather_variable]} for row in json_station_data]
    df = pd.DataFrame(rows)
    df.date = pd.to_datetime(df.date)
    if filter_year:
        df = df[df.date.dt.year == filter_year]
    df_formatted = df.groupby(df.date.dt.dayofyear).agg({weather_variable :np.sum, "date" :np.max})  # apply(lambda x: np.sum(x[weather_variable]))  ## take max readings of the hour.
    return df_formatted


    # def pairwise_view(station1, station2, )
