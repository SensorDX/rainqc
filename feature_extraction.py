import pandas as pd 
import numpy as np 

def nearby_stations(site_code, k=10, radius=500):
    """
    Return k-nearest stations, given a station name.
    args:
        site_code: station site code.
        k: number of stations. 
        radius: radius of distance in KM
    """
    stations = pd.read_csv("nearest_stations.csv")  # Pre-computed value. 
    k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]
    k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)[0:k]

    return k_nearest


def k_nearby_features(target_station, k=5, year=2016):
    """
    Extract percipitation from k nearby stations. 
    """
    filter_stn = lambda stn_name, year: tahmo_rain[(tahmo_rain.station == stn_name) & (tahmo_rain.year == year)]
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