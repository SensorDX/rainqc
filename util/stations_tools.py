## compute nearest station from a given
import pandas as pd


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


