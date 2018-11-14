import math
import numpy as np
def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371.16
    deg2rad = lambda deg: deg*math.pi/180
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(deg2rad(lat1)) * \
    math.cos(deg2rad(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    d = earth_radius * c #// Distance in km
    return d


if __name__ == '__main__':
    print haversine_distance(38.898556, -77.037852, 15.897147, 17.043934)