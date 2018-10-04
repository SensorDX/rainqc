import numpy as np
import pandas as pd

#from fancyimpute import MICE

def haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371.16
    deg2rad = lambda deg: deg*np.pi/180
    dlat = deg2rad(lat2 - lat1)
    dlon = deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2) * np.sin(dlat / 2) + np.cos(deg2rad(lat1)) * \
    np.cos(deg2rad(lat2)) * np.sin(dlon / 2) * np.sin(dlon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    d = earth_radius * c #// Distance in km
    return d

# def impute_value(X, na_value=-60):
#     """
#     Accept X value of numpy array.
#     Remove large < -50 values from the array
#     remplace with np.nan and impute the values using mice.
#     improve to only replace the cells instead of the whole rows.
#     :param X: ndarray
#     :return:
#     """
#     miss_index = np.unique(np.where(X<na_value)[0])
#     if len(miss_index)>0:
#         X[miss_index,:] = np.nan
#         mice = MICE(verbose=False)
#
#         return mice.complete(X)
#     return X

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
def rain_max(series):
    if any(series<0.0):
        return 0.0
    else:
        return np.max(series)
def group_station_data(station_name, group_type='H', apply_fn=np.mean, year=None, month=None, variables=None):
    """

    :param station_name:
    :param group_type:
    :param apply_fn: np.fun type summarization function.
    :param year: int  year
    :param month: int month number
    :param variables: variables to

    :return:
    """
    df = pd.read_csv(station_name)

    df['hour'] = df['rTime']//100
    f = merge_two_dicts({col:apply_fn for col in df.columns[2:28:2]},
                        {flag:np.max for flag in df.columns[3:29:2]})
    f['RAIN'] = rain_max
    if group_type=='H':
        grouped = df.groupby(['rDate','hour']).agg(f)
    elif group_type=='D':
        grouped = df.groupby(['rDate']).agg(f)
    else:
        return "The group is not yet defined."
    #print grouped.head(5)
    # Filter the grouped data.
    grouped['rDate'] =pd.to_datetime(grouped.index)
    #grouped.rDate = pd.to_datetime(grouped.rDate)
    if year is not None:
        grouped = grouped[grouped.rDate.dt.year == year]
    if month is not None:
        grouped = grouped[grouped.rDate.dt.month == month]

    if variables is not None:
        grouped = grouped.ix[:, variables]



    return grouped

if __name__ == '__main__':
    import os
    pth = "/media/tadeze/kofo/research/Temp/dump"
    files = os.listdir(pth)
    gg = group_station_data(os.path.join(pth,files[0]),group_type='D',variables=['RAIN'],
                            year=2008)
    df = pd.DataFrame()
    for ffile in files:
        colname = ffile.split('.')[0]
        gg = group_station_data(os.path.join(pth, ffile), group_type='D', variables=['RAIN'],
                                year=2008)
        gg.columns =[colname]
        if gg.shape[0]==366:
            df = pd.concat([df,gg],axis=1)
    print df.shape
    df.to_csv("mesonet_2008.csv",index=False)