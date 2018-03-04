import numpy as np
import pandas as pd

input_dir='/scratch/projects/sensordx/dump/'
output_dir = "~/adams/tahmoqcsensordx/odm/hourly"

#def impute_miss(df):

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def group_hourly(station_name, group_type='H'):
    df = pd.read_csv(station_name)
    df['hour'] = df['rTime']//100
    f = merge_two_dicts({col:np.mean for col in df.columns[2:28:2]},
                        {flag:np.max for flag in df.columns[3:29:2]})
    f['RAIN'] = np.sum
    if group_type=='H':
        grouped = df.groupby(['rDate','hour']).agg(f)
    elif group_type=='D':
        grouped = df.groupby(['rDate']).agg(f)
    else:
        return "The group is not yet defined."
    # Remove or impute the missing values.
    grouped.index = range(0, grouped.shape[0])
    ix = grouped[grouped< -400].stack().index().tolist()
    grouped[ix] = np.nan
    return grouped


if __name__ == '__main__':
    import os
    for stn in os.listdir(input_dir):
        station_name = os.path.join(input_dir,stn)
        hdata= group_hourly(station_name, 'H')
        pd.DataFrame.to_csv(hdata,os.path.join(output_dir,stn))
        print "Saved {0:s}".format(stn)
    #print hdata.shape
    #print hdata.head(5)