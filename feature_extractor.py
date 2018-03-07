import numpy as np
import pandas as pd
import os
from fancyimpute import MICE

input_dir = "/nfs/guille/bugid/adams/ifTadesse/tahmoqcsensordx/odm/hourly"
input_dir_dump='/scratch/projects/sensordx/dump/'
preprocessed_output = "preprocessed"
def impute_value(X, na_value=-60):
    """
    Accept X value of numpy array.
    Remove large < -50 values from the array
    remplace with np.nan and impute the values using mice.
    improve to only replace the cells instead of the whole rows.
    :param X: ndarray
    :return:
    """
    miss_index = np.unique(np.where(X<na_value)[0])
    if len(miss_index)>0:
        X[miss_index,:] = np.nan
        mice = MICE(verbose=False)
        return mice.complete(X)
    return X

def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z

def group_station_data(station_name, group_type='H', apply_fn=np.mean, year=None, month=None, variables=None):
    """

    :param station_name:
    :param group_type:
    :param apply_fn: np.fun type summarization function.
    :param year: int  year
    :param month: int month number
    :param variables: variables to include
    :return:
    """
    df = pd.read_csv(station_name)

    df['hour'] = df['rTime']//100
    f = merge_two_dicts({col:apply_fn for col in df.columns[2:28:2]},
                        {flag:np.max for flag in df.columns[3:29:2]})
    f['RAIN'] = np.sum
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


class StationData:

    def __init__(self, station_name, response=['RAIN'], variables= ['SRAD', 'TAIR', 'RELH', 'WDIR', 'WSPD', 'PRES'],
                 date_time=['rDate','hour'], output_prefix="daily"):
        self.station_name = station_name
        self.variables = variables
        self.response = response
        self.date_time_column = date_time
        self.k = 5 # k-nearest neigbhors
        self.radius = 50 # default radius
        self.na_impute_value = -100
        self.output_prefix = output_prefix


    def make_data(self, year=None, month=None, save=False):#, response=['RAIN'],
                  #variable=['SRAD', 'TAIR', 'RELH', 'WDIR', 'WSPD', 'PRES'], date_time_column=['rDate','hour']):
        """
        Create the matrix for training the regression.
        :return:
        """
        df = self.load_data(station_name=self.station_name, year=year, month=month,
                            variables=self.date_time_column + self.response + self.variables)  # Load and fill missing values. return numpy of response variables.
        date_time = df.ix[:,self.date_time_column]
        yX = impute_value(df.ix[:, self.response + self.variables ].as_matrix(), na_value= self.na_impute_value)
        X, y  = yX[:,1:], yX[:,0]

        # Nearest stations
        k_station = self.nearby_stations(radius=self.radius, k =self.k)
        for stid in k_station:

            dfs = self.load_data(station_name=stid, year=year, month=month,
                                 variables=self.date_time_column + self.response + self.variables)
            #print dfs.shape[0]
            #print X.shape
            #date_time = dfs.ix[:, date_time_column]
            yX = impute_value(dfs.ix[:, self.response + self.variables].as_matrix(), na_value= self.na_impute_value)
            R = yX[:,0]
            R = R.reshape([R.shape[0],1])
            #print R.shape
            X = np.hstack([X,R])

            # Load the data
            # get the respone variable from each station & concatnate.
            print stid
        if save:
            # If save the data.
            column = self.date_time_column + self.response + self.variables + [self.response[0]+'_'+ stid for stid in k_station]
            #print column
            tempdf = pd.concat([date_time,pd.DataFrame(y),pd.DataFrame(X)],axis=1)
            tempdf.columns = column # = tempdf.rename(column=)
            tempdf.to_csv(os.path.join(preprocessed_output,"pre-"+self.station_name+ "-" +str(year)+"-.csv"), index=False)



        return y, X


    def extract_features(self, apply_fn = np.var, grouping='D', year=None, month=None, save=False):
        """
        Extract features from neighboring of 5 stations, using daily variance of the predictor variables.
        Group data using daily/hourly intervals.  compute group statistics using summarization_fn.
        Append all predictor variables to original dataset.
        :param station_name: target station name
        :param apply_fn: apply function for summarization.
        :return: return X: numpy array of feature matrix.
        """
        station_full_path = lambda  station_name : os.path.join(input_dir_dump, station_name+".csv")
        target_station = group_station_data(station_full_path(self.station_name), group_type=grouping, apply_fn=apply_fn, year=year, month=month)

        date_time = target_station.index #[:, self.date_time_column[0]]
        #print date_time[1:10]
        yX = impute_value(target_station.ix[:, self.response + self.variables].as_matrix(), na_value = self.na_impute_value)
        X, y = yX[:, 1:], yX[:, 0]
        k_nearest = self.nearby_stations(k=self.k)

        label = self.response + self.variables + [stn+"_"+predictor for stn in k_nearest for predictor in self.response + self.variables ]
        for k_stn in k_nearest:
            k_std_df = group_station_data(station_full_path(k_stn), group_type=grouping, apply_fn=apply_fn, year=year,month=month) #self.load_data(station_name=stid, year=year, month=month,
            yX = impute_value(k_std_df.ix[:, self.response + self.variables].as_matrix(), na_value=self.na_impute_value)
            #R = yX[:, :]
            #R = R.reshape([R.shape[0], 1])
            # print R.shape
            X = np.hstack([X, yX])
        if save:
            # If save the data.
            column = [self.date_time_column[0]] + label
            #print column
            tempdf = pd.concat([pd.DataFrame(date_time), pd.DataFrame(y), pd.DataFrame(X)], axis=1)
            tempdf.columns = column  # = tempdf.rename(column=)
            tempdf.to_csv(os.path.join(preprocessed_output, self.output_prefix + self.station_name + "-" + str(year) + "-.csv"),
                          index=False)
            return tempdf

        return X



    def load_data(self, station_name=None, year=None, month=None, variables=None):
        """
         Load data
        :param file_name:
        :return:
        """
        if station_name is None:
            station_name = self.station_name

        file_name = station_name+".csv"
        df = pd.read_csv(os.path.join(input_dir,file_name))
        df.rDate = pd.to_datetime(df.rDate)
        if year is not None:
            df = df[df.rDate.dt.year==year]
        if month is not None:
            df = df[df.rDate.dt.month==month]

        if variables is not None:
            df = df.ix[:,variables]
        return df
    def fill_na(self, df, to_df=False):
        date_time = df.ix[:,:2]
        X = df.ix[:,2:].as_matrix()
        X = impute_value(X)
        if to_df:
            return pd.concat([date_time, pd.DataFrame(X)],axis=1)
        return X, date_time


    def nearby_stations(self, k=100, radius=100, load_stations=False):
        """
        Return k-nearest stations
        """
        site_code = self.station_name  # target_station
        stations = pd.read_csv("Nearest_station.dt")
        k_nearest = stations[(stations['siteFrom'] == site_code) & (stations['distanceD'] < radius)]
        all_stations = [ff.split('.csv')[0] for ff in os.listdir(input_dir)]
        self.k_nearest = k_nearest[k_nearest['siteTo'].isin(all_stations)]
        self.k_nearest = self.k_nearest.sort_values(by=['distanceD', 'elevD'], ascending=True)[:k]
        return self.k_nearest.siteTo.tolist()



if __name__ == '__main__':

    # List of target stations
    target_stations = ['CLAY', 'OKCE', 'PRYO']
    obs_year = 2008
    response = ['RAIN']
    variables = ['TAIR', 'RELH', 'SRAD', 'WSPD', 'PRES']

    for t_station in target_stations:
        stn = StationData(t_station, variables=variables, response=response)
        X = stn.extract_features(year=obs_year, save=True)
        print "{0:s} extracted".format(t_station)


    #print X.shape
    #print lbl, len(lbl)
    #y,x = stn.make_data(year=2008, save=True)
    #print y.shape, x.shape
    #sss = group_station_data(os.path.join(input_dir_dump,'CAMA.csv'),'D',apply_fn=np.var, year=2009)
    #print sss.shape
    #print sss.tail(5)






