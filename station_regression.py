import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.base import BaseEstimator
from fancyimpute import MICE

input_dir = "/nfs/guille/bugid/adams/ifTadesse/tahmoqcsensordx/odm/hourly"
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


class StationData:

    def __init__(self, station_name, response=['RAIN'], variables= ['SRAD', 'TAIR', 'RELH', 'WDIR', 'WSPD', 'PRES'],
                 date_time=['rDate','hour']):
        self.station_name = station_name
        self.variables = variables
        self.response = response
        self.date_time_column = date_time


    def make_data(self, year=None, month=None, save=False):#, response=['RAIN'],
                  #variable=['SRAD', 'TAIR', 'RELH', 'WDIR', 'WSPD', 'PRES'], date_time_column=['rDate','hour']):
        """
        Create the matrix for training the regression.
        :return:
        """
        df = self.load_data(station_name=self.station_name, year=year, month=month,
                            variables=self.date_time_column + self.response + self.variables)  # Load and fill missing values. return numpy of response variables.
        date_time = df.ix[:,self.date_time_column]
        yX = impute_value(df.ix[:, self.response + self.variables ].as_matrix(), na_value=-60)
        X, y  = yX[:,1:], yX[:,0]

        # Nearest stations
        k_station = self.nearby_stations(radius=50, k =5)
        for stid in k_station:

            dfs = self.load_data(station_name=stid, year=year, month=month,
                                 variables=self.date_time_column + self.response + self.variables)
            #print dfs.shape[0]
            #print X.shape
            #date_time = dfs.ix[:, date_time_column]
            yX = impute_value(dfs.ix[:, self.response + self.variables].as_matrix(), na_value=-60)
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
            print column
            tempdf = pd.concat([date_time,pd.DataFrame(y),pd.DataFrame(X)],axis=1)
            tempdf.columns = column # = tempdf.rename(column=)
            tempdf.to_csv(os.path.join(preprocessed_output,"pre-"+self.station_name+ "-" +str(year)+"-.csv"), index=False)



        return y, X



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



class Regressor(BaseEstimator):
    def __init__(self):
        self.clf = LinearRegression()
    def fit(self, X,y):
        self.clf.fit(X,y)
    def predict(self, X):
        return self.clf.predict(X)






if __name__ == '__main__':
    stn = StationData('CAMA')
    y,x = stn.make_data(year=2008, save=True)
    print y.shape, x.shape
    #print stn.nearby_stations(radius=50)
    #print stn.load_data().head(5)
    #print stn.fill_na(stn.load_data(),True)

    #dx, mtx = preprocess_station('CAMA.csv')





