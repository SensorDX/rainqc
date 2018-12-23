""" Steps for training operations.
 Train batch of stations for training. The steps are.

check the database for which stations are trained for the last Y month.
 For each not trained stations:
 Push the station, date_From and date_to along with other relevant information to Queue,
 From the message queue, perform training operation of the stations one by one. And save the fitted model to the database with
 time of training, number of stations used ..etc

 Pop station from the MQ, then for each station, query it associated station for the given time period, and measure the amount of data available
 - Pre-assess data from all stations nearby to it, and decide whether there are enough data from the station to perform the operations.
 - if the data from the station makes sense and there are enough stations go-for training and save the model. If form some reason the training
 operation failed. Make the last trained model as active. Otherwise save the trained model and set it active for scoring operations.
 Continue this operations for all station periodically.

**Requirements**

 - There should be an available API from the bluemix to query.



 ## Scoring operations:
  - Check if there is any saved model before scoring operation, else return error or message that there is no saved models for
  the given station.
  - Load model from the stations.
  - Load data from the nearby stations/Mostly daily data. If some of them don't have try to impute the value.

"""



# 1. Test operation for checking data availability.
# 2.
from dateutil import parser
from pytz import utc
from src.common.utils import is_valid_date
from src.datasource.fake_tahmo import FakeTahmo
from definition import RAIN
import logging

logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
logging.basicConfig(filename="logfile.log",
                    level=logging.DEBUG, format=logger_format,
                    filemode='w')  # use filemode='a' for APPEND
logger = logging.getLogger(__name__)

def train_all(data_source, variable, start_date, end_date, config):
    ## Train all station anynchronously starting
    ## Check if all can be trained.
    all_station = data_source.online_stations()
    trained_station = []
    for stn in all_station:
         trained, error= check_for_training(data_source, stn, variable, start_date, end_date, config)
         if trained:
             # if trained succesfully. save and move
             pass
         else:
             # Register the errors and continue to another stations.
        # sleep for sometime.
    # print statistics once done.



def check_for_training(data_source, target_station, variable, start_date, end_date, config):
    error_message = {}
    try:

        if target_station is None:
            raise ValueError("Target station is not given or None")
            #return False, error_message
        # Check the date format.
        if not is_valid_date(start_date, end_date):
            raise ValueError("Invalid date format")
            #return False, error_message
        k_stations_data = {}
        target_station_data = data_source.daily_data(target_station, variable, start_date, end_date)
        if target_station_data is None:
            raise ValueError("Target station has no data for the given date")

        k_stations = data_source.nearby_stations(target_station=target_station, radius=config["radius"])
        if len(k_stations) < 1:
            raise ValueError("The number of neighbor stations are few {}".format(len(k_stations)))
            #return False, error_message

        nrows = target_station_data.shape[0]
        active_date = utc.localize(parser.parse(end_date))
        active_station = data_source.active_stations(k_stations, active_day_range=active_date)

        if len(active_station) < 1:
            raise ValueError("There is no active station to query data")
            #return False, error_message

        k = config["MAX_K"]  # Filter k station if #(active stations) > k
        for stn in active_station:
            if k < 1:
                break
            current_data = data_source.daily_data(stn, variable, start_date, end_date)
            if current_data is None:
                continue
            # if current_data.shape[0] <= nrows:

            k_stations_data[stn] = current_data
            k -= 1

        if len(k_stations_data.keys()) < 1:
            raise ValueError( "All of the active station {} don't have data starting date {} to {}.".format(active_station, start_date,
                                                                                              end_date))
            #return False, error_message

        min_threshold_rows = nrows * config["FRACTION_ROWS"]
        usable_station = []
        for key, value in k_stations_data.items():
            if len(value) > min_threshold_rows:
                usable_station.append(key)
        if len(usable_station) < config["MIN_STATION"]:
            raise ValueError("The number of usable station {} is lower than the minimum required {}".format(len(usable_station),
                                                                                                            config['MIN_STATION']))


        error_message['message'] = "There are {} available stations to use".format(k_stations_data.keys())

        return True, error_message
    except ValueError as ex:
        error_message["message"] = ex.message
        print (ex.message)
        error_message["parameters"] = {'station':target_station, 'variable':variable,
                                       'startDate':start_date, 'endDate':end_date, 'config':config}
        logger.error(ex.message + ",{}, {},{},{}".format(target_station,variable,start_date, end_date))
        return False, error_message


    # return target_station_data, k_stations_data

if __name__ == '__main__':
    target_station, variable = "TA00021", RAIN
    start_date = "2016-01-01"
    end_date = "2016-12-31"

    config = {"radius":100, "MAX_K":5, "FRACTION_ROWS":0.5, "MIN_STATION":2}
    fk = FakeTahmo()
    print (fk.stations())
    print (check_for_training(fk, None,variable, start_date, end_date, config))