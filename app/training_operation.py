""" Steps for training operations.
 Train batch of stations for training. The steps are.

check the database for which stations are trained for the last Y month.
 For each not trained stations:
 Push the station, date_From and date_to along with other relevant information to Queue,
 From the message queue, perform training operation of the stations one by one. And save the fitted model to the database with
 time of training, number of stations used ..etc

 Pop station from the MQ, then for each station, query its associated station for the given time period, and measure the amount of data available
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
from logging.handlers import RotatingFileHandler

from dateutil import parser
from pytz import utc
from src.common.utils import is_valid_date
from src.datasource import FakeTahmo, DataSource, TahmoDataSource, ModelDB
from definition import RAIN
import logging
import numpy as np
from src.rqc import MainRQC
from definition import ROOT_DIR
import json
import datetime
import os, pickle
from collections import deque

log_path = os.path.join(ROOT_DIR, "log/score_" + datetime.datetime.today().strftime("%Y-%m-%d") + ".log")
handler = RotatingFileHandler(log_path, maxBytes=10000, backupCount=1)
handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(handler)

app_config = json.load(open(ROOT_DIR + '/config/app.json', 'r'))
training_config = app_config['train']


def train_all(data_source, variable, start_date, end_date, config):
    all_station = data_source.online_stations()
    trained_station = {}
    for stn in all_station:
        trained, error = check_for_training(data_source, stn, variable, start_date, end_date, config)
        if trained:
            # if trained succesfully. save and move
            error['failed'] = False
            trained_station[stn] = error
        else:
            # Register the errors and continue to another stations.
            error['failed'] = True
            trained_station[stn] = error

    return trained_station


def is_valid_variables(data_source, target_station, variable, start_date, end_date):
    if data_source is None:
        raise ValueError("Data source is None")
    if target_station is None:
        raise ValueError("Target station is not given or None")
    if not is_valid_date(start_date, end_date):
        raise ValueError("Invalid date format")


def check_for_training(data_source, target_station, variable, start_date, end_date, config,
                       k_stations=None):
    error_message = {}
    try:
        is_valid_variables(data_source, target_station, variable, start_date, end_date)

        target_station_data = data_source.daily_data(target_station, variable, start_date, end_date)
        if target_station_data is None:
            raise ValueError("Target station has no data for the given date")
        if k_stations is None:

            k_stations = data_source.nearby_stations(target_station=target_station, radius=config["radius"])
        else:
            assert isinstance(k_stations, list)
            assert len(k_stations) > 0

        if len(k_stations) < 1:
            raise ValueError("The number of neighbor stations are few {}".format(len(k_stations)))

        nrows = target_station_data.shape[0]
        active_date = utc.localize(parser.parse(end_date))
        active_station = data_source.active_stations(k_stations, active_day_range=active_date)

        if len(active_station) < 1:
            raise ValueError("There is no active nearby stations to query data")

        k_stations_data = {}
        k = config["max_k"]  # Filter k station if #(active stations) > k
        for stn in active_station:
            if k < 1:
                break
            current_data = data_source.daily_data(stn, variable, start_date, end_date)
            if current_data is None:
                continue

            k_stations_data[stn] = current_data
            k -= 1

        if len(k_stations_data.keys()) < 1:
            raise ValueError(
                "All of the active station {} don't have data starting date {} to {}.".format(active_station,
                                                                                              start_date,
                                                                                              end_date))

        min_threshold_rows = np.ceil(nrows * config["FRACTION_ROWS"])
        usable_station = []
        for key, value in k_stations_data.items():
            if len(value) > min_threshold_rows:
                usable_station.append(key)
        if len(usable_station) < config["MIN_STATION"]:
            raise ValueError(
                "The number of usable station {} is lower than the minimum required {}".format(len(usable_station),
                                                                                               config['MIN_STATION']))
        error_message['message'] = "There are {} available stations to use".format(k_stations_data.keys())

        return True, error_message
    except ValueError as ex:
        error_message["message"] = ex.message
        print (ex.message)
        error_message["parameters"] = {'station': target_station, 'variable': variable,
                                       'startDate': start_date, 'endDate': end_date, 'config': config}
        logger.error(ex.message + ",{}, {},{},{}".format(target_station, variable, start_date, end_date))
        return False, error_message

    # return target_station_data, k_stations_data


# Scoring process for stations.

"""
 - At interval of time, pull station from pool of active stations with available trained model.
  - Retrieve all its metadata and associated data for it asynchronously.
  - If success:
     - saved the score of the station and write log as success withe timestamp of work.
  - if error:
     - If all necessary data didn't get pulled, put it in waiting list and write reason to log.
     - If it don't have trained model, write erro to log and move to the next stations. Put the or mark the station
     with no fitted data.
     - Next station
 - fill the pool based on active stations and availability of stations.

"""


def score_it(start_date, end_date, model_config):  # model_config, training_config):

    rqc_pk = pickle.loads(model_config['model'])
    rqc = MainRQC.load(rqc_pk)
    scores = rqc.score(start_date, end_date)
    return scores



def score_operation(data_source, start_date, end_date=None, weather_variable=RAIN):
    assert isinstance(data_source, DataSource)
    active_stations = data_source.online_stations()  # active_day_range=start_date)
    logger.info("Score operation from {} to {} of variable {}".format(start_date, end_date, weather_variable))


    all_fitted_stations = data_source.fitted_stations()
    fitted_stations = [stn['station'] for stn in all_fitted_stations]

    if len(fitted_stations) < 1:
        raise ValueError('There is no fitted model in the database')

    active_fitted_stations = set(fitted_stations).intersection(active_stations)
    non_score_stations = set(active_fitted_stations).difference(fitted_stations)
    logger.info("fitted stations, {}, active fitted station {}".format(fitted_stations, active_fitted_stations))

    # put the non_score_stations on the pool non-scored stations..& check their date for their interval next time.
    logger.info("Active fitted stations {} -- non-score stations {}".format(active_fitted_stations, non_score_stations))

    station_status = {"start_date": start_date, "end_date": end_date, "failure": [], "not_fitted": list(non_score_stations),
                      "success": []}
    last_failed_log = data_source.last_failure_pool()

    stations_q = deque(active_fitted_stations)

    while len(stations_q)>0:

        station = stations_q.popleft()
        model_config = data_source.get_fitted_station(station)
        left_over_operation = False

        if model_config is None:
            return "Station not yet fitted"

        fitted, error = check_for_training(data_source=data_source, target_station=station,
                                           variable=weather_variable,
                                           start_date=start_date, end_date=end_date, config=training_config,
                                           k_stations=model_config['k_stations'])
        if not fitted:
            logger.error(" Model for station {} not available".format(station))
            continue
         # Check if the station is in the failure mode from last run.
        if station in last_failed_log['failure'] and not left_over_operation:
            start_date = last_failed_log['start']
            left_over_operation = True

        score_result = score_it(start_date, end_date, model_config)
        logger.info("Scoring for station {}".format(station))

        if len(score_result) < 1:

            # Write failure log to db.
            # non_score_stations.add(station)
            if left_over_operation:
                stations_q.appendleft(station)
                continue
            station_status['failure'].append(station)
            logger.warning("Station {} can't be scored due to error".format(station))
        else:
            # save score of the station.
            data_source.save_scores(score_result, model_config, start_date, end_date)
            station_status['success'].append(station)
            logger.info("Station {} scored with success".format(station))

            # Write success log to db.
    # Write non-trained stations:

    data_source.save_score_pool(station_status)
    return station_status


def mongo_config():
    from bson.binary import Binary
    from pymongo import MongoClient
    import os
    mode = "dev"
    db_config = json.load(open(os.path.join(ROOT_DIR, "config/config.json"), "r"))["mongodb"]
    mongo_uri = db_config[mode]  # could be
    mongo = MongoClient(mongo_uri)
    db = mongo['rainqc']
    # print([x for x in db.model.find({},{'station':1})])
    return db


if __name__ == '__main__':
    mongo_conn = mongo_config()
    target_station, variable = "TA00021", RAIN
    start_date = "2016-04-01"
    end_date = "2016-12-31"

    config = {"radius": 100, "max_k": 5, "FRACTION_ROWS": 0.5, "MIN_STATION": 3}
    fk = FakeTahmo()
    mongo_db = ModelDB(mongo_conn)
    fk.set_modeldb(mongo_db)
    # fitted = fk.fitted_stations(query={},selector={'station':1, '_id':0})
    # for kk in fitted:
    #     print(kk)

    # print (fk.stations())
    # print (check_for_training(fk, None, variable, start_date, end_date, config))
    # ## Check all training operations
    # training_error_metric = train_all(fk, variable, start_date, end_date, config)
    # for stn, value in training_error_metric.items():
    #     print (stn, value)

    rs = score_operation(fk, start_date, end_date)
    print(rs)
    #mongo_db.delete(collection_name="score")
## TODO: Add decision threshold function based on the score of the synthetic faults.
## TODO: Add the datasource for the influxdb with the datasource from the real-data.
## TODO: Derive the ll for the













































































































































































































































































































































































































































































































































































































































































