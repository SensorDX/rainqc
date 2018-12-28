
import os,sys ,shutil, pickle
import time
from datetime import datetime
import pandas as pd
import argparse
import traceback
from src import MainRQC
from flask import Flask, request, jsonify, render_template, redirect
from definition import RAIN
import logging
from logging.handlers import RotatingFileHandler
from src.datasource import FakeTahmo, TahmoDataSource, TahmoAPILocal, evaluate_groups, DataSourceFactory
import numpy as np
from dateutil import parser
from pytz import utc
from bson.binary import Binary
from flask_pymongo import PyMongo,ObjectId
from definition import ROOT_DIR
import json
from graph import build_graph, out_plot, build_metric_graph
from training_operation import check_for_training
app = Flask(__name__)

# Parameters
RADIUS = 100
MAX_K = 5
WEATHER_VARIABLE = RAIN
VIEWS = ['PairwiseView']
MODELS = ['MixLinearModel']
VERSION = 0.1
WAIT_TIME_THRESHOLD = 72
training_config = {"radius": RADIUS, "MAX_K": MAX_K, "FRACTION_ROWS": 0.5,
                   "MIN_STATION": 2}  # later load from config file

@app.route('/<station>')
@app.route('/')
def main(station=None):
    global data_source

    all_stations = data_source.stations()
    online_stations = data_source.online_stations(threshold=WAIT_TIME_THRESHOLD)

    for stn in all_stations:
        if stn['id'] in online_stations:
            stn['online'] = True
        else:
            stn['online'] = False
    #all_stations = sorted(all_stations.items(), key=lambda key: key['online'])
    #print station_status
    return render_template('main.html', all_stations=all_stations, save=True)


@app.route('/trained')
def trained_model():
    """
    Retrieve trained models from the db and display it.
    Returns:

    """

    query = {'version': 1, 'start_date': 1, 'end_date': 1,
                        'weather_variable': 1,
                        'radius': 1, 'k': 1, 'views': 1, 'station': 1, 'training_date':1}

    trained_models  = mongo.db.model.find({}, query)
    trained_models = [ m for m in trained_models]
    return render_template("trained_model.html", trained_models=trained_models)

@app.route('/fitted')
def fitted_detail():
    """
    Display fitted parameter for the trained model.
    Returns:

    """
    try:

        _id = request.args.get('_id')
        app.logger.info(_id)
        query = {'_id': ObjectId(_id)}
        model_config = mongo.db.model.find_one(query)  # type: object

        if model_config is None:
            app.logger.error("Couldn't find the model {}", model_config)
            raise ValueError("Couldn't find the model")

        rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)

        scores, group_data = rqc.evaluate(model_config['start_date'], model_config['end_date']
                                          )
        metric_result, grp_eval= evaluate_groups(group_data, scores)
        graphs = {'point':{}, 'group':{}}
        # Matplotlib need to be installed on the bluemix to work correctly.
        # graphs['point']['ap'] = build_metric_graph(pred=scores, lbl=grp_eval['label'], plt_type='ap')
        # graphs['point']['auc'] = build_metric_graph(pred=scores, lbl=grp_eval['label'], plt_type='auc')
        #
        # graphs['group']['ap'] = build_metric_graph(pred=group_data['gp_score'], lbl=grp_eval['label'], plt_type='ap')
        # graphs['group']['auc'] = build_metric_graph(pred=group_data['gp_score'], lbl=grp_eval['label'], plt_type='auc')

        date_range = pd.date_range(model_config['start_date'], model_config['end_date'], freq='1D')

        raw_data = group_data['data'].reshape(-1)
        flag_data = group_data['label']
        graph_data = out_plot(raw_data, date_range, model_config['station'],flag_data=flag_data)
        return render_template('scores.html', title=model_config['station'], line_chart=graph_data, metrics=metric_result,
                               graphs=graphs)
    except Exception as e:
        app.logger.error(str(e))
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/train/<station>')
def train(station):
    """
    This train a single station using arguments
    Args:
        station (str): station name
        start_date (str): start_date
        end_date (str): end_date
        save (bool): if true saves the trained model.

    Returns:

    """
    # target_station = request.args.get('station')
    start_date = request.args.get('startDate',type=str)
    end_date = request.args.get('endDate', type=str)
    weather_variable = request.args.get('variable')
    save = request.args.get('save',type=bool)

    # target_station_q = "TA00030"
    if (start_date is None) or (end_date is None):
        start_date = "2016-01-01"
        end_date = "2016-12-31"
    if weather_variable is None:
        weather_variable = RAIN

    check_station, err = check_for_training(data_source=data_source, target_station=station, variable=weather_variable,
                                       start_date=start_date, end_date=end_date, config=training_config)
    if not check_station:
       return render_template("errorpage.html", error = err)

    fitted, train_parameters = train_station(station, weather_variable, start_date, end_date)

    if save:
       # Save model, commented is to save as file.
       # model_name = os.path.join(MODEL_DIR, station + weather_variable + "v00.pk")
       # joblib.dump(fitted.save(), open(model_name, 'w'))
        save_model(fitted, train_parameters)
    return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
                           train_config=train_parameters)


def save_model(fitted, train_parameters):
    pk_model = pickle.dumps(fitted.save())
    train_parameters['model'] = Binary(pk_model)
    mongo.db.model.insert(train_parameters)


def train_station(station, weather_variable, start_date, end_date):

    rqc = MainRQC(target_station=station, variable=weather_variable, data_source=data_source,
                  num_k_stations=MAX_K, radius=RADIUS)
    for view_name in VIEWS:
        rqc.add_view(name=view_name)
    for m_name in MODELS:
        rqc.add_module(name=m_name)
    fitted = rqc.fit(start_date=start_date,
                     end_date=end_date)
    train_parameters = {'version': VERSION, 'start_date': start_date, 'end_date': end_date,
                        'weather_variable': weather_variable,
                        'radius': RADIUS, 'k': len(fitted.k_stations), 'views': VIEWS, 'models': MODELS,
                        'station': station, 'k_stations':fitted.k_stations,
                        'training_date': datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')}
    return fitted, train_parameters


def train_all(data_source, start_date, end_date):
    """
    Batch training for all active stations.
    Args:
        data_source:
        start_date:
        end_date:

    Returns:

    """
    variable = RAIN
    config = {"radius": 100, "MAX_K": 5, "FRACTION_ROWS": 0.5, "MIN_STATION": 3}
    threshold_waiting_hour = 72
    all_stations = data_source.online_stations(threshold=threshold_waiting_hour) #, active_day_range=end_date)
    trained_station = {}

    for stn in all_stations:
         print ("Starting training operation for station {}".format(stn))
         trained, error= check_for_training(data_source, stn, variable, start_date, end_date, config)
         if trained:
             # if trained succesfully. save and move
             fitted, train_parameters = train_station(station=stn, weather_variable=variable, start_date=start_date, end_date=end_date)
             if fitted:
                 save_model(fitted, train_parameters)
                 error['failed'] = False
                 trained_station[stn] = error
                 print ("Model {} trained ".format(stn))
         else:
             # Register the errors and continue to another stations.
             error['failed'] = True
             trained_station[stn] = error
             app.logger.error(error)
         time.sleep(3)


    # save to json
    json.dump(trained_station, open(ROOT_DIR+'/app/asset/train_stat.json','w'))

## Scoring operations.

@app.route('/score/<target_station>')
def score(target_station):
    """
    Score a station from a trained model.
    Args:
        target_station:
        start
    Returns:

    """
    try:
        save_score = request.args.get('save',type=bool)
        start_date = request.args.get('startDate', type=str) # "2016-01-01"
        end_date = request.args.get('endDate', type=str) #"2016-09-30"

        if (start_date is None) or (end_date is None):
            return render_template("infopage.html", title="Error in scoring", message="Either start data or end date is missing")

        weather_variable = request.args.get('variable',type=str) #RAIN
        if weather_variable is None:
            weather_variable = RAIN

        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')

        query = {'station': target_station, 'weather_variable': weather_variable}
        model_config = mongo.db.model.find_one(query)
        if model_config is None:
            raise ValueError("Couldn't find the model")

        fitted, error = check_for_training(data_source=data_source, target_station=target_station,
                                           variable=weather_variable,
                                           start_date=start_date, end_date=end_date, config=training_config,
                                           k_stations=model_config['k_stations'])
        if not fitted:
            return render_template("errorpage.html", error=error)

        rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        # try plot.
        scores = result['MixLinearModel'].reshape(-1).tolist()
        threshold = np.quantile(scores, 0.95)

        if DEBUG:
            app.logger.error(scores)
        if save_score:
            score_result = {'model_id': model_config['_id'], 'station': target_station,
                            'weather_variable': weather_variable,
                            'scores': {str(date): score for date, score in zip(date_range, scores)}
                            }
            mongo.db.qc_score.insert(score_result)
            message = "Scores saved"
            return render_template('score_result.html',title=target_station, score_result=score_result['scores'], message=message,
                                   threshold=threshold)
    except Exception as e:
        app.logger.error(str(e))
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})
@app.route('/evaluate/<station>')
def evaluate(station):
    """
    Load trained model and evaluate on data with synthetic faults.
    Given the ground truth evaluate the AUC/PR or accuracy or number of false alaram.
    Returns:

    """
    try:
        start_date = request.args.get('startDate', type=str)  # "2016-01-01"
        end_date = request.args.get('endDate', type=str)  # "2016-09-30"
        weather_variable = request.args.get('variable', type=str)  # RAIN
        date_range = pd.date_range(start_date, end_date, freq='1D')

        query = {'station': station, 'weather_variable': weather_variable}

        # query db
        model_config = mongo.db.model.find_one(query)
        if model_config is None:
            raise ValueError("Couldn't find the model")

        rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        scores, group_data = rqc.evaluate(start_date, end_date, target_station=station)
        metric_result = evaluate_groups(group_data, scores)
        message = "metric sucss."
        return jsonify(metric_result)
        # Apply metric and plot result of it.
        # try plot.
        #scores = result['MixLinearModel'].reshape(-1).tolist()

        #graph_data = out_plot(scores, date_range, target_station)
        #return render_template('scores.html', title=target_station, line_chart=graph_data, message=message)
    except Exception as e:
        app.logger.error(str(e))
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})


# Delete operation.
@app.route('/wipe', methods=['GET'])
#@app.route('/wipe/<station>')
def wipe():
    try:
        #shutil.rmtree(MODEL_DIR)
        #os.makedirs(MODEL_DIR)
        query = {}
        station = request.args.get('station', type=str)
        _id = request.args.get('_id', type=str)
        if station is not None:
            query = {'station': station} #, 'weather_variable': variable, 'version': version}
            mongo.db.model.delete_many(query)
            app.logger.info('Models {} wiped'.format(query))
            message = "All model for station {} deleted".format(station)
        elif _id:
            query = {'_id': ObjectId(_id)}
            row = mongo.db.model.find_one(query)
            mongo.db.model.delete_one(query)
            app.logger.info('Models {} wiped'.format(query))
            message = "Model with id {} of station {} deleted ".format(_id, row['station'])
        else:
            app.logger.info('Models {} wiped'.format(query))
            mongo.db.model.delete_many(query)
            message = " All models in db are deleted"

        return render_template('infopage.html', title="Delete model", message=message)
    except Exception as e:
        app.logger.error('An error occured {}'.format(str(e)))
        return 'Could not remove and recreate the model directory {}'.format(str(e))


def parse_args():
    parser = argparse.ArgumentParser(description = "RQC command line arguments")
    parser.add_argument('-m', '--mode', help="Mode either production or dev")
    parser.add_argument('-d', '--datasource', help='Data source. TahmoAPI, FakeTahmo or TahmoBluemix')
    parser.add_argument('-p','--port', help='Flask port. Default 8000')
    parser.add_argument('-s','--startdate', help='start date for training')
    parser.add_argument('-e', '--enddate', help='end date for training')
    parser.add_argument('-t','--trainall', help='weather variable', type=bool)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    log_path = os.path.join(ROOT_DIR, "log/LOG_" + datetime.today().strftime("%Y-%m-%d") + ".log")
    #if not os.path.exists('log'):
    #    os.mkdir('log')
    log_path = os.path.basename(log_path)

    handler = RotatingFileHandler(log_path, maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)

    DEBUG = True
    args = parse_args()
    #args.datasource = 'TahmoAPILocal'
    #args.mode = 'production'

    if args.port is not None:
        port = int(args.port)
    else:
        port = 8000
    if args.mode is not None:
        mode = args.mode
    else:
        mode = 'dev'
    if args.datasource is not None:
        data_source = DataSourceFactory.get_model(args.datasource)
    else:
        data_source = DataSourceFactory.get_model('FakeTahmo')

    db_config = json.load(open(os.path.join(ROOT_DIR, "config/config.json"), "r"))["mongodb"]
    app.config["MONGO_URI"] = db_config[mode]  # could be
    MODEL_DIR = os.path.join(ROOT_DIR, "app/asset")

    mongo = PyMongo(app)
    app.logger.addHandler(handler)
    if args.trainall is not None:
        os.environ['PYTHONPATH'] = '.'

        if args.startdate is not None or args.enddate is not None:
            train_all(data_source, start_date=args.startdate, end_date=args.enddate)

        else:
            raise ValueError("Either startdate or enddate are missing")
    else:

        # mode = "production" #"dev" #"dev" #" #could "dev" or "production"
        # Db configuration


        port = int(os.getenv('PORT', port))

        app.run(host='0.0.0.0', port=port, debug=DEBUG)
