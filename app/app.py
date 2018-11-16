import os,sys ,shutil, pickle
from datetime import datetime
import pandas as pd
import traceback
from src import MainRQC
from flask import Flask, request, jsonify, render_template, redirect
from definition import RAIN
from sklearn.externals import joblib
import logging
from logging.handlers import RotatingFileHandler
from src.datasource import FakeTahmo, TahmoDataSource
import pygal
from bson.binary import Binary
from flask_pymongo import PyMongo,ObjectId
from definition import ROOT_DIR
import json


app = Flask(__name__)
mode = "dev" #"dev" #" #could "dev" or "production"
# Db configuration
db_config = json.load(open(os.path.join(ROOT_DIR, "config/config.json"),"r"))["mongodb"]
app.config["MONGO_URI"] = db_config[mode]  #could be
MODEL_DIR = os.path.join(ROOT_DIR, "app/asset")

mongo = PyMongo(app)

# Parameters
RADIUS = 100
MAX_K = 5
WEATHER_VARIABLE = RAIN
VIEWS = ['PairwiseView']
MODELS = ['MixLinearModel']
VERSION = 0.1

#global variable.
data_source = TahmoDataSource() # FakeTahmo()  #  datasource connection object.
WAIT_TIME_THRESHOLD = 72
@app.route('/<station>')
@app.route('/')
def main(station=None):
    global data_source

    all_stations = data_source.stations()
    online_stations = data_source.online_station(threshold=WAIT_TIME_THRESHOLD)

    for stn in all_stations:
        if stn['id'] in online_stations:
            stn['online'] = True
        else:
            stn['online'] = False

    #print station_status
    return render_template('main.html', all_stations=all_stations, save=True)


@app.route('/trained')
def trained_model():
    """
    Retrieve trained models from the db and display it.
    Returns:

    """
    #query = {'weather_variable': weather_variable}
    query = {'version': 1, 'start_date': 1, 'end_date': 1,
                        'weather_variable': 1,
                        'radius': 1, 'k': 1, 'views': 1, 'station': 1}


    # mongo.db.src.find(query)
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

        result = rqc.score(model_config['start_date'], model_config['end_date'])
            # try plot.
        date_range = pd.date_range(model_config['start_date'], model_config['end_date'], freq='1D')
        scores = result['MixLinearModel'].reshape(-1).tolist()
        graph_data = out_plot(scores, date_range, model_config['station'])
        return render_template('scores.html', title=model_config['station'], line_chart=graph_data)
    except Exception, e:
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
    evaluate = request.args.get('evaluate', type=bool)
    save = request.args.get('save',type=bool)

    # target_station_q = "TA00030"
    if (start_date is None) or (end_date is None):
        start_date = "2016-01-01"
        end_date = "2016-12-31"
    if weather_variable is None:
        weather_variable = RAIN
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
                        'radius': RADIUS, 'k': MAX_K, 'views': VIEWS, 'models': MODELS, 'station': station}

    if save:
        model_name = os.path.join(MODEL_DIR, station + weather_variable + "v00.pk")
        joblib.dump(fitted.save(), open(model_name, 'w'))
        pk_model = pickle.dumps(fitted.save())
        train_parameters['model'] = Binary(pk_model)
        mongo.db.model.insert(train_parameters)
    if evaluate:
        # scores = _score(target_station=station, weather_variable=weather_variable,
        #                 start_date=start_date, end_date=end_date)
        scores = rqc.score(start_date, end_date, target_station=station)['MixLinearModel'].reshape(-1).tolist()
        date_range = pd.date_range(start_date, end_date, freq='1D')
        graph = out_plot(scores, date_range=date_range, target_station=station)
        # raw data.
        raw_data = data_source.daily_data(target_station=station, target_variable=RAIN,
                                          start_date=start_date, end_date=end_date).reshape(-1)
        raw_plt = out_plot(raw_data, date_range, target_station=station)
        return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
                               train_config=train_parameters, line_chart=raw_plt)

    return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
                           train_config=train_parameters)

@app.route('/trainall')
def train_all():
    """
    Train all stations
    Returns:

    """
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')
    weather_variable = request.args.get('variable')


    threshold_waiting_hour = 72
    all_stations = data_source.online_station(threshold=threshold_waiting_hour, active_day_range=end_date)
    for station in all_stations:
        # train each station seperately
        # Add waiting time and error handling for each station.
        train(station)

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
        save = request.args.get('save', type=bool)
        start_date = "2016-01-01"  # request.args.get('start_date') # "2016-01-01"
        end_date = "2016-06-30"  # request.args.get('end_date') #"2016-09-30"
        weather_variable = RAIN  # request.args.get('variable') #RAIN
        date_range = pd.date_range(start_date, end_date, freq='1D')

        query = {'station': target_station, 'weather_variable': weather_variable}

        # query db
        model_config = mongo.db.model.find_one(query)
        if model_config is None:
            raise ValueError("Couldn't find the model")

        rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        # try plot.
        scores = result['MixLinearModel'].reshape(-1).tolist()
        message = "Scores not yet saved."
        if DEBUG:
            app.logger.error(scores)
        if save:
            score_result = {'model_id': model_config['_id'], 'station': target_station,
                            'weather_variable': weather_variable,
                            'scores': {str(date): score for date, score in zip(date_range, scores)}
                            }
            mongo.db.qc_score.insert(score_result)
            message = "Scores saved"

        graph_data = out_plot(scores, date_range, target_station)
        return render_template('scores.html', title=target_station, line_chart=graph_data, message=message)
    except Exception, e:
        app.logger.error(str(e))
        return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def evaluate():
    """
    Load trained model and evaluate on data with synthetic faults.
    Given the ground truth evaluate the AUC/PR or accuracy or number of false alaram.
    Returns:

    """
    pass
def display_score():
    """
    Display anomlay score with their ground truth.
    - Load the data and anomaly score and mark it with
    Returns:

    """
    pass

# def _score(target_station, weather_variable, start_date, end_date):
#     try:
#
#         #model_name = os.path.join(MODEL_DIR, target_station + weather_variable + "v00.pk")
#         query = {'station': target_station, 'weather_variable': weather_variable}
#
#         # mongo.db.model.find(query)
#         model_config = mongo.db.model.find_one(query)
#         if model_config is None:
#             return ValueError("Couldn't find the model")
#
#         rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
#         rqc = MainRQC.load(rqc_pk)
#         result = rqc.score(start_date, end_date)
#         # try plot.
#         scores = result['MixLinearModel'].reshape(-1).tolist()
#         return scores, model_config['_id']
#
#         # return jsonify(result) #{'model_name':rqc.target_station, 'status':'success'})
#     except Exception, e:
#         app.logger.error(str(e))
#         return jsonify({'error': str(e), 'trace': traceback.format_exc()})
#
#
# @app.route('/evaluate/<station>')
# def evaluate_metric(station):
#     """
#     Evaluate station from fitted data.
#     Args:
#         station:
#
#     Returns:
#
#     """
#     start_date = request.args.get('startDate')
#     end_date = request.args.get('endDate')
#     weather_variable = request.args.get('variable')
#
#     # Load trained model.
#
#     scores = _score(target_station=station, weather_variable=weather_variable,
#                     start_date=start_date, end_date=end_date)
#     date_range = pd.date_range(start_date, end_date, freq='1D')
#     graph = out_plot(scores, date_range=date_range, target_station=station)
#     # raw data.
#     # raw_data = data_source.daily_data(target_station=station, target_variable=RAIN,
#     #                                   start_date=start_date, end_date=end_date).reshape(-1)
#     # raw_plt = out_plot(raw_data, date_range, target_station=station)
#     #
#     # return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
#     #                        train_config=train_parameters, line_chart=raw_plt)


@app.route('/wipe/<station>')
@app.route('/wipe', methods=['GET'])
def wipe(station=None):
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
        query = {}
        if station is not None:
            station = request.args.get('station')
            variable = request.args.get('weather_variable')
            version = request.args.get('version')
            query = {'station': station, 'weather_variable': variable, 'version': version}
        mongo.db.model.deleteMany(query)
        app.logger.info('Models wiped')
        print('Models wiped')

        return redirect('/')

    except Exception, e:
        app.logger.error('An error occured {}'.format(str(e)))
        return 'Could not remove and recreate the model directory'


# Plot operations

def out_plot(scores, date_range, target_station):
    line_map = [(dt.date(), sc) for dt, sc in zip(date_range, scores)]
    graph = pygal.DateLine(x_label_rotation=35, stroke=False, human_readable=True)  # disable_xml_declaration=True)
    graph.force_uri_protocol = 'http'
    graph.title = '{}: Score.'.format(target_station)
    graph.add(target_station, line_map)
    graph_data = graph.render_data_uri()  # is_unicode=True)
    return graph_data


if __name__ == '__main__':
    log_path = os.path.join(ROOT_DIR, "log/LOG_" + datetime.today().strftime("%Y-%m-%d") + ".log")
    handler = RotatingFileHandler(log_path, maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    DEBUG = True
    if len(sys.argv)>1:
        port = int(sys.argv[1])
    else:
        port = 5000
    app.run(host='0.0.0.0', port=port, debug=DEBUG)
