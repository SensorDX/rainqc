import sys
import os
import shutil
import pickle
import pandas as pd
import traceback
from rqc_main import MainRQC
from flask import Flask, request, jsonify, render_template,  redirect
from definition import RAIN, ROOT_DIR
from sklearn.externals import joblib
import logging
from logging.handlers import RotatingFileHandler
from datasource.fake_tahmo import FakeTahmo
import pygal
from bson.binary import Binary
from flask_pymongo import PyMongo
#from pymongo import MongoClient

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/rainqc"
#client = MongoClient('localhost',27017)
#db = client['rainqc']
mongo = PyMongo(app)
#app
app = Flask(__name__)

#parameters, # radius, num_station, weather_variable


MODEL_DIR = os.path.join(ROOT_DIR, "app/asset")
data_source = FakeTahmo() #datasource connection object.
RADIUS = 100
MAX_K = 5
WEATHER_VARIABLE = RAIN
VIEWS = ['PairwiseView']
MODELS = ['MixLinearModel']
VERSION = 0.1



@app.route('/<station>')
@app.route('/')
def rqc(station=None):
    global  data_source
    #data_source = FakeTahmo()
    all_station = data_source.stations()
    active_station = data_source.active_stations(all_station)
    statation_status = [{"active":True if stn in active_station else False, "site_code":stn} for stn in all_station]

    return render_template('main.html', all_stations=statation_status, save=True) #'<h2> This is an  RQC server</h2>'

@app.route('/train/<station>')
def train(station):
    #target_station = request.args.get('station')
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')
    weather_variable = request.args.get('variable')
    evaluate = request.args.get('evaluate')
    save = request.args.get('save')

    #target_station_q = "TA00030"
    start_date = "2016-01-01"  # (datetime.datetime.now(timezone('utc'))-datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M')
    end_date = "2016-06-30"  # (datetime.datetime.now(timezone('utc')) - datetime.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M')


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

    train_parameters = {'version':VERSION, 'start_date': start_date, 'end_date': end_date, 'weather_variable': weather_variable,
                        'radius':RADIUS, 'k':MAX_K, 'views':VIEWS, 'models':MODELS, 'station':station}

    if save is not None:
        model_name = os.path.join(MODEL_DIR, station+weather_variable+"v00.pk")
        joblib.dump(fitted.save(), open(model_name,'w'))
        pk_model = pickle.dumps(fitted.save())
        train_parameters['model'] = Binary(pk_model)
        mongo.db.model.insert(train_parameters)
    if evaluate:
        scores = _score(target_station=station, weather_variable=weather_variable,
                        start_date=start_date, end_date=end_date)

        date_range = pd.date_range(start_date, end_date, freq='1D')
        graph = output_score(scores, date_range=date_range,target_station=station)
        return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
                        train_config=train_parameters, line_chart=graph)

    return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save,
                           train_config=train_parameters)


@app.route('/score/<target_station>')
def score(target_station):
    """
    Score a station from a trained model.
    Args:
        target_station:
        start
    Returns:

    """
    start_date = "2016-01-01" #request.args.get('start_date') # "2016-01-01"
    end_date = "2016-09-30" #request.args.get('end_date') #"2016-09-30"
    weather_variable = RAIN #request.args.get('variable') #RAIN
    date_range = pd.date_range(start_date, end_date, freq='1D')
    scores = _score(target_station, weather_variable, start_date, end_date)
    graph_data = output_score(scores, date_range, target_station)
    return render_template('scores.html', title=target_station, line_chart=graph_data)


def _score(target_station, weather_variable, start_date, end_date):
    try:

        model_name = os.path.join(MODEL_DIR, target_station + weather_variable + "v00.pk")
        query = {'station':target_station, 'weather_variable':weather_variable}

        #mongo.db.model.find(query)
        model_config = mongo.db.model.find_one(query)
        if model_config is None:
            return ValueError("Couldn't find the model")

        rqc_pk =  pickle.loads(model_config['model'])   #joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        # try plot.
        scores = result['MixLinearModel'].reshape(-1).tolist()
        return scores

        #return jsonify(result) #{'model_name':rqc.target_station, 'status':'success'})
    except Exception, e:
         app.logger.error(str(e))
         return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/evaluate/<station>')
def evaluate_metric(station):
    """
    Evaluate station from fitted data.
    Args:
        station:

    Returns:

    """
    pass


def output_score(scores, date_range, target_station):
    line_map = [(dt.date(), sc) for dt, sc in zip(date_range, scores)]
    graph = pygal.DateLine(x_label_rotation=35, stroke=False, human_readable=True)  # disable_xml_declaration=True)
    graph.force_uri_protocol = 'http'
    graph.title = '{}: Score.'.format(target_station)
    graph.add(target_station, line_map)
    graph_data = graph.render_data_uri()  # is_unicode=True)
    return graph_data
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
           query = {'station':station, 'weather_variable':variable, 'version':version}
        mongo.db.model.deleteMany(query)
        print('Models wiped')

        return redirect('/')

    except Exception, e:
        app.logger.error('An error occured {}'.format(str(e)))
        return 'Could not remove and recreate the model directory'







if __name__ == '__main__':
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    port = 8000
    app.run(host='0.0.0.0', port=port, debug=True)
