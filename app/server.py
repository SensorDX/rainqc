import sys
import os
import shutil
import time
import pandas as pd
import traceback
from rqc_main import MainRQC
from flask import Flask, request, jsonify, render_template, Response, redirect
from definition import RAIN, ROOT_DIR
from sklearn.externals import joblib
import logging
from logging.handlers import RotatingFileHandler
from datasource.fake_tahmo import FakeTahmo
from view.view import PairwiseView
from model.hurdle_regression import MixLinearModel
import pygal

app = Flask(__name__)

#parameters
MODEL_DIR = os.path.join(ROOT_DIR, "app/asset")


@app.route('/<station>')
@app.route('/')
def rqc(station=None):
    global  data_source
    data_source = FakeTahmo()
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
    target_station_q = "TA00030"
    start_date = "2016-01-01"  # (datetime.datetime.now(timezone('utc'))-datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M')
    end_date = "2016-06-30"  # (datetime.datetime.now(timezone('utc')) - datetime.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M')

    print("Training ...")
    save = request.args.get('save')
    if weather_variable is None:
        weather_variable = RAIN
    rqc = MainRQC(target_station=station, variable=weather_variable, data_source=data_source)
    rqc.add_view(name="PairwiseView", view=PairwiseView())
    rqc.add_module(name="MixLinearModel", module=MixLinearModel())
    fitted = rqc.fit(start_date=start_date,
                    end_date=end_date)

    if save is not None:
        model_name = os.path.join(MODEL_DIR, station+weather_variable+"v00.pk")
        joblib.dump(fitted.save(), open(model_name,'w'))
    return render_template('training.html', stationlist=fitted.k_stations, target_station=station, save=save)


@app.route('/score/<station_target>')
def score(station_target):
    """
    Score a station from a trained model.
    Args:
        station_target:

    Returns:

    """

    try:

        start_date = "2016-01-01"
        end_date = "2016-09-30"
        model_name = os.path.join(MODEL_DIR, station_target + RAIN + "v00.pk")
        rqc_pk = joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        # try plot.
        scores = result['MixLinearModel'].reshape(-1).tolist()
        date_range = pd.date_range(start_date, end_date, freq='1D')
        return output_score(scores, date_range, station_target)

        #return jsonify(result) #{'model_name':rqc.target_station, 'status':'success'})
    except Exception, e:
         app.logger.error(str(e))
         return jsonify({'error': str(e), 'trace': traceback.format_exc()})

def output_score(scores, date_range, target_station):
    line_map = [(dt.date(), sc) for dt, sc in zip(date_range, scores)]
    graph = pygal.DateLine(x_label_rotation=35, stroke=False, human_readable=True)  # disable_xml_declaration=True)
    graph.force_uri_protocol = 'http'
    graph.title = '{}: Score.'.format(target_station)
    graph.add(target_station, line_map)
    graph_data = graph.render_data_uri()  # is_unicode=True)
    return render_template('scores.html', title=target_station, bar_chart=graph_data)


@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
        print('Models wiped')

        return redirect('/')

    except Exception, e:
        app.logger.error('An error occured {}'.format(str(e)))
        return 'Could not remove and recreate the model directory'



## Testing

@app.route('/pygal')
def pygalexample():

    try:
        graph = pygal.Line(disable_xml_declaration=True)
        graph.title = 'Change Coolness of programming languages over time.'
        graph.x_labels = ['2011','2012','2013','2014','2015','2016']
        graph.add('Python',  [15, 31, 89, 200, 356, 900])
        graph.add('Java',    [15, 45, 76, 80,  91,  95])
       # graph.add('C++',     [5,  51, 54, 102, 150, 201])
        graph.add('All others combined!',  [5, 15, 21, 55, 92, 105])

        graph_data = graph.render_data_uri() #.render(is_unicode=True)
        return render_template('scores.html', title="pygalgraph", bar_chart=graph_data)

    except Exception, e:
        return(str(e))



if __name__ == '__main__':
    handler = RotatingFileHandler('foo.log', maxBytes=10000, backupCount=1)
    handler.setLevel(logging.INFO)
    app.logger.addHandler(handler)
    
    # try:
    #     port = int(sys.argv[1])
    # except Exception, e:
    #     port = 8000
    #
    # try:
    #     clf = joblib.load(model_file_name)
    #     print 'model loaded'
    #     model_columns = joblib.load(model_columns_file_name)
    #     print 'model columns loaded'
    #
    # except Exception, e:
    #     print 'No model here'
    #     print 'Train first'
    #     print str(e)
    #     clf = None
    port = 8000

    app.run(host='0.0.0.0', port=port, debug=True)
