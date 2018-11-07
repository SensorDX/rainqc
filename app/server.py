import sys
import os
import shutil
import time
import traceback
from rqc_main import MainRQC
from flask import Flask, request, jsonify, render_template
from definition import RAIN, ROOT_DIR
from sklearn.externals import joblib
import logging
from logging.handlers import RotatingFileHandler
from datasource.fake_tahmo import FakeTahmo
from view.view import PairwiseView
from model.hurdle_regression import MixLinearModel
app = Flask(__name__)

#parameters
MODEL_DIR = os.path.join(ROOT_DIR, "app/asset")


@app.route('/<station>')
@app.route('/')
def rqc(station=None):
    return render_template('main.html', station=station) #'<h2> This is an  RQC server</h2>'

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
    rqc = MainRQC(target_station=station, variable=weather_variable, data_source=FakeTahmo())
    rqc.add_view(name="PairwiseView", view=PairwiseView())
    rqc.add_module(name="MixLinearModel", module=MixLinearModel())
    fitted = rqc.fit(start_date=start_date,
                    end_date=end_date)

    if save:
        model_name = os.path.join(MODEL_DIR, station+start_date+end_date+weather_variable+"v00.pk")
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
        end_date = "2016-06-30"
        model_name = os.path.join(MODEL_DIR, station_target + start_date + end_date + RAIN + "v00.pk")
        rqc_pk = joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        return jsonify(result) #{'model_name':rqc.target_station, 'status':'success'})
    except Exception, e:
         return jsonify({'error': str(e), 'trace': traceback.format_exc()})

@app.route('/wipe', methods=['GET'])
def wipe():
    try:
        shutil.rmtree(MODEL_DIR)
        os.makedirs(MODEL_DIR)
        return 'Model wiped'

    except Exception, e:
        print str(e)
        return 'Could not remove and recreate the model directory'






# # input
# training_data = 'data/titanic.csv'
# include = ['Age', 'Sex', 'Embarked', 'Survived']
# dependent_variable = include[-1]
#
# model_directory = 'model'
# model_file_name = '%s/model.pkl' % model_directory
# model_columns_file_name = '%s/model_columns.pkl' % model_directory
#
# # These will be populated at training time
# model_columns = None
# clf = None
# @app.route('/')
# def foo():
#     app.logger.warning('A warning occurred (%d apples)', 42)
#     app.logger.error('An error occurred')
#     app.logger.info('Info')
#     return "foo"
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     if clf:
#         try:
#             json_ = request.json
#             query = pd.get_dummies(pd.DataFrame(json_))
#
#             # https://github.com/amirziai/sklearnflask/issues/3
#             # Thanks to @lorenzori
#             query = query.reindex(columns=model_columns, fill_value=0)
#
#             prediction = list(clf.predict(query))
#
#             return jsonify({'prediction': prediction})
#
#         except Exception, e:
#
#             return jsonify({'error': str(e), 'trace': traceback.format_exc()})
#     else:
#         print 'train first'
#         return 'no model here'
#
#
# @app.route('/train', methods=['GET'])
# def train():
#     # using random forest as an example
#     # can do the training separately and just update the pickles
#     from sklearn.ensemble import RandomForestClassifier as rf
#
#     df = pd.read_csv(training_data)
#     df_ = df[include]
#
#     categoricals = []  # going to one-hot encode categorical variables
#
#     for col, col_type in df_.dtypes.iteritems():
#         if col_type == 'O':
#             categoricals.append(col)
#         else:
#             df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic
#
#     # get_dummies effectively creates one-hot encoded variables
#     df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)
#
#     x = df_ohe[df_ohe.columns.difference([dependent_variable])]
#     y = df_ohe[dependent_variable]
#
#     # capture a list of columns that will be used for prediction
#     global model_columns
#     model_columns = list(x.columns)
#     joblib.dump(model_columns, model_columns_file_name)
#
#     global clf
#     clf = rf()
#     start = time.time()
#     clf.fit(x, y)
#     print 'Trained in %.1f seconds' % (time.time() - start)
#     print 'Model training score: %s' % clf.score(x, y)
#
#     joblib.dump(clf, model_file_name)
#
#     return 'Success'
#
#
#
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
