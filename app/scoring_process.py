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
import pandas as pd
import pickle
from training_operation import check_for_training
from src.rqc import MainRQC

def score_station(data_source, target_station, start_date, end_date, weather_variable, model_config, training_config):

        date_range = pd.date_range(start=start_date, end=end_date, freq='1D')

        # find model in training session.

        fitted, error = check_for_training(data_source=data_source, target_station=target_station,
                                           variable=weather_variable,
                                           start_date=start_date, end_date=end_date, config=training_config,
                                           k_stations=model_config['k_stations'])
        if not fitted:
            return "Model not available"
            #return render_template("errorpage.html", error=error)

        rqc_pk = pickle.loads(model_config['model'])  # joblib.load(open(model_name,'r'))
        rqc = MainRQC.load(rqc_pk)
        result = rqc.score(start_date, end_date)
        # try plot.
        scores = result['MixLinearModel'].reshape(-1).tolist()
        threshold = np.quantile(scores, float(app_config["score"]["quantile"]))

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


# noinspection PyUnreachableCode
class StationPool:

    def __init__(self, data_source):
        self.data_source = data_source
    @classmethod
    def pool_active_stations(cls, start_date):
        return cls.data_source.active_stations(start_date)

def score(station, start_date, end_date):
    pass

def fitted_stations():
    pass


# Begin score operations.
def score_operation(start_date, end_date):

    active_stations = StationPool.pool_active_stations(start_date)
    # list all fitted stations.
    query_fitted_stations = {'station':1,'_id':0}
    # query = {'station': target_station, 'weather_variable': weather_variable}
    fitted_stations = mongo.db.model.find({}, query_fitted_stations)
    if len(fitted_stations)<1:
        raise ValueError('There is no fitted model in the database')
    active_fitted_stations = set(fitted_stations).intersection(active_stations)
    non_score_stations = set(active_fitted_stations).difference(fitted_stations)
    for station in active_fitted_stations:
        # For each available fitted station. Score the
        score_result = score(start_date, end_date, station)
        if not score_result:
            # Write failure log to db.
            non_score_stations.add(station)
            continue
        else:
            pass
            # Write success log to db.
    # Write non-trained stations:
    return non_score_stations