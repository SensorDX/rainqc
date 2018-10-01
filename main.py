import numpy as np
import pandas as pd

from datasource.data_source import  LocalDataSource
from model.hurdle_regression import MixLinearModel
from view.feature_extractor import PairwiseView

"""
Main workflow:
 1. Input:
    target_station: 
"""

LocalDataSource.local_project_path = './'
def main_local():

    target_station = "TA00025"
    fe = PairwiseView(data_source=LocalDataSource, variable='pr', num_k_station=5)
    fe.make_view(target_station=target_station,
                 date_from='2017-01-01 00:00:00',
                 date_to='2017-05-01 00:00:00')
    X, y, label = fe.X, fe.y, fe.label

    ## Train the model.
    mlm = MixLinearModel()
    mlm.fit(X, y)
    mlm.save(model_id=target_station, model_path="rainqc_model")
    score = mlm.predict(y, X)
    print score
    #print -np.log(score)

    # Score/ anomalies for incoming observation.
    mscore= MixLinearModel()
    mscore.load(model_id=target_station, model_path="rainqc_model")
    score = mscore.predcit(X, y)
    print -np.log(score)

def main_pairwise_station():
    target_station = "TA00025"
    fe = PairwiseView(data_source=LocalDataSource)
    fe.make_view(target_station=target_station)
    X, y, label = fe.X, fe.y, fe.label

    ## Train pairwise



def main_test():
    """
    Operation:
     - Train operation
        -input: target_station, year.
        -output: trained_model for a given station
        -operation:
            - train
     - Prediction:
        - input: target_station observation
        - output: likelihood score.
        - operation:
            - load trained model
            - load observed data from target_station and nearby stations.






    Input:
        target_station: target station
        datetime: datetime for qc.

        process:
          -
    :return:
    """


    df = pd.read_csv('sampletahmo.csv')
    w = np.random.rand(100, 6)
    #print df.head(5)
    y, x = w[:, 0], w[:, 1:]
    # print y
    y,x = df.iloc[:,1].as_matrix(), df.iloc[:,2:].as_matrix()

    mixl = MixLinearModel()
    mixl.reg_model.fit(x, y)
    yz = (y > 0.5).astype(int)
    mixl.log_reg.fit(x, yz)
    res = y.reshape(-1, 1) - mixl.reg_model.predict(x).reshape(-1, 1)
    # print res, res.shape
    # mixl.kde.fit(res)
    # print mixl.kde.score_samples(y.reshape(-1,1))
    mixl.train(y, x)
    #print mixl.predict(y, x)
    #mixl.save()
    mxx = MixLinearModel()
    mxx.to_json()
    mxx.from_json()
    print (mxx.predict(y, x))



if __name__ == '__main__':
    main_local()



