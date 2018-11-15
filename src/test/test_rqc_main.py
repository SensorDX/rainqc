from src import MainRQC
from src.datasource import FakeTahmo
from src.datasource import TahmoDataSource
from definition import RAIN
import pickle
if __name__ == "__main__":
    from src.view import PairwiseView
    from src.model import MixLinearModel
    fd = globals()['FakeTahmo']()
    print(fd)
    x = 2
    #data_sourcex = FakeTahmo(local_data_source="experiments/dataset/tahmostation2016.csv",
    #                        nearby_station="localdatasource/nearest_stations.csv")
    tahmo_datasource = TahmoDataSource() #nearby_station_location="datasource/station_nearby.json")
    target_station_q = "TA00094"
    start_date = "2017-01-01"  # (datetime.datetime.now(timezone('utc'))-datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M')
    end_date = "2017-06-30"  # (datetime.datetime.now(timezone('utc')) - datetime.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M')
    dd = MainRQC(data_source=tahmo_datasource,
                 target_station=target_station_q, radius=200)
    dd.add_view(name="PairwiseView", view=PairwiseView())
    dd.add_module(name="MixLinearModel", module=MixLinearModel())
    #print tahmo_datasource.get_stations()
    #print tahmo_datasource.daily_data(target_station_q, RAIN, start_date, end_date)
    fitted = dd.fit(start_date=start_date,
                    end_date=end_date)
    pickle.dump(dd.save(), open(target_station_q+'_trained.pk','w'))
    result = dd.score(start_date=start_date, end_date=end_date)
    print result
    # #print(result)
    # # save src
    # print result['MixLinearModel'].shape, type(result['MixLinearModel'])
    # # jj = dd.save()
    # print (jj)
    # joblib.dump(jj, open('dump.pk','w'))
    # dx = MainRQC.load(jj)
    # result2 = dx.score(start_date=start_date, end_date=end_date)
    # print (result2)
    # assert all([r1==r2 for r1, r2 in zip(result, result2)])
    #assert result==result2
## Station and create a perfect data that can work with the algorithm.
#TODO: write unit-test for the tahmodata source api.