from rqc_main import MainRQC
from datasource.FakeTahmo import FakeTahmo
from datasource.tahmo_datasource import TahmoDataSource
import datetime
if __name__ == "__main__":
    from view.view import PairwiseView
    from model.hurdle_regression import MixLinearModel

    x = 2
    data_sourcex = FakeTahmo(local_data_source="experiments/dataset/tahmostation2016.csv",
                            nearby_station="localdatasource/nearest_stations.csv")
    tahmo_datasource = TahmoDataSource(nearby_station_location="datasource/station_nearby.json")
    target_station_q = "TA00030"
    start_date = "2016-01-01"  # (datetime.datetime.now(timezone('utc'))-datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M')
    end_date = "2016-06-30"  # (datetime.datetime.now(timezone('utc')) - datetime.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M')
    dd = MainRQC(data_source=data_sourcex,
                 target_station=target_station_q, radius=200)
    dd.add_view(name="pairwise", view=PairwiseView())
    dd.add_module(name="MixLinearModel", module=MixLinearModel())
    fitted = dd.fit(start_date=start_date,
                    end_date=end_date)
    result = dd.score(start_date=start_date, end_date=end_date)
    print(result)

## TODO: Work on downloaded data, the bluemix data is unreliable.{ The downloaded data is not also consistent}
## TODO: Work on synthetic data, and make sure the algorithm can be deployed and tested. Sample rainfall data or weather data, from a given
## Station and create a perfect data that can work with the algorithm.
