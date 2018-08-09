from services.data_source import LocalDataSource as LDS
from feature_extractor import PairwiseView


def test_paire_view():
    fe = PairwiseView(data_source=LDS, variable='pr', num_k_station=5)
    fe.make_features(target_station='TA00024',
                     date_from='2017-01-01 00:00:00', date_to='2017-01-11 00:00:00')

    print fe.X

if __name__ == '__main__':
    test_paire_view()

    # target_station = 'TA00020'
    # print LDS.local_project_path
    # ## Nearest station to 'TA0020'
    # #print LDS.nearby_stations(target_station)
    #
    #
    # # Measurements
    # print LDS.measurements(station_name=target_station, weather_variable='pr',
    #                        date_from='2017-01-01 00:00:00', date_to='2017-01-11 00:00:00',
    #                        group='D')
    #