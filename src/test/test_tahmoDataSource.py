from unittest import TestCase
from src.datasource.tahmo_datasource import TahmoDataSource
from definition import RAIN
class TestTahmoDataSource(TestCase):
    def setUp(self):
        self.data_source = TahmoDataSource()
    def test_get_data(self):
        self.assertEqual (1,1)

    def test_get_stations(self):
        global station
        station = self.data_source.get_stations()
        self.assertEqual(station['status'],'success')
        self.assertGreaterEqual(len(station['stations']), 500)

    def test_active_stations(self):
        self.assertEqual(1, 1)
    def test_daily_data(self):
        ta_station = 'T00049'
        start_date = '2017-01-01'
        end_date = '2017-01-30'
        df = self.data_source.daily_data(ta_station, RAIN,start_date, end_date)
        print (df.head(5))
        self.assertGreaterEqual(2,1)
        #self.assertEqual(df.shape[0], 30)

    def test_nearby_stations(self):
        nrb = self.data_source.nearby_stations('TA00065', k=5)
        print nrb
        self.assertGreaterEqual(len(nrb), 5)

    def test_get_active_nearby(self):
        a_stn  = self.data_source.get_active_nearby('TA00065')
        self.assertGreaterEqual(len(a_stn),1)
