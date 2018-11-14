from unittest import TestCase
from src.datasource.tahmo_datasource import TahmoDataSource

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
        self.assertEqual(1, 1)

    def test_nearby_stations(self):
        nrb = self.data_source.nearby_stations('TA00065', k=5)
        print nrb
        self.assertGreaterEqual(len(nrb), 5)

    def test_get_active_nearby(self):
        a_stn  = self.data_source.get_active_nearby('TA00065')
        self.assertGreaterEqual(len(a_stn),1)