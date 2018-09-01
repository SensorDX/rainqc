from unittest import TestCase
from RQC import RQC
from services.data_source import LocalDataSource
class TestRQC(TestCase):
    def test_build_view(self):
        target_station = 'TA00025'
        view_list = self.rqc.build_view(target_station=target_station,
                 date_from='2017-01-01 00:00:00',
                 date_to='2017-05-01 00:00:00')

        self.assertEqual(len(view_list), 1)

    def setUp(self):
        self.rqc = RQC(target_station='TA00020')
        self.rqc.data_source = LocalDataSource
        self.rqc.add("View","PairwiseView")
        self.rqc.add_model()




