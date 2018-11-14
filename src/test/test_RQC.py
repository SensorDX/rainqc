from unittest import TestCase

from RQC import RQC
from src.datasource import ToyDataSource
from src.view import ViewDefinition


class TestRQC(TestCase):
    def test_build_view(self):
        target_station = 'TA0005'
        view_list = self.rqc.build_view(target_station=target_station,
                date_from='2017-01-01 00:00:00',
               date_to='2017-05-01 00:00:00')

        self.assertEqual(len(view_list), 1)
        self.assertIsInstance(view_list.values()[0], ViewDefinition)

    def setUp(self):
        self.rqc = RQC()
        self.rqc.data_source = ToyDataSource
        self.rqc.add("View","PairwiseView")
        #self.rqc.add_model()
    def test_add_view(self):
        self.assertEqual(self.rqc.view_registry.values()[0],'PairwiseView')
    def test_toy_datasource(self):
        self.assertEqual(len(self.rqc.data_source.station_list()), 19)



