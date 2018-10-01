from unittest import TestCase
from view.view import PairwiseView, View
import numpy as np
class TestPairwiseView(TestCase):

    def setUp(self):
        self.num_stations = 4
        self.n = 200
        self.stations = np.random.randn(self.n, self.num_stations)
        self.pv = PairwiseView(variable='pr')

    def test_make_view(self):

        ## Slicing in this way preserves the dimension of the array
        #ref: https://stackoverflow.com/questions/3551242/numpy-index-slice-without-losing-dimension-information

        X = self.pv.make_view(self.stations[:, 0:1:], [self.stations[:, 1:2:]]).x
        self.assertEqual(X.shape[0], self.n)
        y = self.pv.make_view(self.stations[:,0:1:],[self.stations[:, 1:2:],
                              self.stations[:, 2:3:],self.stations[:, 3:4:]]).x
        self.assertEqual(y.shape, (self.n, self.num_stations-1))
        self.assertIsNone(self.pv.label)

        ## Test multiple pairwise views
        for i in range(self.num_stations-1):
            vw = self.pv.make_view(self.stations[:, 0:1:],
                                   [self.stations[:, (i+1):(i+2):]]).x
            self.assertEqual(vw.shape[0], self.n )

        # Name


