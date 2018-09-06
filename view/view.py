import numpy as np
import pandas as pd
from collections import defaultdict

class View(object):
    def __init__(self, variable="pr"):
        self.X = None
        self.y = None
        self.label = None
        self.view_metadata = defaultdict()

        self.variable = variable
        # self.data_source = data_source

    def make_view(self, target_station, stations, **kwargs):
        return NotImplementedError


class ViewDefinition:
    """
    View definition format.
    """

    def __init__(self, name=None, label=None, x=None, y=None):
        self.name = name
        self.label = label
        self.x = x
        self.y = y


class ViewFactory:
    @staticmethod
    def create_view(view_type):
        if view_type == 'PairwiseView':
            return PairwiseView()

class PairwiseView(View):

    def __init__(self, variable=None):
        self.__name__ = "PairwiseView"
        super(PairwiseView, self).__init__(variable=variable)

    def make_view(self, target_station, stations, **options):
        """

        Args:
            target_station: 1xD numpy array
            stations: list of 1xD numpy array
            **options: key,value optional

        Returns:

        """

        len_series = len(target_station)
        assert all(len_series == len(stn) for stn in stations)  # Check dimension mismatch.
        tuples_list = [target_station] + stations
        X = np.vstack(tuples_list).T
        self.label = options.get("label")
        if options.get("diff"):
            pass
        if options.get("normalize"):
            pass

        return ViewDefinition(name=self.__class__.__name__,
                              label=self.label, x=X[:, 1:], y=X[:, 0:1:])
