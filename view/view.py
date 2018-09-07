import numpy as np
import pandas as pd
from collections import defaultdict

class View(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.label = None
        self.view_metadata = defaultdict()



    def make_view(self, target_station, stations, **kwargs):
        return NotImplementedError


class ViewDefinition:
    """
    View definition format.
    """

    def __init__(self, name=None, label=None, x=None, y=None):
        self.name = name
        self.label = label
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            return NameError("The input should be given as ndarray")

        self.x = x.reshape(-1,1)
        self.y = y.reshape(-1,1)
    # def transform(self, x):
    #     if options.get("diff"):
    #         pass
    #     if options.get("normalize"):
    #         pass
    #     pass

class ViewFactory:
    @staticmethod
    def create_view(view_type):
        if view_type == 'PairwiseView':
            return PairwiseView()

class PairwiseView(View):

    def __init__(self, variable=None):
        self.__name__ = "PairwiseView"
        super(PairwiseView, self).__init__()
        self.variable = variable

    def make_view(self, target_station, covariate_stations, **options):
        """

        Args:
            target_station: Nx1 numpy array
            covariate_stations: list of Nx1 numpy array
            **options: key,value optional

        Returns: Nxlen(sations) numpy matrix.

        """
        #print target_station.shape ,
        len_series = target_station.shape[0]
        # Check dimension mismatch.
        if all(len_series == stn.shape[0] for stn in covariate_stations):
            return ValueError("Dimension mismatch b/n target station and covariate covariate_stations")

        tuples_list = [target_station] + covariate_stations
        print [xx.shape for xx in tuples_list]
        dt = np.hstack(tuples_list)
        self.x, self.y = dt[:,1:], dt[:, 0:1]
        self.label = options.get("label")
        print dt.shape
        return self


