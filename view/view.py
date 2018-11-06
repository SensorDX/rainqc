import numpy as np
from collections import defaultdict


def pairwise_view(target_station, next_station, mismatch='error'):
    if target_station is None or next_station is None:
        return ValueError("The data is empty.")
    if target_station.shape != next_station.shape:
        return None  # ValueError("Paired station mismatched")
    return ViewDefinition(y=target_station, x=next_station)


def multipair_view(target_station, stations):
    """

    Args:
        target_station:
        stations:

    Returns:

    """
    assert all(target_station.shape == n_station.shape for n_station in stations)
    dt = np.hstack(stations)
    return ViewDefinition(y=target_station, x=dt)


class View(object):
    def __init__(self):
        self.X = None
        self.y = None
        self.label = None
        self.view_metadata = defaultdict()

    def make_view(self, target_station, k_stations):
        return NotImplementedError
    def to_json(self):
        return NotImplementedError
    @classmethod
    def from_json(cls, json_file):
        return NotImplementedError

class ViewDefinition:
    """
    View definition format.
    """

    def __init__(self, name=None, label=None, x=None, y=None):
        self.name = name
        self.label = label
        if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
            self.x = x
            self.y = y

class ViewFactory:
    @staticmethod
    def get_view(view_type):
        if view_type == 'PairwiseView':
            return PairwiseView()


class PairwiseView(View):

    def __init__(self, variable=None):
        self.__name__ = "PairwiseView"
        super(PairwiseView, self).__init__()
        self.variable = variable

    def make_view(self, target_station, k_stations):
        len_series = target_station.shape[0]
        # Check dimension mismatch.
        if not all([len_series == value.shape[0] for value in k_stations.values()]):
            raise ValueError("Dimension mismatch b/n target station and one of the k stations")

        tuples_list = [target_station] + k_stations.values()
        dt = np.hstack(tuples_list)
        vw = ViewDefinition(name=self.__name__, label=k_stations.keys(),
                            x=dt[:, 1:], y=dt[:, 0:1])

        return vw

    def to_json(self):
        view_config = {"variable": self.variable}
        return view_config

    def from_json(cls, json_file):
        variable = json_file["variable"]
        pwv = PairwiseView(variable=variable)
        return pwv