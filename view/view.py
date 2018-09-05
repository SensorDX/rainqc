import numpy as np
import pandas as pd


class View(object):
    def __init__(self, variable="pr"):
        self.X = None
        self.y = None
        self.label = None

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
        tuples_list = [target_station]
        #for stn in stations:
         #   tuples_list.append(stn)
        tuples_list = [target_station] + stations
        X = np.vstack(tuples_list).T

        if options.get("diff"):
            pass
        if options.get("normalize"):
            pass

        label = options.get('label')
        if options.get('split'):
            return [ViewDefinition(name=self.__class__.__name__, label=label, x=X[:, [i]], y=X[:, 0:1:])
                    for i in X.shape[1]]
        return ViewDefinition(name=self.__class__.__name__,
                              label=label, x=X[:, 1:], y=X[:, 0:1:])

    #
    # def make_view(self, target_station, date_from, date_to):
    #
    #     t_station = self.data_source.measurements(target_station, self.variable, date_from=date_from, date_to=date_to,
    #                                               group='D')
    #
    #     self.y, self.label = t_station[self.variable].values.reshape(-1, 1), t_station['date']  # .values.reshape(-1, 1)
    #
    #     k_nearest_station = self.data_source.nearby_stations(target_station, self.num_k_station, self.radius)
    #     print k_nearest_station
    #     stn_list = []
    #     for k_station in k_nearest_station:
    #          k_station_data = self.data_source.measurements(k_station, self.variable, date_from=date_from, date_to=date_to,
    #                                               group='D')
    #          k_station_data = k_station_data[self.variable].values.reshape(-1,1)
    #
    #          if k_station_data.shape[0] == self.y.shape[0]:
    #             stn_list.append(k_station_data)
    #     if len(stn_list) < 2:
    #         return NameError("There are less than 2 stations with equal number of observation as the target station.")
    #     self.X = np.hstack(stn_list)
    #     df = pd.concat([pd.DataFrame(self.label), pd.DataFrame(self.y), pd.DataFrame(self.X)], axis=1)
    #     pd.DataFrame.dropna(df, inplace=True)
    #     self.label, self.y, self.X = df.iloc[:,0], df.iloc[:,1], df.iloc[:,2:]
