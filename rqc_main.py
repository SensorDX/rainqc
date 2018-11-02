from common.weather_variable import RAIN
from model.models import Module
from view.view import View, ViewDefinition
from collections import OrderedDict
import datetime
from datasource.tahmo_datasource import TahmoDataSource


class MainRQC:
    """"
    Perform main RQC operations.

    Args:
         target_station (str): Target station to perform QC operation.

    """

    def __init__(self, target_station=None, variable=RAIN, num_k_stations=5, radius=100, data_source=None):
        """

        Args:
            target_station (str
            variable (str): Weather variable code
            num_k_stations:
            radius:
            data_source:
        """
        self.data_source = data_source
        self.target_station = target_station
        self.variable = variable
        self.view_registry = OrderedDict()
        self.num_k_stations = num_k_stations
        self.radius = radius
        self._modules = OrderedDict()
        self.training = True
        self.parameters = OrderedDict()
        self.views = OrderedDict()

    def add_module(self, name, module):
        if not isinstance(module, Module) and module is not None:
            raise TypeError("{} is not a Module subclass".format(
                type(module)))
        elif hasattr(self, name) and name not in self._modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self._modules[name] = module

    def add_view(self, name, view):
        if not isinstance(view, View) and view is not None:
            raise TypeError("{} is not a view class".format(type(view)))
        elif hasattr(self, name) and name not in self.views:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("view name can't contain \".\"")
        self.views[name] = view

    def available_neighbors(self, target_station, start_date, end_date):
        """
        Get available neigbhor stations.
        Args:
            target_station:

        Returns:

        """
        #TODO: handle start_date and end_date
        nearby_station_list = self.data_source.nearby_stations(target_station, self.radius)
        available_stations = {}
        k = self.num_k_stations
        for stn in nearby_station_list:
            if self.data_source.check_data(stn, self.variable, start_date, end_date):
                available_stations[stn] = self.data_source.get_data(stn, self.variable, start_date, end_date)
                k -= 1
            if k < 1:
                break
        if len(available_stations) < 1:
            raise ValueError("No available station in the given range");
        return available_stations

        # check if the stations are active and have data for the given range date.

    #
    def fit(self, date_from, date_to, **kwargs):
        """
        Fit the RQC module using the data from the give date range.
        1. Fetch data from db
        2. get_nearby station.
        3. Create view using the stations.
        4. If things passed, train the model.
        Args:
            date_from:
            date_to:
            **kwargs:

        Returns: self, fitted class of RQC.

        """

        target_station_data = self.data_source.daily_data(self.target_station, self.variable, date_from, date_to)
        k_stations = self.data_source.nearby_stations(target_station=self.target_station, radius=self.radius,
                                                      k=self.num_k_stations)

        active_station = self.data_source.get_active_station(k_stations.keys())

        for station in active_station:
            vw = make_view(target_station_data, k_stations)
        self._modules[0].train(vw.x, vw.y)
        return self
    #
    # def fetch_data(self, target_station, variable, date_from, date_to, check_size=True, **kwargs):
    #     fetched_data = self.data_source.measurements(target_station, variable,
    #                                                  date_from=date_from, date_to=date_to,
    #                                                  group=kwargs.get('group'))
    #     if fetched_data:
    #         fetched_data = fetched_data[variable].as_matrix()
    #     else:
    #         return ValueError("Target source is empty")
    #     if check_size:
    #         date_diff = (date_to - date_from).days()
    #         if fetched_data.shape[0] != date_diff:
    #             return ValueError("Mismatch in the date and returned data")
    #
    #     return fetched_data
    #
    # def build_view(self, target_station, date_from, date_to, **kwargs):
    #     """
    #     This should load all the views added with its metadata.
    #     Given range of date, construct views from the given view names. Construct view for each stations.
    #     Args:
    #         date_from:
    #         date_to:
    #         **kwargs:
    #
    #     Returns:
    #     """
    #     station_list = kwargs.get("station_list")
    #     if station_list is None:
    #         nearby_stations = self.data_source.nearby_stations(target_station, self.radius)
    #     else:
    #         nearby_stations = station_list
    #
    #     view_list = OrderedDict()
    #     query_data = lambda station_name: self.data_source.measurements(station_name, self.variable,
    #                                                                     date_from=date_from, date_to=date_to,
    #                                                                     group=kwargs.get('group'))[
    #         self.variable].as_matrix()
    #
    #     target_station_data = self.__fetch_data(target_station, self.variable, date_from, date_to,
    #                                             group=kwargs.get("group"))  # query_data(target_station)
    #
    #     if nearby_stations is None or target_station is None:
    #         return ValueError("There are no available nearby stations for {}".format(target_station))
    #
    #     # Fetch data from nearby station and make view.
    #
    #     view_list = [view.make_view(target_station_data, nearby_stations_data) for view in self.views]
    #
    #     # for station_name in nearby_stations:
    #     #    view_list[station_name] = pairwise_view(target_station, query_data(station_name))
    #
    #     return view_list
    #
    # def fit_from_view(self, view_list):
    #     model_list = defaultdict()
    #     # model_list2 = defaultdict()
    #     for vw_id in view_list:
    #         current_vwd = view_list[vw_id]
    #         if current_vwd is not None:
    #             model_list[vw_id] = self.model_factory.create_model(self.model).fit(current_vwd.x, current_vwd.y, True)
    #             # model_list2[vw_id] = self.model_factory.create_model(self.model).fit(current_vwd.y, current_vwd.x, True)
    #
    #     self.model_list = model_list
    #     return model_list  # , model_list2)
    #
    # #
    # # def fit(self, target_station, date_from, date_to, **kwargs):
    # #
    # #     # Build view of the data, using the added view
    # #     # For each view build a model added to the system or build view add to separate views.
    # #
    # #
    # #     view_list = self.build_view(target_station, date_from, date_to, **kwargs)
    # #     return self.fit_from_view(view_list)
    # #     # train model for each view.
    # #     # pairwise = kwargs.get('pairwise')
    # #     # model_registry = {}
    # #     # pairwise_view = view_list['PairwiseView']
    # #     # if pairwise:
    # #     #     ## Train separate model for each station pair with the target stations.
    # #     #
    # #     #     for i in range(1, pairwise_view.x.shape[1]):
    # #     #         #print pairwise_view.x[1:10,:]
    # #     #         model_registry[i] = self.model_factory.create_model(self.model).\
    # #     #             fit(pairwise_view.x[:,[i]], pairwise_view.y)
    # #     # else:
    # #     #     model_registry = self.model_factory.create_model(self.model).train(pairwise_view.y, pairwise_view.x)
    # #     #
    # #     # return model_registry
    # #     ## TODO: Add for training model of the separte pairwise operations.
    #
    # def resultant_score(self, model_1, model_2):
    #     pass
    #
    # def score(self, model_registry, target_station, date_from, date_to, **kwargs):
    #     view_object_list = self.build_view(target_station, date_from, date_to, **kwargs)
    #     return self.score_from_view(model_registry=model_registry, view_object_list=view_object_list)
    #
    # def score_from_view(self, model_registry, view_object_list):
    #     scores = {}
    #     for vw_name in view_object_list:
    #         current_vw = view_object_list[vw_name]
    #         if current_vw:
    #             model = model_registry[vw_name]
    #             if model:
    #                 scores[vw_name] = model.predict(current_vw.x, current_vw.y)
    #
    #     return scores
    #
    # def save(self, path_name):
    #     """
    #     Save the QC model.
    #     Args:
    #         path_name: path to the save the models.
    #
    #     Returns:
    #
    #     """
    #     pass
    #
    # @classmethod
    # def load(cls, path_name):
    #     """
    #     Load trained RQC model.
    #     Args:
    #         path_name:
    #
    #     Returns: RQC model.
    #
    #     """
    #     # make sure all available models are saved.
    #     pass


if __name__ == "__main__":
    x = 2
    dd = MainRQC()
