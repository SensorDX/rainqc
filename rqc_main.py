from common.weather_variable import RAIN
from view.view import View
from model import Module, MixLinearModel
from collections import OrderedDict
import datetime
from dateutil import parser, tz
from pytz import utc, timezone
from datasource.tahmo_datasource import TahmoDataSource
from datasource.FakeTahmo import FakeTahmo


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
        self.module_registry = OrderedDict()
        self.num_k_stations = num_k_stations
        self.radius = radius
        self._modules = {} #OrderedDict()
        self.training = True
        self.parameters = OrderedDict()
        self.views = {} #OrderedDict()
        self.k_stations = None

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
            raise ValueError("No available station in the given range")
        return available_stations

        # check if the stations are active and have data for the given range date.

    def fetch_data(self, start_date, end_date, k_station_list=None):

        k_stations_data = {}
        target_station_data = self.data_source.daily_data(self.target_station, self.variable, start_date, end_date)
        if target_station_data is None:
            return ValueError("Target station has no data")
        if k_station_list is None:
            k_stations = self.data_source.nearby_stations(target_station=self.target_station, radius=self.radius)
        else:
            assert isinstance(k_station_list, list)
            assert len(k_station_list) > 0
            k_stations = k_station_list

        nrows = target_station_data.shape[0]
        active_date = utc.localize(parser.parse(end_date))
        active_station = self.data_source.active_stations(k_stations, active_day_range=active_date)
        if len(active_station) < 1:
            return ValueError("There is no active station to query data")

        k = self.num_k_stations  # Filter k station if #(active stations) > k
        for stn in active_station:
            if k < 1:
                break
            current_data = self.data_source.daily_data(stn, self.variable, start_date, end_date)
            if current_data is None or (current_data.shape[0] != nrows):
                continue
            k_stations_data[stn] = current_data
            k -= 1
        if len(k_stations_data.keys()) < 1:
            print("All of the active station don't have data starting date {} to {}.".format(start_date, end_date))
            return
        else:
            print ("There are {} available stations to use".format(k_stations_data.keys()))
        return target_station_data, k_stations_data

    def fit(self, start_date, end_date, **kwargs):
        """
        Fit the RQC module using the data from the give date range.
        1. Fetch data from db
        2. get_nearby station.
        3. Create view using the stations.
        4. If things passed, train the model.
        Args:
            start_date:
            end_date:
            **kwargs:

        Returns: self, fitted class of RQC.

        """
        #assert parser.parser(start_date)<=parser.parser(end_date)
        assert len(self.views) > 0
        assert len(self._modules) > 0

        target_station, k_station = self.fetch_data(start_date, end_date, k_station_list=None)

        if len(self.views) > 1:
            for vw_name, vw in self.views:
                self.view_registry[vw_name] = vw.make_view(target_station, k_station)

        elif len(self.views) == 1:
            # Create a single view
            vw_name, vw = self.views.keys()[0], self.views.values()[0]
            self.view_registry[vw_name] = vw.make_view(target_station, k_station)
        else:

            return ValueError("View object is not added")

        # Train the model.
        # Get the current view.
        vw = self.view_registry[vw_name]
        print (vw.x.shape, vw.y.shape)
        if len(self._modules) > 1:
            for name, module in self._modules:
                self.module_registry[name] = module.fit(vw.x, vw.y)
        else:

            name, module = self._modules.keys()[0], self._modules.values()[0]
            self.module_registry[name] = module.fit(x=vw.x, y=vw.y)

        return self

    def score(self, target_station, start_date, end_date):
        """
        1. Fetch data from source.
        2. Load nearby station, from saved model.
        3. Create view using the nearby stations.
        4. Predict using the trained model

        Args:
            model_registry:
            target_station:
            start_date:
            end_date:
            **kwargs:

        Returns:

        """
        target_station, k_station_data = self.fetch_data(start_date, end_date, k_station_list=self.k_stations)
        if len(self.view_registry)<1:
            return ValueError("No valid view found.")
        if len(self._modules)<1:
            return ValueError("No available module found.")

        vw = self.view_registry[vw_name]
        print (vw.x.shape, vw.y.shape)
        if len(self._modules) > 1:
            for name, module in self._modules:
                self.module_registry[name] = module.fit(vw.x, vw.y)
        else:

            name, module = self._modules.keys()[0], self._modules.values()[0]
            self.module_registry[name] = module.fit(x=vw.x, y=vw.y)

        #pass
     #view_object_list = self.build_view(target_station, start_date, end_date, **kwargs)
     #return self.score_from_view(model_registry=model_registry, view_object_list=view_object_list)


# def build_view(self, target_station, start_date, end_date, **kwargs):
    #     """
    #     This should load all the views added with its metadata.
    #     Given range of date, construct views from the given view names. Construct view for each stations.
    #     Args:
    #         start_date:
    #         end_date:
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
    #                                                                     start_date=start_date, end_date=end_date,
    #                                                                     group=kwargs.get('group'))[
    #         self.variable].as_matrix()
    #
    #     target_station_data = self.__fetch_data(target_station, self.variable, start_date, end_date,
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
    # # def fit(self, target_station, start_date, end_date, **kwargs):
    # #
    # #     # Build view of the data, using the added view
    # #     # For each view build a model added to the system or build view add to separate views.
    # #
    # #
    # #     view_list = self.build_view(target_station, start_date, end_date, **kwargs)
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
    # def score(self, model_registry, target_station, start_date, end_date, **kwargs):
    #     view_object_list = self.build_view(target_station, start_date, end_date, **kwargs)
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
    from view.view import PairwiseView
    from model.hurdle_regression import MixLinearModel
    x = 2
    data_source = FakeTahmo(local_data_source="experiments/dataset/tahmostation2016.csv",
                            nearby_station="localdatasource/nearest_stations.csv")
    tahmo_datasource = TahmoDataSource(nearby_station_location="datasource/station_nearby.json")
    target_station = "TA00030"
    start_date ="2016-01-01" #(datetime.datetime.now(timezone('utc'))-datetime.timedelta(days=50)).strftime('%Y-%m-%dT%H:%M')
    end_date ="2016-06-30" #(datetime.datetime.now(timezone('utc')) - datetime.timedelta(days=40)).strftime('%Y-%m-%dT%H:%M')
    dd = MainRQC(data_source=data_source,
                 target_station=target_station, radius=200)
    dd.add_view(name="pairwise", view=PairwiseView())
    dd.add_module(name="MixLinearModel", module=MixLinearModel())
    fitted = dd.fit(start_date=start_date,
           end_date=end_date)
    print(fitted)


## TODO: Work on downloaded data, the bluemix data is unreliable.{ The downloaded data is not also consistent}
## TODO: Work on synthetic data, and make sure the algorithm can be deployed and tested. Sample rainfall data or weather data, from a given
## Station and create a perfect data that can work with the algorithm.
