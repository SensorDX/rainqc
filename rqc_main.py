from definition import RAIN
from view.view import View, ViewFactory
from model import Module, ModelFactory
from collections import OrderedDict
from dateutil import parser
from pytz import utc
from datasource.abcdatasource import DataSource
from datasource.fake_tahmo import FakeTahmo
from common.utils import is_valid_date


class MainRQC(object):
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
            num_k_stations (int): Number of nearest stations
            radius (int): radius in km
            data_source (DataSource): interface for querying data source
        """
        self.data_source = data_source
        self.target_station = target_station
        self.variable = variable
        self.view_registry = {}
        self.module_registry = {}
        self.num_k_stations = num_k_stations
        self.radius = radius
        self._modules = {}  # OrderedDict()
        self.training = True
        self._parameters = OrderedDict()
        self._views = {}  # OrderedDict()
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
        elif hasattr(self, name) and name not in self._views:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("view name can't contain \".\"")
        self._views[name] = view

    def set_parameters(self):
        self._parameters['data_source'] = type(self.data_source).__name__
        self._parameters['target_station'] = self.target_station
        self._parameters['num_k_stations'] = self.num_k_stations
        self._parameters["radius"] = self.radius
        self._parameters['variable'] = self.variable
        self._parameters["_views"] = self._views.keys()
        self._parameters["_models"] = self._modules.keys()
        self._parameters["k_stations"] = self.k_stations
        #return self._parameters

    # def set_parameters(self, _parameters):
    #
    #     for key, value in _parameters.items():
    #         if key in self.__dict__:
    #             if key == "data_source":
    #                 continue
    #             setattr(self, key, value)
    #             if key in self._parameters:
    #                 continue
    #             self._parameters[key] = value
    #         else:
    #             continue
    #



    def fetch_data(self, start_date, end_date, target_station=None, k_station_list=None):

        if target_station is None:
            target_station = self.target_station

        k_stations_data = {}
        target_station_data = self.data_source.daily_data(target_station, self.variable, start_date, end_date)
        if target_station_data is None:
            return ValueError("Target station has no data")
        if k_station_list is None:
            k_stations = self.data_source.nearby_stations(target_station=target_station, radius=self.radius)
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

    def make_view(self, target_station_data, k_station_data):
        """

        Args:
            target_station_data (ndarray): array of readings.
            k_station_data (dict(str:ndarray): dict of ndarray of nearby station readings.

        Returns:

        """
       # trained_views = {}
        if len(self._views) > 1:
            for vw_name, vw in self._views:
                self.view_registry[vw_name] = vw.make_view(target_station_data, k_station_data)
            return self.view_registry.keys(), self.view_registry.values()
        elif len(self._views) == 1:
            # Create a single view
            vw_name, vw = self._views.keys()[0], self._views.values()[0]
            self.view_registry[vw_name] = vw.make_view(target_station_data, k_station_data)
            return vw_name, self.view_registry[vw_name]
        else:

            raise ValueError("View object is not added")

    def fit(self, start_date, end_date, **kwargs):
        """
        Fit the RQC module using the data from the give date range.
        1. Fetch data from db
        2. get_nearby station.
        3. Create view using the stations.
        4. If all passed, fit the model defined at self._modules
        Args:
            start_date:
            end_date:
            **kwargs:

        Returns: self, fitted class of RQC.

        """

        is_valid_date(start_date, end_date)
        assert len(self._views) > 0
        assert len(self._modules) > 0

        target_station_data, k_station_data = self.fetch_data(start_date, end_date, k_station_list=None)

        vw_name, vw = self.make_view(target_station_data, k_station_data)

        print (vw.x.shape, vw.y.shape)
        if len(self._modules) > 1:
            for name, module in self._modules:
                self.module_registry[name] = module.fit(vw.x, vw.y)
        else:

            name, module = self._modules.keys()[0], self._modules.values()[0]
            self.module_registry[name] = module.fit(x=vw.x, y=vw.y)

        return self

    def score(self, start_date, end_date, target_station=None):
        """
        1. Fetch data from source.
        2. Load nearby station, from saved model.
        3. Create view using the nearby stations.
        4. Predict using the trained model at self.modules_registry

        Args:
            model_registry:
            target_station:
            start_date:
            end_date:
            **kwargs:

        Returns:

        """

        is_valid_date(start_date, end_date)
        if target_station is None:
            target_station = self.target_station
        target_station_data, k_station_data = self.fetch_data(start_date, end_date, k_station_list=self.k_stations,
                                                              target_station=target_station)
        if len(self._views) < 1:
            return ValueError("No valid view definition found.")
        if len(self.module_registry) < 1:
            return ValueError("No available fitted module found.")

        # Check how many view definition are available.
        vw_name, vw = self.make_view(target_station_data, k_station_data)
        scores = {}
        print (vw.x.shape, vw.y.shape)
        if len(self.module_registry) > 1:
            for name, module in self._modules:
                scores[name] = module.predict(x=vw.x, y=vw.y)
            return scores
        else:

            name, module = self.module_registry.keys()[0], self.module_registry.values()[0]
            result = module.predict(x=vw.x, y=vw.y)
            return {name: result}

    def save(self, path_name=None):
        """
        Get all fitted modules and _parameters.
        Items to save
        return JSON representation of the model.
        -------------
        1. Predictive model and its parameter
        2. View object and its parameter
        3. Various parameter
        Args:
            path_name: path to the save the models.

        Returns:

        """
        # Assume the model is single for now.
        # May be use globals()[className)(constructor) for creating class from string.
        rqc_config = {"models":{}, "views":{}}
        self.set_parameters()
        for model_name, model in self.module_registry.items():
            rqc_config["models"][model_name] = model.to_json()
        for view_name, view in self.view_registry.items():
            rqc_config['views'][view_name] = view.name
        rqc_config["parameters"] = self._parameters
        rqc_config["data_source"] = {type(self.data_source).__name__: self.data_source.to_json()}

        return rqc_config

    @classmethod
    def load(cls, rqc_config):
        """
        Load trained RQC model.
        Args:
            path_name (dict): Serialized model representation.

        Returns: RQC model.

        """
        # make sure all available models are saved.
        # make sure it is valid json format and have all model.
        rqc = MainRQC()
        #rqc.set_parameters(rqc_config['parameters'])
        # set parameters
        print rqc_config["parameters"]
        for param, value in rqc_config["parameters"].items():
            if param in rqc.__dict__:
                if param =='views':
                    continue
                setattr(rqc, param, value)


        ds_name, ds_config = next(rqc_config["data_source"].iteritems())
        #rqc.data_source = globals()[ds_name].from_json(ds_config)
        rqc.data_source = FakeTahmo.from_json(ds_config)
        for model_name, model_config in rqc_config["models"].items():
            rqc.module_registry[model_name] = ModelFactory.get_model(model_name).from_json(model_config)
        for view_name, view_config in rqc_config["views"].items():
            rqc._views[view_name] = ViewFactory.get_view(view_name)

        return rqc

        #module_registry.append(rqc_config["model_config"])

    #TODO: Fix the error with the view saving & loading operations, datasource loading.
