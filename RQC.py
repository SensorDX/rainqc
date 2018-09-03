from view.view import ViewFactory
from model.hurdle_regression import MixLinearModel
class RQC(object):

    def __init__(self, target_station, variable='pr', num_k_stations=5, radius=100):
        self.target_station = target_station
        self.variable = variable
        self.view_registry = {}
        self.data_source = None
        self.num_k_stations = num_k_stations
        self.radius = radius
        self.view_object_list = None
        self.view_factory = ViewFactory()
    def add(self, type, name):
        # Add entity type and model name to the system
        if type=='View':
            self.add_view(name)
        elif type=='model':
            self.add_model(name)
    def add_view(self, view_name):
        self.view_registry.update({"name":view_name})
    def add_model(self, model_name=None):
        if model_name is None:
            self.model = MixLinearModel()
        else:
            self.model = model_name
    def build_view(self,date_from, date_to, **kwargs):
        """
        Given range of date, construct views from the given view names. Construct view for each stations.
        Args:
            date_from:
            date_to:
            **kwargs:

        Returns:

        """
        nearby_stations = self.data_source.nearby_station(self.target_station, self.num_k_stations, self.radius)
        station_data = {}
        query_data = lambda station_name: self.data_source.measurements(station_name, self.variable,
                                                                          date_from=date_from, date_to=date_to,
                                                                          group=kwargs.get('group'))

        station_data[self.target_station] = query_data(self.target_station)
        for station_name in nearby_stations:
            station_data[station_name] = query_data(station_name)

        for view in self.view_registry:
            vw = self.view_factory.create_view(view)
            vw.make_view(station_data.values)
            self.view_object_list[view] = vw
        return self.view_object_list

    def fit(self, separate_model=True):
        if separate_model:
            ## Train separate model for each station pair with the target stations.
            self.model.train()
        ## TODO: Add for training model of the separte pairwise operations.

    def evaluate(self):
        pass
    def score(self):
        pass

    ##TODO: Complete basic model.