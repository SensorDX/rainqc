

class RQC(object):

    def __init__(self, target_station, variable='pr', num_k_stations=5, radius=100):
        self.target_station = target_station
        self.variable = variable
        self.view_registry = {}
        self.data_source = None
        self.num_k_stations = num_k_stations
        self.radius = radius


    def add(self, type, name):
        # Add entity type and model name to the system
        if type=='View':
            self.add_view(name)
        elif type=='model':
            self.add_model(name)
    def add_view(self, view_name):
        self.view_registry.update({"name":view_name})
    def add_model(self, model_name):
        self.model = model_name

    def build_view(self,date_from, date_to, **kwargs):
        nearby_stations = self.data_source.nearby_station(self.target_station, self.num_k_stations, self.radius)
        station_data = {}
        query_data = lambda station_name: self.data_source.measurements(station_name, self.variable,
                                                                          date_from=date_from, date_to=date_to,
                                                                          group=kwargs.get('group'))

        station_data[self.target_station] = query_data(self.target_station)
        for station_name in nearby_stations:
            station_data[station_name] = query_data(station_name)



    def fit(self, sensor, date_range):
        pass
    def evaluate(self):
        pass
    def score(self):
        pass