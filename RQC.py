from view.view import ViewFactory
from model.hurdle_regression import ModelFactory
from services.toy_datasource import ToyDataSource
class RQC(object):

    def __init__(self, target_station=None, variable='pr', num_k_stations=5, radius=100):
        self.target_station = target_station
        self.variable = variable
        self.view_registry = {}
        self.data_source = None
        self.num_k_stations = num_k_stations
        self.radius = radius
        self.view_object_list = {}
        self.view_factory = ViewFactory()
        self.model_factory = ModelFactory()

    def add(self, type, name):
        # Add entity type and model name to the system
        if type=='View':
            self.add_view(name)
        elif type=='model':
            self.add_model(name)
        else:
            return
    def add_view(self, view_name):
        self.view_registry.update({"name": view_name})

    def add_model(self, model_name=None):
        self.model = model_name


    def build_view(self, target_station, date_from, date_to, **kwargs):
        """
        Given range of date, construct views from the given view names. Construct view for each stations.
        Args:
            date_from:
            date_to:
            **kwargs:

        Returns:

        """
        nearby_stations = self.data_source.nearby_stations(target_station, self.num_k_stations, self.radius)
        station_data = {}
        ## check if data arrives from all the stations.
        ## Assume for now data is available from the stations.
        query_data = lambda station_name: self.data_source.measurements(station_name, self.variable,
                                                                        date_from=date_from, date_to=date_to,
                                                                        group=kwargs.get('group'))[self.variable].as_matrix()

        station_data[target_station] = query_data(target_station)
        if len(nearby_stations)<1:
            return NameError("There are no available nearby stations.")

        for station_name in nearby_stations:
            station_data[station_name] = query_data(station_name)

        for view in self.view_registry.values():
            vw = self.view_factory.create_view(view)
            self.view_object_list[view] = vw.make_view(station_data[target_station],
                                                       station_data.values()[1:])
        return self.view_object_list

    def build_model(self, model, view_object, pairwise=True):
        pass

    def fit(self, target_station, date_from, date_to, **kwargs):

        # Build view of the data, using the added view
        # For each view build a model added to the system or build view add to separate views.
        pairwise = kwargs.get('pairwise')
        view_object = self.build_view(target_station, date_from, date_to, **kwargs)
        model_registry = {}
        pairwise_view = view_object['PairwiseView']
        if pairwise:
            ## Train separate model for each station pair with the target stations.

            for i in range(1, pairwise_view.x.shape[1]):
                print pairwise_view.x[1:10,:]
                model_registry[i] = self.model_factory.create_model(self.model).\
                    fit(pairwise_view.x[:,[i]], pairwise_view.y)
        else:
            model_registry = self.model_factory.create_model(self.model).train(pairwise_view.y, pairwise_view.x)

        return model_registry
        ## TODO: Add for training model of the separte pairwise operations.

    def evaluate(self):
        pass
    def score(self, model_registry, target_station, date_from, date_to, **kwargs):
        pass



if __name__ == '__main__':
    from services.toy_datasource import ToyDataSource
    from services.data_source import LocalDataSource
    rqc = RQC()
    rqc.data_source = ToyDataSource
    rqc.add_view('PairwiseView')
    rqc.add_model('MixLinear')
    #vw = rqc.build_view(target_station='TA0001', date_from='2010-01-01', date_to='2010-02-01', group='D')
    rqc.fit(target_station='TA00005', date_from='2010-01-01', date_to='2010-10-01', group='D',
            pairwise=True)

    #print vw.values()[0].x
