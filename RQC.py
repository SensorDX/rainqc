from collections import OrderedDict, defaultdict

from model.hurdle_regression import ModelFactory
from view.view import ViewFactory, ViewDefinition


def pairwise_view(target_station, next_station, mismatch='error'):
    ## make sure the stations mismatch each  other

    if len(target_station) != len(next_station):
        return None #ValueError("Paired station mismatched")
    return ViewDefinition(y=target_station, x=next_station)

def evaluate(model_list, sample_data):
    pass
class RQC(object):
    data_source = None
    def __init__(self, target_station=None, variable='pr', num_k_stations=5, radius=100, data_source=None):
        data_source = data_source
        self.target_station = target_station
        self.variable = variable
        self.view_registry = {}
        self.num_k_stations = num_k_stations
        self.radius = radius
        #self.view_object_list = {}
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
        self.view_registry.update({"view": view_name})

    def add_model(self, model_name=None):
        self.model = model_name
    @classmethod
    def __fetch_data(cls,target_station,variable, date_from, date_to, **kwargs):
        fetched_data = cls.data_source.measurements(target_station, variable,
                                                                        date_from=date_from, date_to=date_to,
                                                                        group=kwargs.get('group'))
        if fetched_data:
            fetched_data = fetched_data[variable].as_matrix()
        else:
            return ValueError("Target source is empty")

        return fetched_data


    def build_view(self, target_station, date_from, date_to, **kwargs):
        """
        This should load all the views added with its metadata.
        Given range of date, construct views from the given view names. Construct view for each stations.
        Args:
            date_from:
            date_to:
            **kwargs:

        Returns:
        """
        station_list = kwargs.get("station_list")
        if station_list is None:
            nearby_stations = self.data_source.nearby_stations(target_station, self.num_k_stations, self.radius)
        else:
            nearby_stations = station_list

        view_list = OrderedDict()
        query_data = lambda station_name: self.data_source.measurements(station_name, self.variable,
                                                                        date_from=date_from, date_to=date_to,
                                                                        group=kwargs.get('group'))[self.variable].as_matrix()

        target_station = query_data(target_station) #self.__fetch_data(target_station, self.variable, date_from, date_to, **kwargs)  #query_data(target_station)

        if nearby_stations is None or target_station is None:
            return NameError("There are no available nearby stations.")

        for station_name in nearby_stations:
            view_list[station_name] = pairwise_view(target_station, query_data(station_name))


        #
        #
        # for view in self.view_registry.values():
        #     vw = self.view_factory.create_view(view)
        #     self.view_object_list[view] = vw.make_view(station_data[target_station],
        #                                                station_data.values()[1:], label=station_data.keys())

        return view_list

    def fit_from_view(self, view_list):
        model_list = defaultdict()
        for vw_id in view_list:
            current_vwd = view_list[vw_id]
            if current_vwd is not None:
                model_list[vw_id] = self.model_factory.create_model(self.model).fit(current_vwd.x, current_vwd.y, True)
        self.model_list = model_list
        return model_list

    def fit(self, target_station, date_from, date_to, **kwargs):

        # Build view of the data, using the added view
        # For each view build a model added to the system or build view add to separate views.


        view_list = self.build_view(target_station, date_from, date_to, **kwargs)
        return self.fit_from_view(view_list)
            # train model for each view.
        # pairwise = kwargs.get('pairwise')
        # model_registry = {}
        # pairwise_view = view_list['PairwiseView']
        # if pairwise:
        #     ## Train separate model for each station pair with the target stations.
        #
        #     for i in range(1, pairwise_view.x.shape[1]):
        #         #print pairwise_view.x[1:10,:]
        #         model_registry[i] = self.model_factory.create_model(self.model).\
        #             fit(pairwise_view.x[:,[i]], pairwise_view.y)
        # else:
        #     model_registry = self.model_factory.create_model(self.model).train(pairwise_view.y, pairwise_view.x)
        #
        # return model_registry
        ## TODO: Add for training model of the separte pairwise operations.

    def evaluate(self):
        pass

    def score(self, model_registry, target_station, date_from, date_to, **kwargs):
        view_object_list = self.build_view(target_station, date_from, date_to, **kwargs)
        return self.score_from_view(model_registry=model_registry, view_object_list=view_object_list)
    def score_from_view(self, model_registry, view_object_list):
        scores = {}
        for vw_name in view_object_list:
            current_vw = view_object_list[vw_name]
            if current_vw:
                model = model_registry[vw_name]
                if model:
                    scores[vw_name] = model.predict(current_vw.x, current_vw.y)

        return scores


if __name__ == '__main__':
    from datasource.toy_datasource import ToyDataSource
    from datasource.data_source import LocalDataSource
    rqc = RQC()
    rqc.data_source = LocalDataSource(dir_path='./localdatasource') #ToyDataSource
    rqc.add_view('PairwiseView')
    rqc.add_model('MixLinear')
    target_station = 'TA00025'
    vw = rqc.build_view(target_station=target_station, date_from='2017-01-01', date_to='2017-05-01', group='D')
    #fitted_model = rqc.fit(target_station=target_station, date_from='2010-01-01', date_to='2010-10-01', group='D',
     #       pairwise=True)
    fitted_model = rqc.fit_from_view(vw)
    #print fitted_model.keys()
    import numpy as np
   # print fitted_model.keys()
    #print len(vw), vw.keys(), [vv.x.shape for vv in vw.values()]
    score = rqc.score_from_view(fitted_model, vw)

    #score = rqc.score(model_registry=fitted_model, target_station=target_station, date_from='2010-01-01', date_to='2010-10-01', group='D',
    #       pairwise=True, station_list =fitted_model.keys())
    ll = {}
    ff = {}
    for key in score:
        ll[key]= -np.sum(np.log(score[key]))
        #ff[key] = -np.sum(np.log(fitted_model[key]))
    print ll
    #print score.values()
    #print vw.values()[0].x
