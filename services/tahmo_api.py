import requests
import json
from data_source import DataSource
authorization = json.load(open("../util/config.json","r"))
url="https://tahmoapi.mybluemix.net"

class TahmoAPI(DataSource):


    @staticmethod
    def station_list():
        stations_api_url = url + "/v1/stations"
        return TahmoAPI().get_data(stations_api_url)



    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):
        return super(TahmoAPI, site_code).nearby_stations(site_code, k, radius)

    @staticmethod
    def measurements(station_name, variable, date_from, date_to, **kwargs):
        measurement_api = url + "/stations/" + station_name
        return TahmoAPI().get_data(measurement_api)


    def __init__(self):
        self.user_id = authorization['tahmo_api']['id']
        self.secret = authorization['tahmo_api']['secret']

    def get_data(self, station_api_url):
        try:

            s = requests.Session()
            s.auth = (self.user_id, self.secret)
            s.headers.update({'Authorization': 'Basic'})
            req = s.get(station_api_url)
            if req.status_code==200:
               return req.json()
            else:
                return ValueError("Request not completed")
        except  Exception as e:
            print e.message

    def get_measurement(self, station_id, start_date, end_date, sensor):
        #"url":"https://tahmoapi.mybluemix.net/v1/stations/" + stationId, "headers": headers
        measurement = url+"/stations/"+station_id

if __name__ == '__main__':
    target_station = 'TA0020'
    #stn_list = TahmoAPI.station_list()
    #print (stn_list)
    measure = TahmoAPI.measurements(station_name=target_station, variable='pr',
                            date_from='2017-01-01 00:00:00', date_to='2017-01-11 00:00:00',
                            group='D')
    print measure