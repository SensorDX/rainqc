import requests
import json

authorization = json.load(open("config.json","r"))
url="https://tahmoapi.mybluemix.net"

class TahmoAPI:

    def __init__(self):
        self.user_id = authorization['tahmo_api']['id']
        self.secret = authorization['tahmo_api']['secret']

    def get_list_stations(self):
        s = requests.Session()
        s.auth = (self.user_id, self.secret)
        s.headers.update({'Authorization': 'Basic'})
        stations_api = url+"/v1/stations"
        # both 'x-test' and 'x-test2' are sent
        req = s.get(stations_api)
        if req.status_code==200:
           return req.json()
        else:
            return ValueError("Request not completed")
    def get_measurement(self, station_id, start_date, end_date, sensor):
        #"url":"https://tahmoapi.mybluemix.net/v1/stations/" + stationId, "headers": headers
        measurement_api = url+"/stations/"+station_id
if __name__ == '__main__':
    tt = TahmoAPI()
    print tt.get_list_stations()
