import requests
from local_source import DataSource
import urllib, urllib2, base64, json, datetime

# Fill in your API credentials
API_ID = "2ERQMSTZ6DFMP2GIFW72YDOJB" #'3HUOQWVK0WDIH3N1I4CZXVESB'
API_SECRET = "brnDWaLJe6KTDSK4rdtqdJ6YxF4sDHzLR52ul7ocFMg" #'sZTgiwMeeAtGRnvtmHGKAEYYs+Rr/uBh1djYAxiUwoM'

# Generate base64 encoded authorization string
basicAuthString = base64.encodestring('%s:%s' % (API_ID, API_SECRET)).replace('\n', '')



# Function to request data from API
def apiRequest(url, params={}):
    # Encode optional parameters
    encodedParams = urllib.urlencode(params)

    # Set API endpoint
    request = urllib2.Request(url + '?' + encodedParams)

    # Add authorization header to the request
    request.add_header("Authorization", "Basic %s" % basicAuthString)

    try:
        response = urllib2.urlopen(request)
        return response.read()
    except urllib2.HTTPError, err:
        if err.code == 401:
            print "Error: Invalid API credentials"
            quit()
        elif err.code == 404:
            print "Error: The API endpoint is currently unavailable"
            quit()
        else:
            print err
            quit()

# # Request stations from API
# response = apiRequest("https://tahmoapi.mybluemix.net/v1/timeseries/" + stationId + "/hourly", {'startDate': startDate, 'endDate':endDate})
# decodedResponse = json.loads(response)
#
# # Check if API responded with an error
# if (decodedResponse['status'] == 'error'):
#     print "Error:", decodedResponse['error']
#
# # Check if API responded with success
# elif (decodedResponse['status'] == 'success'):
#
#     # Print the amount of stations that were retrieved in this API call
#     print "API call success:", "Station", decodedResponse['station']['id'], decodedResponse['station'][
#         'name'], "timeseries retrieved"
#     print "Timeseries available for ", ", ".join(decodedResponse['station']['variables'])
#
#     # Loop through temperature timeseries and print values
#     try:
#         if decodedResponse['timeseries'][variable]:
#             print "\nTemperature measurements during last two days: "
#             print decodedResponse['timeseries'][variable]
#             for timestamp, value in sorted(decodedResponse['timeseries'][variable].items()):
#                 print timestamp, value
#
#     except:
#         print "Temperature timeserie starting at " + startDate + " unavailable"
#         quit()
#


class TahmoAPI(DataSource):
    @classmethod
    def __check_response(cls, decoded_response):
        # Check if API responded with an error
        if (decoded_response['status'] == 'error'):
            return ValueError("Error:", decoded_response['error'])
        else:

            return decoded_response['status'] == 'success'

    @staticmethod
    def station_list():
        response = apiRequest("https://tahmoapi.mybluemix.net/v1/stations")
        decoded_response = json.loads(response)
        if TahmoAPI.__check_response(decoded_response):
            # list active stations and
            return decoded_response
    @staticmethod
    def measurements(station_name, variable, date_from, date_to, **kwargs):
        response = apiRequest("https://tahmoapi.mybluemix.net/v1/timeseries/" + station_name + "/hourly",
                              {'startDate': date_from, 'endDate': date_to})
        decoded_response = json.loads(response)
        if TahmoAPI.__check_response(decoded_response):
            return decoded_response['timeseries'][variable]
    def active_stations(self):
        return

    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):
        return  TahmoAPI.station_list()


if __name__ == '__main__':
    target_station = 'TA00055'
    stn_list = TahmoAPI.station_list()
    print (stn_list['stations'][0])
    variable = 'precipitation'
    startDate = datetime.datetime.strftime(datetime.datetime.utcnow() - datetime.timedelta(20), '%Y-%m-%dT%H:%M')
    endDate = datetime.datetime.strftime(datetime.datetime.utcnow() - datetime.timedelta(10), '%Y-%m-%dT%H:%M')

    #nearby_station = tt.nearby_stations(target_station)
    #print nearby_station
    measure = TahmoAPI.measurements(station_name=target_station, variable='temperature',
                            date_from= startDate, date_to=endDate)
                            #group='D')
    print measure