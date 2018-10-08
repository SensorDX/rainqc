import json
import sys
import pandas as pd
from cloudant.client import Cloudant
from cloudant.result import Result, ResultByKey
from cloudant.query import Query, QueryResult
import os
from data_source import DataSource
project_path ="/home/tadeze/projects/sensordx/rainqc"

class CloudantAuthentication:
    config = json.load(open(os.path.join(project_path,"common/config.localdatasource"),"rb"))["cloudant"]
    URL = config["URL"] #'https://tahmobluemix.cloudant.com/'
    client = Cloudant(config["USERNAME"],
                      config["PASSWORD"],
                      account=config["ACCOUNT_NAME"],
                      connect=True)
    @classmethod
    def open_session(cls):
        cls.session = cls.client.session()
    @classmethod
    def close(cls):
        cls.client.disconnect()


class BluemixSource(DataSource):

    @staticmethod
    def nearby_stations(site_code, k=10, radius=500):
        #return super(BluemixSource, site_code).nearby_stations(site_code, k, radius)
        pass
    @staticmethod
    def get_daily_rainfall(station_list, date):
        #return super(BluemixSource, station_list).get_daily_rainfall(station_list, date)
        pass
    @staticmethod
    def get_weather_data(station_name, variable, date_range):
        #return super(BluemixSource, station_name).get_weather_data(station_name, variable, date_range)
        station_name = "rm_"+station_name
        client = CloudantAuthentication.client
        CloudantAuthentication.open_session()
        current_station = client[station_name]
        selector = {u'date': {'$gte': date_range[0], '$lte': date_range[1]}}
        # query = Query(current_station,selector=selector)
        docs = current_station.get_query_result(selector)
        CloudantAuthentication.close()
        return docs

if __name__ == '__main__':
    js = BluemixSource.get_weather_data(station_name='ta00020', variable='pr',
                                        date_range=[u'2016-08-11', u'2016-08-12'])

    print len(js[:])








def extract_station(station_name='ta00001', date_from=u'2012-08-11', date_to=u'2012-08-12'):
    """
    Download data from station
    :param staton_name: station name
    :param  date_from: query from date.
    :param date_to: query to date.
    """
    station_name = "rm_"+station_name
    client = CloudantAuthentication.client
    CloudantAuthentication.open_session()
    current_station = client[station_name]
    selector = {u'date': {'$gte': date_from, '$lte': date_to}}
    #query = Query(current_station,selector=selector)
    docs = current_station.get_query_result(selector)
    CloudantAuthentication.close()
    return docs

#
# def main():
#     """
#         1. Choose station
#         2. Query data from a station
#         3. Convert JSON to csv
#         4. save as csv with station name
#         5. disconnect connection.
#     """
#     kenyan_station = ['TA00020', 'TA00021', 'TA00023', 'TA00024', 'TA00025', 'TA00026', 'TA00027', 'TA00028', 'TA00029',
#                       'TA00030', 'TA00054', 'TA00056', 'TA00057', 'TA00061', 'TA00064', 'TA00065', 'TA00066', 'TA00067',
#                       'TA00068', 'TA00069', 'TA00070', 'TA00071', 'TA00072', 'TA00073', 'TA00074', 'TA00076', 'TA00077',
#                       ]
#     index = 1
#     start_date = '2016-01-01T00:00:00.000Z'
#     end_date = '2017-12-31T23:55:00.000Z'
#     station_name = 'rm_' + kenyan_station[index]  # 'rm_TA00067'
#
#     print len(kenyan_station), "total stations--", station_name
#     # Loop for all stations.
#     result_doc = extract_station(
#         station_name=station_name.lower(), date_from=start_date, date_to=end_date)
#     # qqr = QueryResult()
#     qeury_len = len(result_doc[:])
#     # convert docs to csv
#     if qeury_len < 1:
#         print qeury_len, " Size of the query"
#         return
#     # localdatasource.dump(result_doc[:], open(os.path.join(
#     #     output_path, station_name + ".localdatasource"), "w"))
#     # # cv_todf = convert_to_df(result_doc)
#     # Save csv
#     # cv_todf.to_csv(os.path.join(output_path,station_name+".csv"),index=False)
#
