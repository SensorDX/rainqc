from tahmo_datasource import TahmoDataSource
from FakeTahmo import FakeTahmo
from datetime import datetime, timedelta
import dateutil
import pytz


if __name__ == '__main__':
    target_stationx = "TA00055"
    thm = TahmoDataSource("station_nearby.json")
    start_datex = datetime.strftime(datetime.utcnow() - timedelta(200), '%Y-%m-%dT%H:%M')
    end_datex = datetime.strftime(datetime.utcnow() - timedelta(180), '%Y-%m-%dT%H:%M')
    print (start_datex)
    print (thm.get_stations())
    # print thm.daily_data(target_station,weather_variable=RAIN, start_date=start_date, end_date=end_date)
    # print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)
    # get active stations
    station_list = ['TA00028', 'TA00068', "TA00108", "TA00187"]
    # thm.compute_nearest_stations()
    # print thm.load_nearby_stations(target_station, k=5)
    #current_day = datetime.now(pytz.tz.tzutc())
    # print thm.active_stations(station_list, active_day_range=current_day)
    # print thm.nearby_stations(target_station=target_station, k=20, radius=200)
    # thm = TahmoDataSource()
    # print thm.get_stations()
    # print thm.get_data("TA00021", start_date="2017-09-01", end_date="2017-09-05")
    kdd = thm.load_nearby_stations("TA00025", radius=100)
    print (len(kdd), kdd)
    print (thm.active_stations([stn['site_to'] for stn in kdd]))
    # print thm.daily_data("TA00021", start_date="2017-09-01", end_date="2017-09-05", weather_variable=RAIN)
