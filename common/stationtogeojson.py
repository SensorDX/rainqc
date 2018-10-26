
import pandas as pd
import json
station_csv = "~/adams/tahmobluemix/stations.csv"
df = pd.read_csv(station_csv)

geo_json = { "features":[], "type":"FeatureCollection"}

#print df.head(5)
for indx, row in  df.iterrows():
    #geo_station = {}
    print row["country"]
    geo_station = { "geometry":{"coordinates":[row["latitude"], row["longitude"], row["elevation"]],
                               "type":"Point"},
                    "properties":{
                                "Country": row["country"],
                                "Station id": row["id"],
                                "name": row["name"],
                                "datalogger": row["datalogger"],
                               # "last measurments": row[u"last measurements"],
                                "Station status": row["status"],
                                "marker-color": "#00ff00",
                                "marker-size": "medium",
                                "marker-symbol": "circle"
                                },
                    "type":"Feature"
                    }
    geo_json["features"].append(geo_station)


print geo_json
json.dump(geo_json, open("africa.geojson","w"))