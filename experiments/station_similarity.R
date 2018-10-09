# Load all the stations. 
library('dplyr')
tahm <- read.csv('dataset/tahmostation2016.csv')
odm <- read.csv("dataset/mesonet_2008.csv")

stations = "../localdatasource/nearest_stations.csv"
thm <- read.csv(stations)
t_station = "TA00024"
nearby_stations <- function(target_station, k=5, radius=100){
  df = read.csv(stations)
  k_stations = df[df$from==target_station & df$distance<radius,]
  top_k_stations = k_stations[order(k_stations$distance),][1:k,'to']
  return(top_k_stations)
}

k_stations = nearby_stations(target_station =t_station )
# the dataset 
df <- tahm[,c(t_station,as.vector(k_stations))]
non_zero <- scale(df[df$TA00024>0.0,])
