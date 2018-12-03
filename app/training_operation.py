""" Steps for training operations.
 Train batch of stations for training. The steps are.

check the database for which stations are trained for the last Y month.
 For each not trained stations:
 Push the station, date_From and date_to along with other relevant information to Queue,
 From the message queue, perform training operation of the stations one by one. And save the fitted model to the database with
 time of training, number of stations used ..etc

 Pop station from the MQ, then for each station, query it associated station for the given time period, and measure the amount of data available
 - Pre-assess data from all stations nearby to it, and decide whether there are enough data from the station to perform the operations.
 - if the data from the station makes sense and there are enough stations go-for training and save the model. If form some reason the training
 operation failed. Make the last trained model as active. Otherwise save the trained model and set it active for scoring operations.
 Continue this operations for all station periodically.

**Requirements**
 - There should be an available API from the bluemix to query.



 ## Scoring operations:
  - Check if there is any saved model before scoring operation, else return error or message that there is no saved models for
  the given station.
  - Load model from the stations.
  - Load data from the nearby stations/Mostly daily data. If some of them don't have try to impute the value.

"""


class TrainingStations(object):
    pass


