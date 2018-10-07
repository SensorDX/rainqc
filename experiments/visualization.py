import matplotlib.pylab as plt
import seaborn as sbn
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

train_data = pd.read_csv('tahmostation2016.csv')
#test_data = pd.read_csv('tahmostation2017.csv')
#train_data = test_data
def nearby_stations(site_code, k=10, radius=500, path="../localdatasource/nearest_stations.csv"):
    stations = pd.read_csv(path)  # Pre-computed value.
    k_nearest = stations[(stations['from'] == site_code) & (stations['distance'] < radius)]

    k_nearest = k_nearest.sort_values(by=['distance', 'elevation'], ascending=True)[0:k]

    # available_stations = LocalDataSource.__available_station(k_nearest, k)
    return k_nearest #.tolist()  # available_stations


#train_data = pd.read_csv('mesonet_2009.csv')
print train_data.shape

def plot_all():


    stations = train_data.columns.tolist()
    print plt.style.available
    plt.style.use(plt.style.available[14]) #'seaborn-notebook')
    with PdfPages("tahmo_rain_fall_2016.pdf") as pdf:
        #for style in plt.style.available:

        for station in stations[:]:


            plt.subplot(3,2,1)
             #= 'white'
            plt.plot(train_data[station],'.g', label=station)
            plt.ylabel("Rain (mm)")
            plt.xlabel("Days")
            plt.legend(loc='best')
            k_stations = nearby_stations(station, k=5) #, path="Nearest_station.dt")
            for ix, k_stn in enumerate(k_stations['to'].tolist()):
                dist = np.round(float(k_stations['distance'][k_stations['to']==k_stn]),1)
                plt.subplot(3,2,ix+2)
                if k_stn not in stations:
                    continue
                plt.plot(train_data[k_stn], '.b',label=k_stn+" dist: "+ str(dist))
                plt.legend(loc="best")

            pdf.savefig()
            plt.close()


if __name__ == '__main__':
    plot_all()