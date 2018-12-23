import numpy as np
import pandas as pd
from sdxutils import group_station_data

#from fancyimpute import MICE





if __name__ == '__main__':
    import os
    pth = "/media/tadeze/kofo/research/Temp/dump"
    files = os.listdir(pth)
   # gg = group_station_data(os.path.join(pth,files[0]),group_type='D',variables=['RAIN'],
    #                        year=2008)
    df = pd.DataFrame()
    for ffile in files:
        colname = ffile.split('.')[0]
        gg = group_station_data(os.path.join(pth, ffile), group_type='D', variables=['RAIN'],
                                year=2009)
        gg.columns =[colname]
        if gg.shape[0]==365:
            df = pd.concat([df,gg],axis=1)
        print (df.shape)
    df.to_csv("mesonet_2009.csv",index=False)
