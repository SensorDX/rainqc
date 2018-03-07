import matplotlib.pyplot as plt
import pandas as pd


minutes_dt = pd.read_csv('/scratch/projects/sensordx/dump/CLAY.csv')
hourly_dt = pd.read_csv('~/adams/tahmoqcsensordx/odm/hourly/CLAY.csv')
day_dt = pd.read_csv('~/adams/tahmoqcsensordx/odm/daily/CLAY.csv')
year=2008

annual = minutes_dt[minutes_dt.rDate<'2009-01-01']
plt.plot(annual.RAIN, 'r.')
print len(annual.RAIN)
plt.ylim(0)
plt.show()