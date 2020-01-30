#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates
from numpy.polynomial import Polynomial as P
from numpy.polynomial import Chebyshev as T
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import r2_score



#%%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/PM_Data/pm_data.csv', ";")
df.head(5)

#%%
df.describe

#%%
df['date'] =  pd.to_datetime(df['date'])
df = df[(df['date'] >= "2018-01-04") & (df['date'] < "2018-01-05")]
#%%
#df['datetime'] = df["date"].astype(str) + " "+ df["heure"].astype(str) #.astype(str).sum(axis=1)
df['datetime'] = pd.to_datetime(df['date']) #.astype('datetime64[ns]') 
df["seconde"] = (pd.datetime.now()- df['datetime']).dt.total_seconds()/1000000
df.describe

#%%
df.head(15)

#%%
df.to_csv('/Users/admin/Documents/ML/Thesis/data/PM_Data/2018-01-04.csv',index=False)

#%%
pm10 = df["pm10"]#.iloc[::-1]
pm25 = df["pm25"]
pm1 = df["pm1"]
Z = df['datetime']
Z1 = range(4027)

plt.plot(Z1, pm10,label="PM10")
plt.plot(Z1, pm25,  label="PM2.5")
plt.plot(Z1, pm1, label="PM1")
#plt.plot(Z, Y, 'yellow', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
