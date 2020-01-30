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
data = pd.read_csv('/Users/admin/Documents/ML/my_data.csv')
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_t.csv')
#%%
df.shape
#%%

data["datetime"] = data["date"].astype(str)+" "+data["heure"].astype(str)+":"+data["minute"].astype(str)+":"+data["seconde"].astype(str)
data['datetime'] = pd.to_datetime(data['datetime'])
#data["datetime"] = data["datetime"].iloc[::-1]
data.describe
dates = matplotlib.dates.date2num(data['datetime'].iloc[::-1])
dates

#%%
data.shape
#%%
X = data["temperature"].iloc[::-1]
X = X[:792]
X.shape

#%%
Z = dates
Z = Z[:792]
#%%
Y = df["estimate"]
#Z = range(795)
#Z =  data["datetime"].iloc[::-1]

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
plt.plot_date(Z, X, c='blue', label="real Data")
#plt.plot(Z, Y, c='red', label="estimated data")
#plt.plot(Z, Y, 'yellow', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("temperature value(C)")
plt.show()



#%%
W = df["lost"]
plt.plot(range(792), W, 'yellow', label="lost Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
df["lost"].mean()

# %%
