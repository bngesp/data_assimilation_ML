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
import matplotlib.dates as mdates

#%%
data = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30.csv')
df21 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_t.csv')

# %%
df21.shape

#%%
data.shape
#%%

data["datetime"] = data["date"]+" "+data["heure"].astype(str)+":"+data["minute"].astype(str)+":"+data["seconde"].astype(str)
data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
#data["datetime"] = data["datetime"].iloc[::-1]
data['datetime'].describe

# %%
X = df21["real"][:76]
Y = df21["estimate"][:76]
#Z = range(47)
Z =  data["datetime"].iloc[::-1]#[:78]
#Z = Z[:]

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(Z, X, c='blue', label="real data")
plt.plot(Z, Y, 'red', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("temperature value(C)")
plt.show()


# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "14"
plt.plot_date(Z, X, c='blue', label="real data")
plt.plot(Z, Y, 'red', label="estimate")
# generate a formatter, using the fields required
fmtr = mdates.DateFormatter("%H:%M")
# need a handle to the current axes to manipulate it
ax = plt.gca()
# set this formatter to the axis
ax.xaxis.set_major_formatter(fmtr)
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("temperature value(C)")
plt.show()

# %%
