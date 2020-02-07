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
data = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/my_data.csv')
df = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperatur/diallo.csv')
#%%
df2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_t.csv')
#%%
a = df['lost'].mean()
a
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
X = X[:50]
X.shape

#%%
Z = dates
Z = Z[:50]
#%%
Y = df["estimate"][:50]
Y.shape
#Z = range(795)
#Z =  data["datetime"].iloc[::-1]
#%%
import matplotlib.dates as mdates
from matplotlib import gridspec
fig = plt.figure(figsize=(15,10))
gs  = gridspec.GridSpec(1, 1, height_ratios=[2])
ax0 = plt.subplot(gs[0])
fmtr = mdates.DateFormatter("%H:%M")
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
ax0.scatter(Z, X, c='blue', label="real Data")
ax0.plot(Z, Y, c='red', label="Diallo Method")
#plt.plot(Z, Y, 'yellow', label="estimate")
ax0.plot(Z, df2["estimate"][:50], 'green', label="my Method")
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(fmtr)
ax0.legend(loc='lower left')
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
