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


# %%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/data_cathedrale.csv', sep=";")
df = df.iloc[::-1]

#%%
df['contents'].describe

# %%
el = df['contents'].str.split(":", expand = True)

df['data1'] = el[0].astype(float)
df['data2'] = el[1].astype(float)
df['data3'] = el[2].astype(float)
df['data4'] = el[3].astype(float)
df['data5'] = el[4].astype(float)


el
# %%
df.head(3)

#%%
df3.shape
# %%
df['receiving_date'] = pd.to_datetime(df['receiving_date'], format='%Y-%m-%d %H:%M:%S')
#data["datetime"] = data["datetime"].iloc[::-1]
df.describe
dates = matplotlib.dates.date2num(df['receiving_date'])
dates

#%%
el = df['receiving_date'].astype(str).str.split(" ", expand = True)
df['date'] = el[0]
df['time'] = pd.to_datetime(el[1], format='%H:%M:%S')
df.head(5)

#%%
df['date'] =  pd.to_datetime(df['date'])
df1 = df[(df['date'] >= "2019-11-18") & (df['date'] < "2019-11-30")]
df2 = df[(df['date'] >= "2019-12-01") & (df['date'] < "2019-12-31")]
df3 = df[(df['date'] >= "2020-01-01") & (df['date'] < "2020-01-31")]

#%%
df3.head(3)
# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
# plt.plot( df['data3'], label="data3")
# plt.plot(df['data2'], label="Data1")
# plt.plot(df1['receiving_date'], df1['data4'],  label="humidite")
plt.plot(df1['receiving_date'], df1['data5'], label="temperature" )
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("value")
plt.show()

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
# plt.plot( df['data3'], label="data3")
# plt.plot(df['data2'], label="Data1")
plt.plot(df1['receiving_date'], df1['data4'],  label="humidite")
# plt.plot(df1['receiving_date'], df1['data5'], label="temperature" )
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("value")
plt.show()


# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"

plt.subplot(3, 3, 1)
plt.plot(df1['receiving_date'], df1['data1'],  label="pm1 1er mois")
plt.tight_layout()
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')

plt.subplot(3, 3, 2)
plt.plot(df2['receiving_date'], df2['data1'],label="pm1 2em mois")
plt.tight_layout()
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')

plt.subplot(3, 3, 3)
plt.plot(df3['receiving_date'], df3['data1'],label="pm1 2em mois")
plt.tight_layout()
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("value")
plt.show()

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
# plt.plot( df['data3'], label="data3")
# plt.plot(df['data2'], label="Data1")
plt.plot(df1['receiving_date'], df1['data3'],c='green',  label="pm10")
#plt.plot(df['receiving_date'], df['data'], label="temperature" )
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("value")
plt.show()

# %%

from matplotlib import gridspec

#%%

fig = plt.figure(figsize=(15,10))
gs  = gridspec.GridSpec(3, 1, height_ratios=[1, 1.5 ,2])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
# ax3 = plt.subplot(gs[3])

fig.subplots_adjust(top=0.85)
lineObjects = ax0.plot(df['receiving_date'], df['data1'])
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')

ax0.set_title('PM1 data 2019-11-18 - 2020-01-27',fontsize=13)
ax0.legend(lineObjects, (1,2,3,4,5))

ax1.plot(df['receiving_date'],df['data2'])
ax1.set_title('PM2.5 data 2019-11-18 - 2020-01-27',fontsize=13)
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
# ax1.set_ylim([0,250])

ax2.plot(df['receiving_date'],df['data3'])
ax2.set_title('PM10 data 2019-11-18 - 2020-01-27',fontsize=13)
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')

# ax3.plot(df['data3'])
# ax3.set_title('Track Split Standard Dev',fontsize=11)
# ax3.set_ylim([0,100])

fig.tight_layout()
plt.show()

#%%
import matplotlib.dates as mdates
# %%
# Variation journaliere
fig = plt.figure(figsize=(15,10))
gs  = gridspec.GridSpec(4, 1, height_ratios=[1, 1.5 ,2, 1.5])
ax0 = plt.subplot(gs[0])
ax1 = plt.subplot(gs[1])
ax2 = plt.subplot(gs[2])
ax3 = plt.subplot(gs[3])

fmtr = mdates.DateFormatter("%H:%M")
# # need a handle to the current axes to manipulate it
# ax = plt.gca()
# # set this formatter to the axis
# ax.xaxis.set_major_formatter(fmtr)

fig.subplots_adjust(top=0.85)
lineObjects = ax0.plot(df1['receiving_date'][:100], df1['data1'][:100])
# plt.legend(loc='upper left')
ax0.xaxis_date()
ax0.xaxis.set_major_formatter(fmtr)
plt.xticks(rotation=45, ha='right')

ax0.set_title('PM1 data 2019-11-18 - 2020-01-27',fontsize=13)
ax0.legend(lineObjects, (1,2,3,4,5))

ax1.plot(df1['receiving_date'][:100],df1['data2'][:100])
ax1.set_title('PM2.5 data 2019-11-18 - 2020-01-27',fontsize=13)
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(fmtr)
plt.xticks(rotation=45, ha='right')
# ax1.set_ylim([0,250])

ax2.plot(df1['receiving_date'][:100],df1['data3'][:100])
ax2.set_title('PM10 data 2019-11-18 - 2020-01-27',fontsize=13)
ax2.xaxis_date()
ax2.xaxis.set_major_formatter(fmtr)
plt.xticks(rotation=45, ha='right')

ax3.plot(df1['receiving_date'][:100], df['data5'][:100])
ax3.set_title('Temperature',fontsize=15)
ax3.xaxis_date()
ax3.xaxis.set_major_formatter(fmtr)
plt.xticks(rotation=45, ha='right')
#ax3.set_ylim([0,100])

fig.tight_layout()
plt.show()

# %%
