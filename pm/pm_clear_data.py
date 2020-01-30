#%% # formation initiale
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
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/data_pm.csv')

#%%
print(df.shape)

#print(df["PM10"].describe)

#%%
pm10 = df["PM10"]#.iloc[::-1]
pm25 = df["PM25"]
pm1 = df["PM1"]
#Z = df['datetime']
Z1 = range(37406)

plt.plot(Z1, pm10,label="PM10")
#plt.plot(Z1, pm25,  label="PM2.5")
#plt.plot(Z1, pm1, label="PM1")
#plt.plot(Z, Y, 'yellow', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
