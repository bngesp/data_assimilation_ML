#%%
#Import
from statsmodels.tsa.ar_model import AR
from random import random
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
#DATA
# contrived dataset
data = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/data_cathedrale.csv', sep=";")
data = data.iloc[::-1]

data.head()
#%%
data['receiving_date'] = pd.to_datetime(data['receiving_date'], format='%Y-%m-%d %H:%M:%S')
#data["datetime"] = data["datetime"].iloc[::-1]
data.describe
data['timestamp'] = data['receiving_date']  

#%%
el = data['receiving_date'].astype(str).str.split(" ", expand = True)
data['date'] = el[0]
data['time'] = pd.to_datetime(el[1], format='%H:%M:%S')
data.head(5)
# %%
el = data['contents'].str.split(":", expand = True)

data['pm1'] = el[0].astype(float)
data['pm2'] = el[1].astype(float)
data['pm10'] = el[2].astype(float)
data['humidite'] = el[3].astype(float)
data['temperature'] = el[4].astype(float)

el
#%%
data.describe
#%%
# AR method
model = AR(data['pm10'])
model_fit = model.fit(maxlag=1, trend='nc')
#%%
# make prediction
yhat = model_fit.predict(start=1, end=len(data))
print(yhat)

# %%
yhat

# %%
plt.figure(num=None, figsize=(8, 5), dpi=70)
plt.scatter(list(range(200)), data['pm10'][:200], color = 'blue')
plt.plot(list(range(200)),yhat[:200], marker = '*', color = 'red')
plt.show()


# %%
from statsmodels.graphics.tsaplots import plot_acf

# %%
plot_acf(data['temperature']), lags=31)
plt.show()

# %%
ar_mse = r2_score(data['pm10'],yhat )
ar_mse

# %%
