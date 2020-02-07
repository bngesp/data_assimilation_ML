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
from statsmodels.tsa.statespace.sarimax import SARIMAX
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
model = SARIMAX(data['pm10'], order = (1, 1, 1), seasonal_order=(1, 1, 1, 1))
model_fit = model.fit(disp=False)
#%%
# make prediction
sarima_predict = model_fit.predict(start=1, end=len(data))
print(sarimas_predict)

# %%
yhat

# %%
plt.figure(num=None, figsize=(8, 5), dpi=70)
plt.scatter(list(range(100)), data['pm10'][:100], color = 'blue')
plt.plot(list(range(100)),arima1_predict[:100], marker = '*', color = 'red')
plt.plot(list(range(100)),arima_predict[:100], color = 'green')
plt.plot(list(range(100)),sarima_predict[:100])
plt.show()


# %%
from statsmodels.graphics.tsaplots import plot_acf

# %%
plot_acf(data['temperature']), lags=31)
plt.show()

# %%
sarima_mse = r2_score(data['pm10'],sarima_predict )
sarima_mse

# %%
arima1_mse - arima_mse


# %%
