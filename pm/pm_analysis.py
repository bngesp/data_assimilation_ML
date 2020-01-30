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
#df5_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_t.csv')
df5_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_t5.csv')
df5_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_10t5.csv')
df5_20 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_20t5.csv')
df5_30 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_30t5.csv')
df5_40 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_40t5.csv')
df5_50 = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/T_egal_50t5.csv')

#%%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/pm/data/data_pm.csv')
# %%
df.shape

#%%
df1 = df['PM1'][:2880]
df1
#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(df['PM10'][:28800], 'black', label="PM10")
# plt.plot(df['PM25'], label="PM25")

plt.plot(df['PM25'][:28800], label="PM25")
plt.plot(df['PM1'][:28800], label="PM1")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper left')
# plt.xticks(rotation=45, ha='right')
# plt.xlabel("correlation between T and t ")
plt.ylabel("\u03BC\m\u33A5")
plt.show()
#%%
df5 = list([
    df5_1["lost"].mean(),
    df5_10["lost"].mean(),
    df5_20["lost"].mean(),
    df5_30["lost"].mean(),
    df5_40["lost"].mean(),
    df5_50["lost"].mean()
])
df5.sort()

# %%
x_axis = list([1,10,20,30,40, 50])

# %%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(x_axis, df5, label="error representation")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("correlation between T and t ")
plt.ylabel("error")
plt.show()

# %%
