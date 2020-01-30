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
df5 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t5.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t5.csv')['lost'].mean()
])
#df5

#%%
df6 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t6.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t6.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t6.csv')['lost'].mean()
])

#%%
df7 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t7.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t7.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t7.csv')['lost'].mean()
])

#%%
df8 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t8.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t8.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t8.csv')['lost'].mean()
])

#%%
df9 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t9.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t9.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t9.csv')['lost'].mean()
])

#%%
df10 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t10.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t10.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t10.csv')['lost'].mean()
])

#%%
df15 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t15.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t5.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t15.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t15.csv')['lost'].mean()
])

#%%
df20 = list([ pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_t/T_egal_t20.csv')['lost'].mean(), 
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_5t/T_egal_5t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_10t/T_egal_10t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_15t/T_egal_15t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_20t/T_egal_20t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_25t/T_egal_25t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_30t/T_egal_30t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_40t/T_egal_40t20.csv')['lost'].mean(),
pd.read_csv('/Users/admin/Documents/ML/Thesis/data/data/T_egal_50t/T_egal_50t20.csv')['lost'].mean()
])
#%%
print(df5.sort())
print(df6.sort())
print(df7.sort())
print(df8.sort())
print(df9.sort())
print(df10.sort())
print(df15.sort())
print(df20.sort())

#%%
x_axis = list([1,5, 10,15,20,25,30,40,50])
#%%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(x_axis, df5, label="w=5")
# plt.plot(x_axis, df6,label="w=6")
# plt.plot(x_axis, df7,label="w=7")
plt.plot(x_axis, df8,label="w=8")
plt.plot(x_axis, df9, label="w=9")
plt.plot(x_axis, df10,  label="w=10")
plt.plot(x_axis, df15,  label="w=15")
plt.plot(x_axis, df20,  label="w=20")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("coorection")
plt.ylabel("error")
plt.show()


#%%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.scatter(x_axis, df5, label="w=5")
plt.scatter(x_axis, df6,label="w=6")
plt.scatter(x_axis, df7,label="w=7")
plt.scatter(x_axis, df8,label="w=8")
plt.scatter(x_axis, df9, label="w=9")
plt.scatter(x_axis, df10,  label="w=10")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("coorection")
plt.ylabel("error")
plt.show()


#%%
