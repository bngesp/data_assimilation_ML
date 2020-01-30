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

#%%
df5_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_t.csv')
df5_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_2t.csv')
df5_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_30t.csv')
df5_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_15t.csv')
df5_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_10t.csv')
df5_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_25t.csv')
df5_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_48t.csv')


#%%
df6_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_t.csv')
df6_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_2t.csv')
df6_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_30t.csv')
df6_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_15t.csv')
df6_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_10t.csv')
df6_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_25t.csv')
df6_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=6/T_egal_48t.csv')


#%%
df7_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_t.csv')
df7_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_2t.csv')
df7_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_30t.csv')
df7_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_15t.csv')
df7_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_10t.csv')
df7_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_25t.csv')
df7_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=7/T_egal_48t.csv')


#%%
df8_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_t.csv')
df8_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_2t.csv')
df8_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_30t.csv')
df8_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_15t.csv')
df8_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_10t.csv')
df8_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_25t.csv')
df8_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=8/T_egal_48t.csv')

#%%
df9_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_t.csv')
df9_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_2t.csv')
df9_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_30t.csv')
df9_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_15t.csv')
df9_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_10t.csv')
df9_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_25t.csv')
df9_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=9/T_egal_48t.csv')


#%%
df10_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_t.csv')
df10_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_2t.csv')
df10_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_30t.csv')
df10_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_15t.csv')
df10_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_10t.csv')
df10_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_25t.csv')
df10_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_48t.csv')

#%%
df5 = list([df5_1["lost"].mean(), df5_2["lost"].mean(),  df5_10["lost"].mean(), df5_22["lost"].mean(),df5_5["lost"].mean(), df5_48["lost"].mean()])
df6 = list([df6_1["lost"].mean(), df6_2["lost"].mean(), df6_10["lost"].mean(), df6_22["lost"].mean(), df6_5["lost"].mean(), df6_48["lost"].mean()])
df8 = list([df8_1["lost"].mean(), df8_2["lost"].mean(), df8_10["lost"].mean(),  df8_22["lost"].mean(), df8_5["lost"].mean(), df8_48["lost"].mean()])
df7 = list([df7_1["lost"].mean(), df7_2["lost"].mean(),  df7_10["lost"].mean(), df7_22["lost"].mean(),df7_5["lost"].mean(), df7_48["lost"].mean()])
df9 = list([df9_1["lost"].mean(), df9_2["lost"].mean(),  df9_10["lost"].mean(),  df9_22["lost"].mean(),df9_5["lost"].mean(), df9_48["lost"].mean()])
df10 = list([df10_1["lost"].mean(), df10_2["lost"].mean(), df10_10["lost"].mean(), df10_22["lost"].mean(),df10_5["lost"].mean(),  df10_48["lost"].mean()])

df5.sort()
df6.sort()
df7.sort()
df8.sort()
df9.sort()
df10.sort()
x_axis = list([1,2,10,25,30,48])

#%%
print(df5)
print(df6)
print(df7)
print(df8)
print(df9)
print(df10)
#%%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(x_axis, df5, label="w=5")
plt.plot(x_axis, df6,label="w=6")
plt.plot(x_axis, df7,label="w=7")
plt.plot(x_axis, df8,label="w=8")
plt.plot(x_axis, df9, label="w=9")
plt.plot(x_axis, df10,  label="w=10")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("λ values")
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
plt.legend(loc='lower right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("λ values")
plt.ylabel("error")
plt.show()


#%%

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(x_axis, df8)
# plt.plot(x_axis, df6,label="w=6")
# plt.plot(x_axis, df7,label="w=7")
# plt.plot(x_axis, df8,label="w=8")
# plt.plot(x_axis, df9, label="w=9")
# plt.plot(x_axis, df10,  label="w=10")
#plt.plot(Z, W, 'green', label="real Data")
#plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("λ value")
plt.ylabel("mean error")
plt.show()


# %%
