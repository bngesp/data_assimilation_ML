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
df5_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_5t.csv')
df5_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_7t.csv')
df5_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_10t.csv')
# df5_13 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_13t.csv')
df5_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_22t.csv')
df5_15 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_15t.csv')
df5_25 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_25t.csv')
df5_30= pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_30t.csv')
df5_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_48t.csv')
# df5_32= pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_32t.csv')

# df5_100 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=5/T_egal_95.csv')

#%%
df6_1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_t.csv')
df6_2 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_2t.csv')
df6_5 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_5t.csv')
df6_7 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_7t.csv')
df6_10 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_10t.csv')
# df5_13 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_13t.csv')
df6_22 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_22t.csv')
df6_15 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_15t.csv')
df6_25 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_25t.csv')
df6_30= pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_30t.csv')
df6_48 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/w=10/T_egal_48t.csv')
#%%
df_test = list([df5_1["lost"].mean(),df5_2["lost"].mean(),df5_5["lost"].mean(),df5_7["lost"].mean(),df5_10["lost"].mean(),df5_15["lost"].mean(),df5_22["lost"].mean(),df5_25["lost"].mean(),df5_30["lost"].mean(),df5_48["lost"].mean()])
df_test.sort()
df_test1 = list([df6_1["lost"].mean(),df6_2["lost"].mean(),df6_5["lost"].mean(),df6_7["lost"].mean(),df6_10["lost"].mean(),df6_15["lost"].mean(),df6_22["lost"].mean(),df6_25["lost"].mean(),df6_30["lost"].mean(),df6_48["lost"].mean()])
df_test1.sort()
#%%
x_axis = list([1,5, 10,15,20,30,40,50,60, 70])

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(x_axis, data, c='red', label="estimated data")
plt.plot(x_axis, df_test, label="w=5")
plt.plot(x_axis, df_test1, label="w=10")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("coorection")
plt.ylabel("error")
plt.show()

#%%
