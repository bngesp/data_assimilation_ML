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
df21_3 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/3/T_egal_t.csv')
df21_4 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/4/T_egal_t.csv')
print(df21_3.shape)
print(df21_4.shape)

#%%
# le 22
df22_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/1/T_egal_t.csv')
df22_2 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/2/T_egal_t.csv')
df22_3 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/3/T_egal_t.csv')
df22_4 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/4/T_egal_t.csv')
print(df22_1.shape)
print(df22_2.shape)
print(df22_3.shape)
print(df22_4.shape)

df21_data = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_t.csv')

#%%
df21_3 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/3/T_egal_t.csv')
df21_4 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/4/T_egal_t.csv')
print(df21_3.shape)
print(df21_4.shape)
#%%
df21 = list([ 
    df21_3[(df21_3['Cycle'] >= 0) & (df21_3['Cycle'] <= 4)]['lost'].mean(),
    df21_3[(df21_3['Cycle'] >= 5) & (df21_3['Cycle'] <= 10)]['lost'].mean(),
    df21_3[(df21_3['Cycle'] >= 11) & (df21_3['Cycle'] <= 16)]['lost'].mean(),
    df21_3[(df21_3['Cycle'] >= 17) & (df21_3['Cycle'] <= 23)]['lost'].mean(),
    df21_4[(df21_4['Cycle'] >= 0) & (df21_4['Cycle'] <= 4)]['lost'].mean(),
    df21_4[(df21_4['Cycle'] >= 5) & (df21_4['Cycle'] <= 10)]['lost'].mean(),
    df21_4[(df21_4['Cycle'] >= 11) & (df21_4['Cycle'] <= 16)]['lost'].mean(),
    df21_4[(df21_4['Cycle'] >= 17) & (df21_4['Cycle'] <= 23)]['lost'].mean()
])

df21

# %% 
# Boxplot 21

fig = plt.figure()
#fig.tight_layout()
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"

plt.subplot(2, 1, 1)
#plt.boxplot([df21_3['lost'], df21_4['lost']])
plt.scatter(range(len(df21_data['lost'])), df21_data['real'], label="real data")
plt.plot(range(len(df21_data['lost'])), df21_data['estimate'], c='red', label="estimated")
# plt.ylim(0, 0.7)
# plt.xticks([1, 2], ['12h - 18h', '18h-23h'])
plt.legend(loc='upper center')
plt.xticks(rotation=45, ha='right')
plt.title('data 2019-08-21')
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.plot(range(0,24,3), df21, label="21")
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("hour")
plt.ylabel("Mean error")
plt.title('error repartition')
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.boxplot([df21_3['lost'], df21_4['lost']])
plt.ylim(0, 0.7)
plt.xticks([1, 2], ['12h - 18h', '18h-23h'])
plt.title('error')
plt.tight_layout()

plt.show()

# %% 
# Boxplot 22
fig = plt.figure()
#fig.tight_layout()

#plt.subplot(2, 2, 1)
#plt.boxplot([df22_1['lost'], df22_2['lost'], df22_3['lost'], df22_4['lost']])
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4], ['0h-6h', '6h-12h', '12h-18h', '18h-23h'])
plt.title('data 2019-08-22')
#plt.tight_layout()
plt.show()
# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"

#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(Z, X, c='blue', label="real data")
plt.plot(range(0,24,3), df21, label="21")
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("hour")
plt.ylabel("Mean error")
plt.show()

# %%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(Z, X, c='blue', label="real data")
plt.scatter(range(len(df21_3['real'])), df21_3['real'], label="real")
plt.plot(range(len(df21_3['estimate'])), df21_3['estimate'], c='red' ,label="estimage")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.xlabel("hours")
plt.ylabel("Mean error")
plt.show()


# %%
