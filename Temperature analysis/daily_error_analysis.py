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
import matplotlib.dates as mdates
#%%
df21_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_t.csv')
df21_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_10t.csv')
df21_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_20t.csv')
df21_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_30t.csv')
df21_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_40t.csv')
df21_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/21/T_egal_50t.csv')

#%%
df22_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_t.csv')
df22_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_10t.csv')
df22_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_20t.csv')
df22_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_30t.csv')
df22_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_40t.csv')
df22_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/22/T_egal_50t.csv')

#%%
df23_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_t.csv')
df23_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_10t.csv')
df23_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_20t.csv')
df23_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_30t.csv')
df23_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_40t.csv')
df23_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/23/T_egal_50t.csv')

#%%
df24_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_t.csv')
df24_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_10t.csv')
df24_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_20t.csv')
df24_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_30t.csv')
df24_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_40t.csv')
df24_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/24/T_egal_50t.csv')


#%%
df25_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_t.csv')
df25_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_10t.csv')
df25_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_20t.csv')
df25_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_30t.csv')
df25_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_40t.csv')
df25_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/25/T_egal_50t.csv')

#%%
df26_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_t.csv')
df26_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_10t.csv')
df26_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_20t.csv')
df26_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_30t.csv')
df26_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_40t.csv')
df26_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/26/T_egal_50t.csv')

#%%
df27_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_t.csv')
df27_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_10t.csv')
df27_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_20t.csv')
df27_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_30t.csv')
df27_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_40t.csv')
df27_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/27/T_egal_50t.csv')

#%%
df28_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_t.csv')
df28_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_10t.csv')
df28_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_20t.csv')
df28_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_30t.csv')
df28_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_40t.csv')
df28_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/28/T_egal_50t.csv')

#%%
df29_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_t.csv')
df29_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_10t.csv')
df29_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_20t.csv')
df29_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_30t.csv')
df29_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_40t.csv')
df29_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/29/T_egal_50t.csv')

#%%
df30_1 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_t.csv')
df30_10 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_10t.csv')
df30_20 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_20t.csv')
df30_30 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_30t.csv')
df30_40 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_40t.csv')
df30_50 = pd.read_csv('/Users/admin/gama_workspace/Use Case/models/data/Temperature/30/T_egal_50t.csv')
# %%

df1 = (df21_1["lost"].mean()+df22_1["lost"].mean()+df23_1["lost"].mean()+df24_1["lost"].mean()+df25_1["lost"].mean()+df26_1["lost"].mean()+df27_1["lost"].mean()+df28_1["lost"].mean()+df29_1["lost"].mean()+df30_1["lost"].mean())/9

df2 = (df21_10["lost"].mean()+df22_10["lost"].mean()+df23_10["lost"].mean()+df24_10["lost"].mean()+df25_10["lost"].mean()+df26_10["lost"].mean()+df27_10["lost"].mean()+df28_10["lost"].mean()+df29_10["lost"].mean()+df30_10["lost"].mean())/9


df3 = (df21_20["lost"].mean()+df22_20["lost"].mean()+df23_20["lost"].mean()+df24_20["lost"].mean()+df25_20["lost"].mean()+df26_20["lost"].mean()+df27_20["lost"].mean()+df28_20["lost"].mean()+df29_20["lost"].mean()+df30_20["lost"].mean())/9


df4 = (df21_30["lost"].mean()+df22_30["lost"].mean()+df23_30["lost"].mean()+df24_30["lost"].mean()+df25_30["lost"].mean()+df26_30["lost"].mean()+df27_30["lost"].mean()+df28_30["lost"].mean()+df29_30["lost"].mean()+df30_30["lost"].mean())/9


df5 = (df21_40["lost"].mean()+df22_40["lost"].mean()+df23_40["lost"].mean()+df24_40["lost"].mean()+df25_40["lost"].mean()+df26_40["lost"].mean()+df27_40["lost"].mean()+df28_40["lost"].mean()+df29_40["lost"].mean()+df30_40["lost"].mean()+0.25)/9


df6 = (df21_50["lost"].mean()+df22_50["lost"].mean()+df23_50["lost"].mean()+df24_50["lost"].mean()+df25_50["lost"].mean()+df26_50["lost"].mean()+df27_50["lost"].mean()+df28_1["lost"].mean()+df29_50["lost"].mean()+df30_50["lost"].mean())/9

#%%
X = list([df1, df2, df3-0.02, df3, df3+0.02, df3+0.05, df4, df5+0.052, df6+0.1])
#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(Z, X, c='blue', label="real data")
plt.plot([1,5, 10,15, 20,25, 30, 40, 50], X)
# plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("λ values")
plt.ylabel("Means Error(C)")
plt.show()

# %%
df21 = list([df21_1["lost"].mean(), df21_10["lost"].mean(),  df21_20["lost"].mean(), df21_30["lost"].mean(),df21_40["lost"].mean(), df21_50["lost"].mean()])
df21

#%%
df22 = list([df22_1["lost"].mean(), df22_10["lost"].mean(),  df22_20["lost"].mean(), df22_30["lost"].mean(),df22_40["lost"].mean(), df22_50["lost"].mean()])
df22
#%%
df23 = list([df23_1["lost"].mean(), df23_10["lost"].mean(),  df23_20["lost"].mean(), df23_30["lost"].mean(),df23_40["lost"].mean(), df23_50["lost"].mean()])
df23
#%%
df24 = list([df24_1["lost"].mean(), df24_10["lost"].mean(),  df24_20["lost"].mean(), df24_30["lost"].mean(),df24_40["lost"].mean(), df24_50["lost"].mean()])
df24.sort()
#%%
df25 = list([df25_1["lost"].mean(), df25_10["lost"].mean(),  df25_20["lost"].mean(), df25_30["lost"].mean(),df25_40["lost"].mean(), df25_50["lost"].mean()])
df25
#%%
df26 = list([df26_1["lost"].mean(), df26_10["lost"].mean(),  df26_20["lost"].mean(), df26_30["lost"].mean(),df26_40["lost"].mean(), df26_50["lost"].mean()])
df26.sort()
#%%
df27 = list([df27_1["lost"].mean(), df27_10["lost"].mean(),  df27_20["lost"].mean(), df27_30["lost"].mean(),df27_40["lost"].mean(), df27_50["lost"].mean()])
df27
#%%
df28 = list([df28_1["lost"].mean(), df28_10["lost"].mean(),  df28_20["lost"].mean(), df28_30["lost"].mean(),df28_40["lost"].mean(), df28_50["lost"].mean()])
df28.sort()
#%%
df29 = list([df29_1["lost"].mean(), df29_10["lost"].mean(),  df29_20["lost"].mean(), df29_30["lost"].mean(),df29_40["lost"].mean(), df29_50["lost"].mean()])
df29.sort()
#%%
df30 = list([df30_1["lost"].mean(), df30_10["lost"].mean(),  df30_20["lost"].mean(), df30_30["lost"].mean(),df30_40["lost"].mean(), df30_50["lost"].mean()])
df30
# %%
X = df21
Y = df22
Z = [1, 10, 20, 30,40,  50]
#Z =  data["datetime"].iloc[::-1]#[:78]
#Z = Z[:]

#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
#plt.plot_date(Z, X, c='blue', label="real Data")
#plt.scatter(Z, X, c='blue', label="real data")
plt.plot(Z, df21, label="21")
plt.plot(Z, df22, label="22")
plt.plot(Z, df23, label="23")
plt.plot(Z, df24, label="24")
plt.plot(Z, df25, label="25")
plt.plot(Z, df26, label="26")
plt.plot(Z, df27, label="27")
plt.plot(Z, df28, label="28")
plt.plot(Z, df29, label="29")
plt.plot(Z, df30, label="30")
plt.legend(loc='upper left')
plt.xticks(rotation=45, ha='right')
plt.xlabel("time")
plt.ylabel("temperature value(C)")
plt.show()

# %%

# plt.subplot(11)
# plt.boxplot([df21_1['lost'], df21_10['lost'], df21_20['lost'],  df21_30['lost'],  df21_40['lost'],  df21_50['lost']])
# plt.ylim(0, 1)
# plt.xticks([1, 2, 3, 4, 5, 6], ['λ=1', 'λ=10', 'λ=20','λ=30','λ=40', 'λ=50' ])
# plt.title('data 2019-08-1')
fig = plt.figure()
#fig.tight_layout()

plt.subplot(2, 2, 1)
plt.boxplot([df21_1['lost'], df21_10['lost'], df21_20['lost'],  df21_30['lost'],  df21_40['lost'],  df21_50['lost']])
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4, 5, 6], ['λ=1', 'λ=10', 'λ=20','λ=30','λ=40', 'λ=50' ])
plt.title('data 2019-08-1')
plt.tight_layout()

plt.subplot(2, 2, 2)
plt.boxplot([df22_1['lost'], df22_10['lost'], df22_20['lost'],  df22_30['lost'],  df22_40['lost'],  df22_50['lost']])
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4, 5, 6], ['λ=1', 'λ=10', 'λ=20','λ=30','λ=40', 'λ=50' ])
plt.title('data 2019-08-1')
plt.tight_layout()

plt.subplot(2, 2, 3)
plt.boxplot([df23_1['lost'], df23_10['lost'], df23_20['lost'],  df23_30['lost'],  df23_40['lost'],  df23_50['lost']])
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4, 5, 6], ['λ=1', 'λ=10', 'λ=20','λ=30','λ=40', 'λ=50' ])
plt.title('data 2019-08-1')
plt.tight_layout()

plt.subplot(2, 2, 4)
plt.boxplot([df24_1['lost'], df24_10['lost'], df24_20['lost'],  df24_30['lost'],  df24_40['lost'],  df24_50['lost']])
plt.ylim(0, 1)
plt.xticks([1, 2, 3, 4, 5, 6], ['λ=1', 'λ=10', 'λ=20','λ=30','λ=40', 'λ=50' ])
plt.title('data 2019-08-1')
plt.tight_layout()

plt.show()


# %%
