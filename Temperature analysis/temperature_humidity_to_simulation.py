#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P
from numpy.polynomial import Chebyshev as T
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import r2_score


# #%%

# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-21.csv')
# df.shape #48

# #%%
# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-22.csv')
# df.shape #93
# %%
# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-23.csv')
# df.shape #93
# #%%

# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-24.csv')
# df.shape #44

#%%

# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-25.csv')
# df.shape #85

# #%%
# df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-26.csv')
# df.shape #93

#%%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-28.csv')
df.shape #86
#%%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-29.csv')
df.shape #86
#%%

df['datetime'] = df['date'].astype(str) + " "+ df["heure"].astype(str)+":"+ df["minute"].astype(str)+":"+ df["seconde"].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'])
df["secondes"] = (pd.datetime.now()- df['datetime']).dt.total_seconds()/1000000
df.head

df["temperature"].shape
#%%
X = np.array(list(range(1, 796)))
X.shape

Y = df["temperature"].iloc[::-1] #.values.reshape(data["heure"].shape[0], 1))

#%%
plt.scatter(X, Y)
plt.show()


#%%
p2 = P.fit(X, Y, 157)

#%%
print(p2)
r_squared2 = r2_score(Y, p2(X))
print('The R-square value is: ', r_squared2)

#%%
#P.mapparms(p2)
P.degree(p2)


#%%
plt.plot(X, Y, 'bo', label="Data")
plt.plot(X, p2(X), 'yellow',label="Polynomial deg=7")
plt.legend(loc='upper right')

plt.show()




#%%
df = pd.read_csv('/Users/admin/Documents/ML/2019-08-21.csv')
df2 = pd.read_csv('/Users/admin/Documents/ML/2019-08-22.csv')
df3 = pd.read_csv('/Users/admin/Documents/ML/2019-08-23.csv')
df4 = pd.read_csv('/Users/admin/Documents/ML/2019-08-24.csv')
df.shape


#%%
X = np.array(list(range(0, 48)))
Y = df["temperature"].iloc[::-1]
p = P.fit(X, Y, 7)
print(p)
r_squared2 = r2_score(Y, p(X))
print('The R-square value is: ', r_squared2)
plt.plot(X, Y, 'bo', label="Data 21-08-2019")
plt.plot(X, p(X), 'r',label="Polyfit deg=3")
plt.legend(loc='upper right')
plt.show()
#%%
P.degree(p)
P.roots(p)
P.mapparms(p)
P._get_coefficients(p, p)


#%%
df2.describe
X2 = np.array(list(range(0, 93)))
Y2 = df2["temperature"].iloc[::-1]
p = P.fit(X2, Y2, 7)
r_squared2 = r2_score(Y2, p(X2))
print('The R-square value is: ', r_squared2)
plt.plot(X2, Y2, 'bo', label="Data 22-08-2019")
plt.plot(X2, p(X2), 'r',label="Polyfit deg=3")
plt.legend(loc='upper right')
plt.show()


#%%
df3.describe

X3 = np.array(list(range(0, 93)))
Y3 = df3["temperature"].iloc[::-1]
p = P.fit(X3, Y3, 7)
r_squared2 = r2_score(Y3, p(X3))
print('The R-square value is: ', r_squared2)
plt.plot(X3, Y3, 'bo', label="Data 23-08-2019")
plt.plot(X3, p(X3), 'r',label="Polyfit deg=3")
plt.legend(loc='upper right')

plt.show()

#%%
df4.shape
X4 = np.array(list(range(0, 44)))
Y4 = df4["temperature"].iloc[::-1]
p = P.fit(X4, Y4, 7)
r_squared2 = r2_score(Y4, p(X4))
print('The R-square value is: ', r_squared2)
plt.plot(X4, Y4, 'bo', label="Data 24-08-2019")
plt.plot(X4, p(X4), 'r',label="Polyfit deg=3")
plt.legend(loc='upper right')
plt.show()

#%%
plt.subplot(2, 2, 1)
plt.plot(X, Y, 'yellow', label="21-08-2019")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 2)
plt.plot(X2, Y2, 'blue', label="22-08-2019")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 3)
plt.plot(X3, Y3, 'red', label="23-08-2019")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.subplot(2, 2, 4)
plt.plot(X4, Y4, 'green', label="24-08-2019")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')

plt.show()
#%%
plt.plot(X, Y, 'yellow', label="21-08-2019")
plt.plot(X2, Y2, 'blue', label="22-08-2019")
plt.plot(X3, Y3, 'red', label="23-08-2019")
plt.plot(X4, Y4, 'green', label="24-08-2019")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
#send data to GAMA Simultation

for i in Y:
    print(i)
#%%
data = pd.read_csv('/Users/admin/Documents/ML/my_data.csv')

#%%
#df1.describe
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/T_egal_t.csv')
df["time"] = (df["Cycle"]-3)*2 -1
#df.describe
df_tmp = pd.DataFrame(columns=['Cycle',  'estimate', 'lost','time'])
for i in range(2, 1589, 2):
    new_row = {'Cycle':np.nan,  'estimate':np.nan, 'lost':np.nan,'time':int(i)}
    df_tmp = df_tmp.append(new_row, ignore_index=True)
#df.sort_values(by=['time'])
df_tmp['time'] =df_tmp['time'].astype(int)
# df_tmp.describe
df = df.append(df_tmp, ignore_index=True)
df.sort_values(by=['time'], inplace=True)
df = df.reset_index(drop=True)
df
#data.describe()
#df.head(5)
# df["Cycle"] = df["Cycle"]*2

# df.head(10)
#data["temperature"].shape
#df1["estimate"].shape

#%%
df1 = pd.read_csv('/Users/admin/Documents/ML/Thesis/save/Paper/T_egal_2t.csv')
df1.describe

#%% 
#Courbe pour T=2t

X = df["estimate"]
Y = df1["estimate"]
Z = range(1589)

plt.scatter(Z, X, c='blue', label="T_data=t_simul")
plt.plot(Z, Y, c='yellow', label="T_data=2t_simul")
#plt.plot(Z, Y, 'yellow', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
X = data["temperature"].iloc[::-1]
Y = df["estimate"]
Z = range(795)

plt.scatter(Z, X, c='blue', label="real Data")
plt.scatter(Z, Y, c='yellow', label="estimate")
#plt.plot(Z, Y, 'yellow', label="estimate")
#plt.plot(Z, W, 'green', label="real Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
W = df["lost"]
plt.plot(Z, W, 'green', label="lost Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()

#%%
X1 = data["temperature"].iloc[::-1]
Y1 = df["estimate"]
Z1 = range(795)

plt.scatter(Z1, X1, c='blue', label="real Data")
plt.plot(Z1, Y1, 'yellow', label="estimate")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
#%%
W1 = df["lost"]
plt.plot(Z1, W1, 'green', label="lost Data")
plt.legend(loc='upper right')
plt.xticks(rotation=45, ha='right')
plt.show()
