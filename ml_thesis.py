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


#%%

df = pd.read_csv('/Users/admin/Documents/ML/Thesis/train.csv', sep='\t')
df['datetime'] = "2018-12-27" + " "+ df["heure"].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'])
df["seconde"] = (pd.datetime.now()- df['datetime']).dt.total_seconds()/1000000
#(pd.to_timedelta(df["datetime"])/np.timedelta64(1, 's'))/1000
df['seconde'].shape
df.head()
df.dtypes


#%%
#data["heure"] = (pd.to_timedelta(data["heure"])/np.timedelta64(1, 's'))/1000
#X = df["seconde"] #.values.reshape(data["heure"].shape[0], 1))
Y = df["temp"] #.values.reshape(data["heure"].shape[0], 1))
X = np.array(list(range(1, 45)))
#ax = sns.regplot(x='heure', y='temp', data=data)
#ax.show()
# print(X)
# print(Y)
f1 = np.polyfit(X, Y, 7)
p = np.poly1d(f1)
#print(p)
r_squared = r2_score(Y, p(X))
print('The R-square value is: ', r_squared)
print(p)
p2 = P.fit(X, Y, 7)
print(p2)
r_squared2 = r2_score(Y, p2(X))
print('The R-square value is: ', r_squared2)

p3 = T.fit(X, Y, 7)
r_squared3 = r2_score(Y, p3(X))
print('The R-square value is: ', r_squared3)

plt.plot(X, Y, 'bo', label="Data")
plt.plot(X, p(X), 'r',label="Polyfit deg=7")
plt.plot(X, p2(X), 'green',label="Polynomial deg=7")
plt.plot(X, p3(X), 'yellow',label="Chebyshev deg=7")
plt.legend(loc='upper right')

plt.show()


#pr=PolynomialFeatures(degree=2)
#Z_pr=pr.fit_transform(Y)
# width = 12
# height = 10
# plt.figure(figsize=(width, height))
# sns.regplot(x="heure", y="temp", data=data)
# plt.ylim(0,)
# plt.show()
# lm = LinearRegression()
# lm.fit(X,Y)
# new_input=np.arange(1, 100, 1).reshape(-1, 1)
# print("\n debut test prediction ")
# Yhat=lm.predict(new_input)
# plt.plot(new_input, Yhat)
# plt.show()
# print(Yhat[0:5])

# print("l'intercept ")
# print(lm.intercept_)
# print("coef : ")
# print(lm.coef_)
# print("fin prediction\n")



#plt.scatter(X, Y)
#plt.show()


#%%

df = pd.read_csv('/Users/admin/Documents/ML/Thesis/train.csv', sep='\t')
df['datetime'] = "2018-12-28" + " "+ df["heure"].astype(str)
df['datetime'] = pd.to_datetime(df['datetime'])
df["seconde"] = (pd.datetime.now()- df['datetime']).dt.total_seconds()/1000000
#(pd.to_timedelta(df["datetime"])/np.timedelta64(1, 's'))/1000
df['seconde'].head()
df.head()
df.dtypes

#%%
#data["heure"] = (pd.to_timedelta(data["heure"])/np.timedelta64(1, 's'))/1000
X2 = df["seconde"] #.values.reshape(data["heure"].shape[0], 1))

plt.plot(X, Y, 'w', label="previous Data")
plt.plot(X2, Y, 'bo', label="Data")
#plt.plot(X2, p(X2), 'r',label="Polyfit deg=7")
#plt.plot(X2, p2(X2), 'green',label="Polynomial deg=7")
#plt.plot(X2, p3(X2), 'yellow',label="Chebyshev deg=7")
plt.legend(loc='upper right')

plt.show()

#%%
