#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
from sklearn.metrics import r2_score

print("terminer les importations")

#%%
df = pd.read_csv("/Users/admin/Documents/ML/Thesis/data/data.csv", delim_whitespace=True)
df.head(100)

#%%
df.dtypes


#%%
df['datetime'] = df["date"].astype(str) + " "+ df["heure"].astype(str) #.astype(str).sum(axis=1)
df['datetime'] = pd.to_datetime(df['datetime']) #.astype('datetime64[ns]') 
df["seconde"] = (pd.datetime.now()- df['datetime']).dt.total_seconds()/1000000
#(pd.to_timedelta(df["datetime"])/np.timedelta64(1, 's'))/1000
df['seconde'].head()
#%%
df.head()
df.dtypes
#%%
register_matplotlib_converters()
df = df.sort_values('seconde', ascending=True)
plt.plot(df['seconde'], df['temp'])
plt.xticks(rotation='vertical')
plt.show()

#%%
X=df["seconde"]
Y=df["temp"]
f1 = np.polyfit(X, Y, 100)
p = np.poly1d(f1)
#print(p)
r_squared = r2_score(Y, p(X))
print('The R-square value is: ', r_squared)

plt.plot(X, Y, 'bo', label="Data")
plt.plot(X, p(X), 'r',label="Polyfit")
plt.show()

#%%
