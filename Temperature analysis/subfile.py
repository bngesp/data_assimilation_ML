#%%
import pandas as pd


# %%
df = pd.read_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30/2019-08-30.csv')
df = df.iloc[::-1]

#%%
df.describe
# %%
df1 = df[(df['heure'] >= 0) & (df['heure'] <= 5)]
df2 = df[(df['heure'] >= 6) & (df['heure'] <= 11)]
df3 = df[(df['heure'] >= 12) & (df['heure'] <= 17)]
df4 = df[(df['heure'] >= 18) & (df['heure'] <= 23)]

#%%
print(df1.shape)
print(df2.shape)
print(df3.shape)
print(df4.shape)
# %%
df1.to_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30/1.csv', index=False)
df2.to_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30/2.csv', index=False)
df3.to_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30/3.csv', index=False)
df4.to_csv('/Users/admin/Documents/ML/Thesis/data/temp_data/2019-08-30/4.csv', index=False)

# %%
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