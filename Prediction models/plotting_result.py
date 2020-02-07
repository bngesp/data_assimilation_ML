#%%
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = "12"
plt.figure(num=None, figsize=(8, 5), dpi=70)
plt.scatter(list(range(100)), data['pm10'][:100], color = 'blue', label="data")
plt.plot(list(range(100)),yhat[:100], marker = '*', label = "AR")
plt.plot(list(range(100)),arima1_predict[:100], label = "ARIMA")
plt.plot(list(range(100)),arima_predict[:100], label = "MA")
plt.plot(list(range(100)),sarima_predict[:100], label = "SARIMA")
plt.plot(list(range(100)),sarimax_predict[:100], label="SARIMAX")
plt.legend(loc='upper right')
plt.show()

# %%
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
langs = ['AR', 'AM', 'ARIMA', 'SARIMA', 'SARIMAX']
students = [ar_mse,arma_mse,arima_mse,sarima_mse,sarimax_mse]
ax.bar(langs,students)
ax.set_ylabel('MSE')
ax.set_title('MSE for Times series Models')
plt.show()

# %%
