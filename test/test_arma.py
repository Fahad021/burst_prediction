import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm

# get data
dta = np.genfromtxt(sys.argv[1], delimiter=",")
dataset = dta[0,:]

dataset = np.diff(dataset)
# plt.plot(dataset)

train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]


# fig = plt.figure(figsize=(12,8))
# ax1 = fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(dataset, lags=400, ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(dataset, lags=400, ax=ax2)

# arima model
arma_mod30 = sm.tsa.ARMA(train, (1,0)).fit()
print arma_mod30.aic,arma_mod30.bic,arma_mod30.hqic

# plt.show()

l = arma_mod30.predict(train_size, len(dataset)-1, dynamic=False)
predict = []
p = dta[0,len(train)-1]
for i in l:
  predict.append(p + i)
  p = p+i

print predict

np.savez(sys.argv[2], arma_predict=predict)
