import numpy as np
from scipy import stats
import pandas
import matplotlib.pyplot as plt
import sys
import statsmodels.api as sm

# get data
dta = np.genfromtxt(sys.argv[1], delimiter=",")
dataset = dta[0,:]
train_size = int(len(dataset) * 0.5)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size], dataset[train_size:len(dataset)]

# arma model
arma_mod30 = sm.tsa.ARMA(train, (3,2)).fit()
predict = arma_mod30.predict(train_size, len(dataset), dynamic=True)

print predict
np.savez(sys.argv[2], arma_predict=predict)
