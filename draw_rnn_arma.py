import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import math
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )

rnn_file = np.load(sys.argv[1])
arma_file = np.load(sys.argv[2])
dates = get_dates(sys.argv[3])

# RNN(LSTM)
dataset = rnn_file["dataset"]
rnn_predict = rnn_file["rnn_predict"]
rnn_train = rnn_file["rnn_train"]

look_back = 15

# shift rnn_ predictions for plotting
rnn_predictplot = np.empty_like(dataset)
rnn_predictplot[:, :] = np.nan
rnn_predictplot[len(rnn_train)+(look_back*2)+1:len(dataset)-1, :] = rnn_predict
# plot baseline and predictions
plt.plot_date(dates, dataset, "b-", label='Original Time Series')
plt.plot_date(dates, rnn_predictplot, "r-", label='LSTM Prediction')

# ARIMA
arma_predict = arma_file["arma_predict"]
arma_predictplot = np.array([float(x) for x in range(len(dataset))])
arma_predictplot[:] = np.nan
arma_predictplot[int(len(dataset) * 0.5):len(dataset)] = arma_predict
print arma_predictplot
plt.plot_date(dates, arma_predictplot, "g-", label='ARMA Prediction')

# show
plt.show()