#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys


def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )


def plot_results(date_file, dataset, seq_len, new_pred, predicted_data):
    rnn_predictplot = np.empty_like(dataset)
    rnn_predictplot[:, :] = np.nan
    rnn_predictplot[seq_len:len(dataset)-1, :] = np.array(predicted_data).reshape(-1,1)
    dates = get_dates(date_file)
    dates = dates[len(dates)-len(dataset):]
    plt.plot_date(dates, dataset, "b-", label="original data")
    plt.plot_date(dates, rnn_predictplot, "g-", label='LSTM Prediction')
    rnn_predictplot[seq_len:len(dataset)-1, :] = np.array(new_pred).reshape(-1,1)
    plt.plot_date(dates, rnn_predictplot, "r-", label='Our Prediction')
    plt.legend()
    plt.show()

print "rst_file, date_file"

rst_file = np.load(sys.argv[1])
plot_results(sys.argv[2], rst_file["dataset"], 
                          rst_file["seq_len"],
                          rst_file["new_pred"],
                          rst_file["pred"])
