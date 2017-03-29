#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import math
import sys
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from lstm import *
from classifier import *
from burst_refinement import *
from prepare import *


def get_dates(filename):
    days = np.loadtxt(filename,
                      unpack=True, 
                      converters={0: mdates.strpdate2num('%Y%m%d')})
    # print days[:5]

    return days


def plot_results(date_file, dataset, train_size, seq_len, predicted_data):
    rnn_predictplot = np.empty_like(dataset)
    rnn_predictplot[:, :] = np.nan
    rnn_predictplot[train_size+(seq_len*2)+1:len(dataset)-1, :] = predicted_data
    dates = get_dates(date_file)
    dates = dates[len(dates)-len(dataset):]
    plt.plot_date(dates, dataset, "b-", label='Original Time Series')
    plt.plot_date(dates, rnn_predictplot, "r-", label='LSTM Prediction')
    plt.legend()


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts"
        exit(0)

    id_ = 4
    seq_len = 30
    epochs  = 10
    
    # load data
    dataset, trainX, trainY, testX, testY = load_data(sys.argv[1], sys.argv[2], seq_len, id_)
    print dataset.shape

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    t_data_c = get_samples_for_classfier(series, n_series, id_, seq_len) # [pos_samples, neg_samples]
    t_data_p_t = get_samples_for_predict_period(bursts, series, id_) # [seqs, time_periods]
    t_data_p_v = get_samples_for_predict_value(bursts, series, id_) # [seqs, values]
    t_data_r = get_samples_for_refine(sys.argv[4], sys.argv[5], sys.argv[6], id_) # [seqs]

    # build model
    lstm_model = build_model(seq_len)
    lstm_model.fit(trainX, trainY, nb_epoch=epochs, batch_size=1, verbose=2)
    classifier = build_svm_model(t_data_c[0], t_data_c[1])
    t_pdt = build_prediction_model(t_data_p_t[0], t_data_p_t[1])
    v_pdt = build_prediction_model(t_data_p_v[0], t_data_p_v[1])
    refiner = build_refine_model(t_data_r)

    # make predictions
    predictions = np.empty_like(testY)
    new_pre = np.empty_like(testY)
    time_step = 0
    for i, testx in enumerate(testX):
        predictions[i] = lstm_model.predict(testx) # should be a value

        # classfy burst
        rst = classfy_burst(classifier, testx, seq_len)
        if rst is not True:
            time_step = 0
            new_pre[i] = predictions[i]
            continue

        time_step += 1
        # predict period and value
        period = predict_burst_period(t_pdt, testx)
        value = predict_burst_value(t_pdt, testx)

        # refine output t+1 value
        new_pre[i] = refine_burst_prediction(refiner, testx, period, value, time_step)

    # invert predictions
    # dataset, new_pre = inverse_data(scaler, [dataset, [new_pre]])

    # plot
    plot_results(sys.argv[3], dataset, len(trainX), seq_len, new_pre)
    plot_results(sys.argv[3], dataset, len(trainX), seq_len, predictions)
    plt.show()
