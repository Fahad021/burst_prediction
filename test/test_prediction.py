#!/usr/bin/env python
# coding=utf-8

# test period/peak prediction model

import numpy as np

import pandas
import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, log_loss

from lstm import *
from classifier import *
from burst_refinement import *
from prepare import *



if __name__ == "__main__":
    if len(sys.argv) != 7:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts"
        exit(0)

    id_ = 2
    seq_len = 30
    epochs  = 100
    
    # load data
    print "load data"
    dataset, trainX, trainY, testX, testY = load_data(sys.argv[1], sys.argv[2], seq_len, id_)
    print dataset.shape

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    t_data_c = get_samples_for_classfier(series, n_series, seq_len) # [pos_samples, neg_samples]
    t_data_p_t = get_samples_for_predict_period(bursts, series, seq_len) # [seqs, time_periods]
    t_data_p_v = get_samples_for_predict_value(bursts, series, seq_len) # [seqs, values]
    # print t_data_p_v[1]
    
    Xt_train, Xt_test, yt_train, yt_test = train_test_split(
        t_data_p_t[0], t_data_p_t[1], test_size=0.33, random_state=50)
    Xv_train, Xv_test, yv_train, yv_test = train_test_split(
        t_data_p_v[0], t_data_p_v[1], test_size=0.33, random_state=50)

    # build model
    classifier = build_svm_model(t_data_c[0], t_data_c[1])
    t_pdt = build_prediction_model(Xt_train, yt_train)
    v_pdt = build_prediction_model(Xv_train, yv_train)

    t_rst = []
    v_rst = []
    for i in range(len(Xv_test)):
        testx = trainX[i]
        seq = testx.reshape(testx.shape[1], )

        # predict
        period = predict_burst_period(t_pdt, seq)
        peak = predict_burst_value(v_pdt, seq)

        t_rst.append(period)
        v_rst.append(peak)

    losst = log_loss(yt_test, t_rst)
    lossv = log_loss(yv_test, v_rst)
    errort = mean_squared_error(yt_test, t_rst)
    errorv = mean_squared_error(yv_test, v_rst)
    print losst, lossv, errort, errorv

    # plot
    f, axarr = plt.subplots(2)
    axarr[0].hist(t_rst, facecolor='b')
    axarr[0].set_title('period prediction')
    axarr[0].text('Loss: %.2f, error: %.2f' % (losst, errort),
                  size=16, horizontalalignment='right')
    bins = xrange(0,200,10)
    axarr[1].hist(v_rst, bins, normed=1, facecolor="r")
    axarr[1].set_title('peak prediction')
    axarr[1].text('Loss: %.2f, error: %.2f' % (lossv, errorv),
                  size=16, horizontalalignment='right')
    plt.grid(True)
    plt.show()

"""
(186, 253, 243, 0.81795333357838396);
(253, 293, 266, 1.1332792157840526);
(402, 502, 468, 1.1835322462677593);
(502, 570, 524, 0.80836857865017009);
(570, 723, 592, 0.75588249108580308);
(723, 779, 739, 0.4954501214060974);
(877, 972, 919, 0.5468147754339312);
(1053, 1092, 1081, 0.54845999025733616)
"""