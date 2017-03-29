#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np

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


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)

    id_ = 2
    seq_len = 30
    epochs  = 100
    
    # load data
    print "load data"
    dataset, trainX, trainY, testX, testY = load_data(sys.argv[1], sys.argv[2], seq_len, id_)
    print dataset.shape, trainX.shape, testX.shape, testY.shape

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    t_data_c = get_samples_for_classfier(series, n_series, seq_len) # [pos_samples, neg_samples]
    t_data_p_t = get_samples_for_predict_period(bursts, series, seq_len) # [seqs, time_periods]
    t_data_p_v = get_samples_for_predict_value(bursts, series, seq_len) # [seqs, values]
    t_data_r = get_samples_for_refine(sys.argv[4], sys.argv[5], sys.argv[6], id_) # [seqs]

    # build model
    lstm_model = build_model(seq_len)
    lstm_model.fit(trainX, trainY, nb_epoch=epochs, batch_size=1, verbose=2)
    classifier = build_svm_model(t_data_c[0], t_data_c[1])
    t_pdt = build_prediction_model(t_data_p_t[0], t_data_p_t[1])
    v_pdt = build_prediction_model(t_data_p_v[0], t_data_p_v[1])
    refiner = build_refine_model(t_data_r)

    # make predictions
    predictions = lstm_model.predict(testX)
    new_pre = np.empty_like(testY)
    time_step = 0
    for i in range(testX.shape[0]):

        testx = testX[i]
        seq = testx.reshape(testx.shape[1], )

        # classfy burst
        rst = classfy_burst(classifier, seq, seq_len)
        if rst != 1: # class 1: burst; class 2: no burst
            # time_step = 0
            new_pre[ii] = predictions[i]
            continue

        # time_step += 1
        # predict period and peak
        period = predict_burst_period(t_pdt, seq)
        peak = predict_burst_value(v_pdt, seq)
        print "period, peak", period, peak

        # get reshaped seq
        new_seq, st, ed, last_p, start = reshape_orginal_seq(seq, period)
        print ed, st
        
        # refine output t+1 value
        new_pre[i] = refine(refiner, t_data_r, seq, new_seq, period, st, ed, last_p)
        print "predicted: ", new_pre[i]

    # invert predictions
    # dataset, new_pre = inverse_data(scaler, [dataset, [new_pre]])

    np.savez(sys.argv[7], new_pred=new_pre, 
                          pred=predictions, 
                          dataset=dataset, 
                          train_size=len(trainX),
                          seq_len=seq_len)
