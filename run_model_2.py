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

from scipy.signal import argrelextrema

from lstm import *
from classifier import *
from burst_refinement import *
from prepare import *


def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)

    id_ = 2
    seq_len = 30
    epochs  = 100
    
    # load data
    print "load data"
    dataset, X, Y = load_data(sys.argv[1], sys.argv[2], seq_len, id_)
    print dataset.shape, X.shape, Y.shape

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    lstm_data = get_samples_for_lstm(series, seq_len)
    t_data_c = get_samples_for_classfier_v2(series, n_series, seq_len) # [pos_samples, neg_samples]
    t_data_p_t = get_samples_for_predict_period(bursts, series, seq_len) # [seqs, time_periods]
    t_data_p_v = get_samples_for_predict_value(bursts, series, seq_len) # [seqs, values]
    t_data_r = get_samples_for_refine(sys.argv[4], sys.argv[5], sys.argv[6], id_) # [seqs]
    t_data_e_v = get_samples_for_predict_end_value(bursts, series, seq_len) # [seqs,periods,st_values,peaks,end_values]


    # build model
    lstm_model = build_model(seq_len)
    lstm_model.fit(lstm_data[0], lstm_data[1], nb_epoch=epochs, batch_size=1, verbose=2)
    c_X, c_y = prepare_svm_input(t_data_c[0], t_data_c[1])
    classifier = build_svm_model(c_X, c_y)
    t_pdt = build_prediction_model(t_data_p_t[0], t_data_p_t[1])
    v_pdt = build_prediction_model(t_data_p_v[0], t_data_p_v[1])
    e_pdt = build_end_value_prediction_model(t_data_e_v[0], 
                                             t_data_e_v[1], 
                                             t_data_e_v[2], 
                                             t_data_e_v[3],
                                             t_data_e_v[4])
    ap_model = build_refine_model(t_data_r)

    # make predictions
    # to be changed!!
    test_lstm = X[0]
    predictions = np.empty_like(Y)
    for i in xrange(0, X.shape[0]):
        predictions[i] = lstm_model.predict(np.reshape(test_lstm,
                                           (test_lstm.shape[0],1,test_lstm.shape[1])))
        test_lstm[:,0:seq_len-1] = test_lstm[:,1:seq_len]
        test_lstm[:,-1] = predictions[i]

    new_pre = np.empty_like(Y)
    i = 0
    while i < X.shape[0]:
        print "index: ", i
        testx = X[i]
        seq = np.reshape(testx,(testx.shape[1],))

        # classfy burst when burst
        rst = classfy_burst(classifier, seq, seq_len)
        if rst == 1: # class 1: burst; class 2: no burst
            # predict period
            period = predict_burst_period(t_pdt, seq)

            # predict peak
            peak = predict_burst_value(v_pdt, seq)
            ed_value = predict_end_value(e_pdt, seq, period, peak)

            new_seq, st, ed, last_p, start = reshape_orginal_seq(seq, period)
            print "period, peak, ed_value, start_value", period, peak, ed_value, seq[start]
            print ed, st
            end = min(i+start+period, len(new_pre))

            # rst = gaussian_prediction(start, seq[start], period, peak)
            reshaped_rst = knn_prediction(ap_model, t_data_r, new_seq, period, peak, ed_value, st, ed)[0]
            print "predicted reshaped burst", reshaped_rst 
            new_burst = stretch_burst(reshaped_rst, period, seq[0])
            # new_pre[i:i+start] = seq[0:start]
            new_pre[i+start:end] = new_burst[0:end-i-start]
            i += period

        else:
            new_pre[i] = Y[i]
            i += 1

    # invert predictions
    # dataset, new_pre = inverse_data(scaler, [dataset, [new_pre]])

    np.savez(sys.argv[7], new_pred=new_pre, 
                          pred=predictions, 
                          dataset=dataset,
                          seq_len=seq_len)
