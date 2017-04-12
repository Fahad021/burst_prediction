#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np

import pandas
import math
import sys
import os
import time
import multiprocessing

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.externals import joblib

from scipy.signal import argrelextrema

from lstm import *
from classifier import *
from burst_refinement import *
from prepare_v2 import *


def get_dates(filename):
    days = np.loadtxt(filename,
                      unpack=True, 
                      converters={0: mdates.strpdate2num('%Y%m%d')})
    # print days[:5]

    return days

pid_list = [118, 6921, 372, 661, 62, 107, 42, 76, 27, 
            24993, 60, 263, 102, 365, 13, 6949, 75, 46, 
            319, 238, 352, 207, 753, 30, 230, 0, 6, 126, 
            15, 237, 11, 20, 59, 108, 78, 196, 24994, 97, 
            10, 8, 104, 105, 385, 440, 61, 5, 135, 4, 438, 
            84, 173, 148, 98, 33, 79, 132, 20482, 119, 28, 
            2923, 377, 161, 6913, 9] # 64, threshold:100

def task(argv, id_, fid):
    seq_len = 30
    
    # load data
    print fid, "load data"
    bursts_info, burst_series, non_burst_series = get_category_bursts(argv[1], 
                                                   argv[3], 
                                                   argv[4], 
                                                   argv[2], 
                                                   id_, 
                                                   seq_len)
    # use 1/3 as test set
    B_train, S_train, N_train, B_test, S_test, N_test = train_test_split(
             bursts_info, burst_series, non_burst_series)
    print fid, "train/test size: ", len(S_train), len(S_test)

    t_data_p_v = get_samples_for_predict_value(B_train, S_train, seq_len) # [seqs, values]
    t_data_p_t = get_samples_for_predict_period(B_train, S_train, seq_len) # [seqs, time_periods]
    t_data_e_v = get_samples_for_predict_end_value(B_train, S_train, seq_len) # [seqs,periods,st_values,peaks,end_values]

    if True:
        # build peak value pred model
        v_X = prepare_prediction_input(t_data_p_v[0])
        v_pdts = []
        v_pdts.append(build_prediction_model(v_X, t_data_p_v[1]))
        v_pdts.append(build_prediction_model(v_X, t_data_p_v[1], "svr"))
        v_pdts.append(build_prediction_model(v_X, t_data_p_v[1], "linear"))
        v_pdts.append(build_prediction_model(v_X, t_data_p_v[1], "bayes"))
        v_pdts.append(build_prediction_model(v_X, t_data_p_v[1], "cart"))

        # build period pred model
        t_X = prepare_period_prediction_input(t_data_p_t[0], t_data_p_t[1])
        t_pdts = []
        t_pdts.append(build_prediction_model(t_X, t_data_p_t[2]))
        t_pdts.append(build_prediction_model(t_X, t_data_p_t[2], "svr"))
        t_pdts.append(build_prediction_model(t_X, t_data_p_t[2], "linear"))
        t_pdts.append(build_prediction_model(t_X, t_data_p_t[2], "bayes"))
        t_pdts.append(build_prediction_model(t_X, t_data_p_t[2], "cart"))

        # build ed value pred model
        e_X = prepare_end_value_prediction_input(t_data_e_v[0], 
                                                 t_data_e_v[1], 
                                                 t_data_e_v[2], 
                                                 t_data_e_v[3])
        e_pdts = []
        e_pdts.append(build_end_value_prediction_model(e_X, t_data_e_v[4]))
        e_pdts.append(build_end_value_prediction_model(e_X, t_data_e_v[4], "svr"))
        e_pdts.append(build_end_value_prediction_model(e_X, t_data_e_v[4], "linear"))
        e_pdts.append(build_end_value_prediction_model(e_X, t_data_e_v[4], "bayes"))
        e_pdts.append(build_end_value_prediction_model(e_X, t_data_e_v[4], "cart"))

        # peak value score
        test_v = get_samples_for_predict_value(B_test, S_test, seq_len)
        test_vx = prepare_prediction_input(test_v[0])
        score2 = []
        for i in range(5):
            y_pred = v_pdts[i].predict(test_vx)
            y = test_v[1]
            hr20 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.2])/float(len(y))
            hr30 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.3])/float(len(y))
            score2.append(mean_squared_error(y_pred, test_v[1]))
            score2.append(mean_absolute_error(y_pred, test_v[1]))
            score2.append([hr20, hr30])

        # period score
        test_t = get_samples_for_predict_period(B_test, S_test, seq_len)
        test_tx = prepare_period_prediction_input(test_t[0], test_t[1])
        score3 = []
        for i in range(5):
            y_pred = t_pdts[i].predict(test_tx)
            y = test_t[2]
            hr20 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.2])/float(len(y))
            hr30 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.3])/float(len(y))
            score3.append(mean_squared_error(y_pred, test_t[2]))
            score3.append(mean_absolute_error(y_pred, test_t[2]))
            score3.append([hr20, hr30])

        # end value score
        test_e = get_samples_for_predict_end_value(B_test, S_test, seq_len)
        test_ex = prepare_end_value_prediction_input(test_e[0], 
                                                     test_e[1], 
                                                     test_e[2], 
                                                     test_e[3])
        score4 = [] # mse, mre, hr20, hr30
        for i in range(5):
            y_pred = e_pdts[i].predict(test_ex)
            y = test_e[4]
            hr20 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.2])/float(len(y))
            hr30 = len([k for k in range(len(y)) if abs(y_pred[k]-y[k])/float(y[k])<=0.3])/float(len(y))
            score4.append(mean_squared_error(y_pred, test_e[4]))
            score4.append(mean_absolute_error(y_pred, test_e[4]))
            score4.append([hr20, hr30])

        print fid, "predict period, peak, end value score: ", score2, score3, score4    

        scores = [score2, score3, score4]

    return scores


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)


    rsts = []

    for fid, id_ in enumerate(pid_list):
        scores = task(sys.argv, id_, fid)
        rsts.append(scores)

    np.savez(sys.argv[7], scores=rsts)