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
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

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
            24993, 60, 263, 102, 
            #365, 13, 6949, 75, 46, 
            #319, 238, 352, 207, 753, 30, 230, 0, 6, 126, 
            #15, 237, 11, 20, 59, 108, 78, 196, 24994, 97, 
            #10, 8, 104, 105, 385, 440, 61, 5, 135, 4, 438, 
            #84, 173, 148, 98, 33, 79, 132, 20482, 119, 28, 
            #2923, 377, 161, 6913, 9
            ] # 64, threshold:100

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

    inputs = []
    for s in S_train[:500]:
        inputs += s.tolist()
    
    # make predictions
    new_S = []
    msq_set = [] # mean square errors, (lstm_error, our_error) pair
    predictions = []
    for series in S_test[:20]:
        ori_p = len(series)
        #print "period", ori_p
        if ori_p <= seq_len:
            continue
        seq = series[0:seq_len]

        # arima
        try:
            arma = sm.tsa.ARMA(inputs+list(seq), (2,1)).fit()
            output = arma.predict(len(inputs)+seq_len, len(inputs)+ori_p, dynamic=True)
            output = output[1:]
        except:
            output = [np.mean(seq)] * (ori_p - seq_len)
            print "error"

        predictions.append(output)

        msq = mean_squared_error(series[seq_len:], output)
        msq_set.append(msq)
        #print msq

    return msq_set,predictions

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)


    rsts = []
    rsts2 = []

    for fid, id_ in enumerate(pid_list):
        scores, pred = task(sys.argv, id_, fid)
        rsts.append(scores)
        rsts2.append(pred)

    np.savez(sys.argv[7], scores=rsts, pred=rsts2)