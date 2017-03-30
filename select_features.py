#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np

import pandas
import math
import sys
import os
import time

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib

from scipy.signal import argrelextrema

from lstm import *
from classifier import *
from burst_refinement_v2 import *
from prepare_v3 import *


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
    bursts_info, burst_series, non_burst_series, features, history = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    # use 1/3 as test set
    B_train, S_train, N_train, B_test, S_test, N_test, F_train, F_test, H_train, H_test = train_test_split(
             bursts_info, burst_series, non_burst_series, features, history)
    print "train/test size: ", len(S_train), len(S_test)

    t_data_c = get_samples_for_classfier(S_train, N_train, F_train, seq_len) # [pos_samples, neg_samples]

    # build clf model
    c_X, c_y = prepare_svm_input_v2(t_data_c[0], t_data_c[1], t_data_c[2], t_data_c[3])
    clf = svm.LinearSVC(C=0.01, penalty="l1", dual=False).fit(c_X, c_y)
    model = SelectFromModel(clf, prefit=True)
    features = model.get_support()
    print features

    # save model
    joblib.dump(model, 'model/select_feature.pkl')


