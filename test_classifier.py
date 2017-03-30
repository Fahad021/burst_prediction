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
from burst_refinement import *
from prepare_v2 import *


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
    
    option2 = ["gaussian", "neighbor", "decision", "adaboost", "randomforest"]
    option1 = ["linearsvc","rbf", "poly", "sigmoid"]

    # load data
    print "load data"
    bursts_info, burst_series, non_burst_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    # use 1/3 as test set
    B_train, S_train, N_train, B_test, S_test, N_test = train_test_split(
             bursts_info, burst_series, non_burst_series)
    print "train/test size: ", len(S_train), len(S_test)

    t_data_c = get_samples_for_classfier_v2(S_train, N_train, seq_len) # [pos_samples, neg_samples]

    # build clf model
    c_X, c_y = prepare_svm_input(t_data_c[0], t_data_c[1])
    clfs = []
    for item in option1:
        clf = build_svm_model(c_X, c_y, item)
        clfs.append(clf)
    for item in option2:
        clf = build_clf_model(c_X, c_y, item)
        clfs.append(clf)

    # test score
    # classifier score
    print "test classifier"
    dataset = get_samples_for_classfier_v2(S_test, N_test, seq_len)
    test_c_x, test_c_y = prepare_svm_input(dataset[0], dataset[1])
    score1 = []
    for clf in clfs:
        score1.append(clf.score(test_c_x, test_c_y))
    print "clf scores: ", score1

    scores = [score1]

    selected_ids = [np.argmax(score) for score in scores]
    clf = globals()['clf' + str(selected_ids[0] + 1)]


    # save models and scores
    joblib.dump(clf, 'model/best_clf.pkl')

