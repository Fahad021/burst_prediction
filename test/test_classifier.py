#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np

import pandas
import math
import sys
import os
import time
import signal
import csv

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score, precision_score
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

def handler(signum, frame):
    raise AssertionError


def test(argv, id_, writer1,writer2,writer3):
    seq_len = 30
    
    option2 = ["neighbor", "decision"]
    option1 = ["rbf", "sigmoid"]


    # load data
    print id_, "load data"
    bursts_info, burst_series, non_burst_series = get_category_bursts(argv[1], 
                                                   argv[3], 
                                                   argv[4], 
                                                   argv[2], 
                                                   id_, 
                                                   seq_len)
    # use 1/3 as test set
    B_train, S_train, N_train, B_test, S_test, N_test = train_test_split(
             bursts_info, burst_series, non_burst_series)
    print "train/test size: ", len(S_train), len(S_test)

    t_data_c = get_samples_for_classfier_v2(S_train, N_train, seq_len) # [pos_samples, neg_samples]

    # build clf model
    print "prepare input"
    c_X, c_y = prepare_svm_input(t_data_c[0], t_data_c[1])
    clfs = []
    for item in option1:
        print "build %s" % item
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(20*60)
            clf = build_svm_model(c_X, c_y, item)
            signal.alarm(0)
            clfs.append(clf)
        except AssertionError:
            print "timeout"
            clfs.append(None)
        
    for item in option2:
        print "build %s" % item
        try:
            signal.signal(signal.SIGALRM, handler)
            signal.alarm(20*60)
            clf = build_clf_model(c_X, c_y, item)
            signal.alarm(0)
            clfs.append(clf)
        except AssertionError:
            print "timeout"
            clfs.append(None)

    # test score
    # classifier score
    print "test classifier"
    dataset = get_samples_for_classfier_v2(S_test, N_test, seq_len)
    test_c_x, test_c_y = prepare_svm_input(dataset[0], dataset[1])
    score1 = []
    score2 = []
    score3 = []
    for clf in clfs:
        if clf is None:
            score1.append(0.0)
            score2.append(0.0)
            score3.append(0.0)
            continue
        score1.append(clf.score(test_c_x, test_c_y))
        y_pred = clf.predict(test_c_x)
        score2.append(precision_score(test_c_y,y_pred))
        score3.append(recall_score(test_c_y,y_pred))
    
    print score1,score2,score3
    #writer1.writerow(score1)
    #writer2.writerow(score2)
    #writer3.writerow(score3)
    return score1,score2,score3
    #return [score2,score3]

pid_list = [#118, 6921, 372, 661, 62, 107, 42, 76, 27, 
            24993, #9
            #60, 263, 102, 365, 13, 6949, 75, 46, 319, 
            238,#19
            #352, 207, 753, 30, 230, 0, 6, 126, 
            #15, 237, 11, 20, 59, 108, 78, 196, 24994, 97, 
            #10, 8, 104, 105, 385, 
            440, #43
            #61, 5, 135, 4, 438, 
            #84, 173, 148, 98, 33, 79, 132, 
            20482,#56
            #119, 28, 2923, 377, 161, 6913, 9
            ]

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print "Usage: series, start_points, bursts, cid_pid, c_burst_index, reshaped_bursts, rst_file"
        exit(0)

    rst1s = []
    rst2s = []
    rst3s = []
    #w1 = csv.writer(open("rst/clf_score1.csv","w"))
    #w2 = csv.writer(open("rst/clf_score2.csv","w"))
    #w3 = csv.writer(open("rst/clf_score3.csv","w"))
    for i, id_ in enumerate(pid_list):
        rst1,rst2,rst3 = test(sys.argv, id_, None, None, None)
        rst1s.append(rst1)
        rst2s.append(rst2)
        rst3s.append(rst3)

    np.savez(sys.argv[7], score1=rst1s, score2=rst2s, score3=rst3s)
 
