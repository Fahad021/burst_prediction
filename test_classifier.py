#!/usr/bin/env python
# coding=utf-8

# test classfy model

import numpy as np

import pandas
import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

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
    print dataset.shape, testX.shape, testY.shape

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    t_data_c = get_samples_for_classfier(series, n_series, seq_len) # [pos_samples, neg_samples]
    X, y = prepare_svm_input(t_data_c[0], t_data_c[1])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=50)
    
    # build model
    clf = build_svm_model(X_train, y_train)
    score = clf.score(X_test, y_test)

    class1 = [] # burst
    class2 = [] # no burst
    for i in range(len(x_test)):
        testx = x_test[i]
        if clf.predict(testx) <= 1:
            class1.append(i)
        else:
            class2.append(i)

    print len(class1), len(class2)
    print score

    # plot
    x = range(len(seq))
    for seq in class1:
        plt.plot(x, seq, "b-")
    for seq in class2:
        plt.plot(x, seq, "r-")
    plt.text(("%.2f" % score).lstrip('0'), 
            size=16, horizontalalignment='right')
    plt.show()



# 141,489