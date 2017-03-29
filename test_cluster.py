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
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from itertools import cycle

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

    bursts, series, n_series = get_category_bursts(sys.argv[1], 
                                                   sys.argv[3], 
                                                   sys.argv[4], 
                                                   sys.argv[2], 
                                                   id_, 
                                                   seq_len)
    t_data_c = get_samples_for_classfier_v2(series, n_series, seq_len) # [pos_samples, neg_samples]
    t_data_p_t = get_samples_for_predict_period(bursts, series, seq_len) # [seqs, time_periods]
    t_data_r = get_samples_for_refine(sys.argv[4], sys.argv[5], sys.argv[6], id_) # [seqs]

    # build model
    af = build_refine_model(t_data_r)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    print "clusters: ", n_clusters_
    print labels

    # build model
    c_X, c_y = prepare_svm_input(t_data_c[0], t_data_c[1])
    classifier = build_svm_model(c_X, c_y)
    t_pdt = build_prediction_model(t_data_p_t[0], t_data_p_t[1])

    i = 0
    rst = []
    while i < X.shape[0]:
        print "index: ", i
        testx = X[i]
        seq = np.reshape(testx,(testx.shape[1],))

        # classfy burst when burst
        rst = classfy_burst(classifier, seq, seq_len)
        if rst == 1: # class 1: burst; class 2: no burst
            # predict period
            period = predict_burst_period(t_pdt, seq)
            new_seq, st, ed, last_p, start = reshape_orginal_seq(seq, period)
            indices, _ = ap_model.predict(np.array(r_seq).reshape(-1,1), st, ed)
            class_members = labels == indices[0]
            tmp = t_data_r[class_members]
            rst.append(np.var(tmp))
            i += period

        else:
            i += 1

    print rst
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    count = []
    for k in range(n_clusters_):
        count.append(np.count_nonzero(labels == k))
    l = np.argsort(count)[-15:]
    labels = af.labels_
    for k, col in zip(l, colors):
        class_members = labels == k
        cluster_center = t_data_r[cluster_centers_indices[k]][1:]
        plt.plot(cluster_center, "g^")
        for x in t_data_r[class_members]:
            plt.plot(x[1:], col+"-")
    plt.text(("%.2f" % np.mean(rst)).lstrip('0'), 
            size=16, horizontalalignment='right')
    plt.show()


# all: 9462  clusters: 240
# cluster size top 10: [299, 304, 316, 318, 325, 349, 354, 408, 410, 492]
# how to decide the number of clusters?!