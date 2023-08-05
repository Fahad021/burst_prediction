#!/usr/bin/env python
# coding=utf-8

# test period/peak prediction model

import numpy as np

import pandas as pd
import math
import sys
import time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import Series, DataFrame
from pandas.tools.plotting import scatter_matrix

from lstm import *
from classifier import *
from burst_refinement import *
from prepare import *

def get_features_2(seqs):
    """
    get time series features in [t-k,t]
    currently not consider other kind of features,
    like cateogory sales, product info, etc.
    """
    features = {'mean_value':[],
                'std_value':[],
                'd_last_first':[],
                'd_last_max':[],
                'd_last_min':[],
                'd_max_min':[],
                'id_max':[],
                'id_min':[],
                'e_fod':[],
                'std_fod':[],
                'last_fod':[],
                'max_fod':[],
                'd_pfod_nfod':[]}

    for seq in seqs:
        end = len(seq) - 1
        s_max = np.amax(seq)
        s_min = np.amin(seq)

        # mean value
        features['mean_value'].append(np.mean(seq))
        # std value
        features['std_value'].append(np.std(seq))
        # d-value between the last minute and the first
        features['d_last_first'].append(seq[end] - seq[0])
        # d-value between the last minute and the max
        features['d_last_max'].append(seq[end] - s_max)
        # d-value between the last minute and the min
        features['d_last_min'].append(seq[end] - s_min)
        # d-value between the max and the min
        features['d_max_min'].append(s_max - s_min)
        # index of the max point
        #!! here not consider the situation of multiple max time
        features['id_max'].append(np.argmax(seq))
        # index of the min point
        features['id_min'].append(np.argmin(seq))
        # mean value of the absolute first-order derivative
        e_fod = 1/float(end+1) * sum(abs(seq[i+1] - seq[i]) for i in range(end))
        features['e_fod'].append(e_fod)
        # standard deviation of the absolute first-order derivative
        features['std_fod'].append(
            np.sqrt(
                1
                / float(end + 1)
                * sum(
                    np.square(abs(seq[i + 1] - seq[i]) - e_fod)
                    for i in range(end)
                )
            )
        )
        # last value of the first-order derivative
        features['last_fod'].append(seq[end] - seq[end-1])
        # maximum value of the first-order derivative
        max_fod = max(abs(seq[i+1] - seq[i]) for i in range(end))
        features['max_fod'].append(max_fod)
        # d-value between positive and negative first-order derivative
        tmp_l = [1 if seq[i+1] - seq[i] >= 0 else 0 for i in range(end)]
        features['d_pfod_nfod'].append(tmp_l.count(1) - tmp_l.count(0))

    return features

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
    t_data_p_t = get_samples_for_predict_period(bursts, series, seq_len) # [seqs, time_periods]
    t_data_p_v = get_samples_for_predict_value(bursts, series, seq_len) # [seqs, values]
    """    
    # dataset dist
    features_t = DataFrame(get_features_2(t_data_p_t[0]))
    features_v = DataFrame(get_features_2(t_data_p_v[0]))

    #print features_t
    cols_1 = ['mean_value',
                'std_value',
                'd_last_first',
                'd_last_max',
                'd_last_min',
                'd_max_min',]
    cols_2 = ['id_max',
                'id_min',
                'e_fod',
                'std_fod',
                'last_fod',
                'max_fod',
                'd_pfod_nfod']
    ax = scatter_matrix(features_t[cols_1], figsize=(12, 12), c='red')
    plt.show()
    ax = scatter_matrix(features_t[cols_2], figsize=(12, 12), c='red')
    plt.show()
    ax = scatter_matrix(features_v[cols_1], figsize=(12, 12), c='red')
    plt.show()
    ax = scatter_matrix(features_v[cols_2], figsize=(12, 12), c='red')
    plt.show()
    """
    f, axarr = plt.subplots(2)
    axarr[0].hist(t_data_p_t[1], normed=1, facecolor='b')
    axarr[0].set_title('period data distribution')
    bins = xrange(0,100,10)
    axarr[1].hist(t_data_p_v[1], normed=1, facecolor="r")
    axarr[1].set_title('peak data distribution')
    plt.grid(True)
    plt.show()
