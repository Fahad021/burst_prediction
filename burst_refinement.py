#!/usr/bin/env python
# coding=utf-8

# burst prediction refinement, using category information
# 1. predict burst time period - SVR/LinearSVR/LinearRegression
# 2. predict burst value - SVR/LinearSVR/LinearRegression
# 3. predict/fit burst shape
# 4. use info above to refine the predicted t+1 value

import numpy as np
import time
import math

from sklearn import svm
from sklearn import linear_model
from scipy import stats 
from sklearn import cluster
from sklearn import tree
from scipy.signal import argrelextrema
from numpy.linalg import norm

from AffinityPropagation import *


n_feature = 15

def get_features(seq):
    """
    get time series features in [t-k,t]
    currently not consider other kind of features,
    like cateogory sales, product info, etc.
    """
    end = len(seq) - 1
    features = dict()
    s_max = np.amax(seq)
    s_min = np.amin(seq)

    # max value
    features['min'] = s_min
    # min value
    features['max'] = s_max
    # max value
    # features['id_min'] = np.argmin(seq)
    # min value
    # features['id_max'] = np.argmax(seq)
    # mean value
    features['mean_value'] = np.mean(seq)
    # std value
    features['std_value'] = np.std(seq)
    # d-value between the last minute and the first
    features['d_last_first'] = seq[end] - seq[0]
    # d-value between the last minute and the max
    features['d_last_max'] = seq[end] - s_max
    # d-value between the last minute and the min
    features['d_last_min'] = seq[end] - s_min
    # d-value between the max and the min
    features['d_max_min'] = s_max - s_min
    # mean value of the absolute first-order derivative
    e_fod = 1/float(end+1) * sum([abs(seq[i+1] - seq[i]) for i in range(end)])
    features['e_fod'] = e_fod
    # standard deviation of the absolute first-order derivative
    features['std_fod'] = np.sqrt(1/float(end+1) * sum([np.square(abs(seq[i+1] - seq[i]) - e_fod) for i in range(end)]))
    # last value of the first-order derivative
    features['last_fod'] = seq[end] - seq[end-1]
    # first value of the first-order derivative
    features['first_fod'] = seq[1] - seq[0]
    # maximum value of the first-order derivative
    tmp_l = [abs(seq[i+1] - seq[i]) for i in range(end)]
    features['max_fod'] = max(tmp_l)
    # minimum value of the first-order derivative
    features['min_fod'] = min(tmp_l)
    # maximum point of the first-order derivative
    # tmp_l = [seq[i+1] - seq[i] for i in range(end)]
    # features['max_fod_point'] = np.argmax(tmp_l)
    # minimum point of the first-order derivative
    # features['min_fod_point'] = np.argmin(tmp_l)
    # d-value between positive and negative first-order derivative
    tmp_l = [1 if seq[i+1] - seq[i] >= 0 else 0 for i in range(end)]
    features['d_pfod_nfod'] = tmp_l.count(1) - tmp_l.count(0)

    return features.values()


def prepare_prediction_input(seqs):
    i = 0
    n_X = len(seqs)
    X = np.empty([n_X, n_feature], dtype=float)
    for seq in seqs:
        X[i] = get_features(seq)
        i += 1
    return X


def prepare_period_prediction_input(seqs, peaks):
    n_X = len(seqs)
    X = np.empty([n_X, n_feature+1], dtype=float)
    i = 0
    for seq in seqs:
        X[i] = get_features(seq) + [peaks[i]]
        i += 1

    return X

n_size = 10000
def build_prediction_model(X, y, type_="linearsvr"):
    start = int(time.time())

    if type_ == "svr":
        model = svm.SVR() # SVR regression
        X = X[:n_size,:]
        y = y[:n_size]
    elif type_ == "linear":
        model = linear_model.LinearRegression() # linear regression
    elif type_ == "bayes":
        model = linear_model.BayesianRidge() # Bayes
    elif type_ == "cart":
        model = tree.DecisionTreeRegressor() # CART
    else:
        model = svm.LinearSVR() # LinearSVR regression

    model.fit(X,y)
    print "T/H Prediction Model Fit Time : ", time.time() - start
    
    return model


def prepare_end_value_prediction_input(seqs, periods, st_values, peaks):
    n_X = len(seqs)
    X = np.empty([n_X, n_feature+3], dtype=float)
    i = 0
    for seq in seqs:
        X[i] = get_features(seq) + [periods[i], st_values[i], peaks[i]]
        i += 1

    return X


def build_end_value_prediction_model(X, y, type_="linearsvr"):
    start = int(time.time())

    if type_ == "svr":
        model = svm.SVR() # SVR regression
        X = X[:n_size,:]
        y = y[:n_size]
    elif type_ == "linear":
        model = linear_model.LinearRegression() # linear regression
    elif type_ == "bayes":
        model = linear_model.BayesianRidge() # Bayes
    elif type_ == "cart":
        model = tree.DecisionTreeRegressor() # CART
    else:
        model = svm.LinearSVR()

    model.fit(X, y)
    print "End Value Prediction Model Fit Time : ", time.time() - start

    return model


def build_refine_model(seqs):
    """
    seqs = [burst_seq1, burst_seq2, ...]
    use the seq that reshaped in the fixed length -> use relative value
    """
    print "build refine model"
    X = np.array(seqs)
    X = np.delete(X, [0], axis=1) # delete 'burst len'
    start = int(time.time())
    af = AffinityPropagation(affinity="euclidean",
                             damping=.8).fit(X) # euclidean distance
    # af = AffinityPropagation(affinity='dtw').fit(X) # dtw distance
    print "AP Cluster Model Fit Time : ", time.time() - start

    return af


def predict_burst_value(model, seq):
    features = get_features(seq)

    start = int(time.time())
    rst = model.predict(np.array(features).reshape(1,-1))
    print "H Predict Time : ", time.time() - start

    return rst[0]


def predict_burst_period(model, peak, seq):
    """
    for time series
    return value: int
    """
    features = get_features(seq) + [peak]

    start = int(time.time())
    rst = model.predict(np.array(features).reshape(1,-1))
    print "T Predict Time : ", time.time() - start
    
    return int(rst)


def predict_end_value(model, seq, period, peak):
    features = get_features(seq) + [period, seq[0], peak]

    start = int(time.time())
    rst = model.predict(np.array(features).reshape(1,-1))
    print "End Value Predict Time : ", time.time() - start
    
    return rst[0]


length = 2*30
def reshape_orginal_seq(seq, burst_period, time_step=0):
    """
    use burst_period and value to reshape seq into the fixed length
    """
    start = 0
    # get the seq only containing the burst part
    # if time_step == 0:
    #     start = 0
    #     tmp = argrelextrema(seq, np.less)[0]
    #     if len(tmp) > 0:
    #         start = tmp[-1]
    #     time_step = -start

    # if time_step < 0:
    #     seq = seq[-time_step:]
    #     start = 0
    # else:
    #     start = time_step

    # reshape the seq using burst period
    t = float(burst_period) / length
    new_seq = []
    st_point = -1
    ed_point = -1
    seq_last_point = -1
    if burst_period % length == 0:
        for i in range(length):
            if i * t >= start + len(seq):
                # out of the seq boundary
                ed_point = i - 1
                break
            if i * t >= start:
                new_seq.append(seq[i * int(t) - start])
                if st_point < 0:
                    st_point = i
                seq_last_point = i * int(t) - start
    else:
        for i in range(length):
            if i * t + 1 >= start + len(seq):
                # out of the seq boundary
                ed_point = i - 1
                break
            if i * t >= start:
                k = int(i * t - start)
                sample = seq[k] + (seq[k+1] - seq[k]) * (i * t - k)
                new_seq.append(sample)
                if st_point < 0:
                    st_point = i 
                seq_last_point = k

    return new_seq, st_point, ed_point, seq_last_point, start

Number = 20 # number of nns
def refine(model, X, seq, new_seq, period, st_point, ed_point, seq_last_point):
    """
    1: use AP to cluster
    2: find where current seq belongs
    3: use the mean value of all t+1 values in cluster to refine t+1 value
    """
    # refine the value
    r_seq = [x - new_seq[0] for x in new_seq]
    start = int(time.time())
    indices,_ = model.predict(np.array(r_seq).reshape(-1,1), st_point, ed_point)
    indice = indices[0] # get the label/cluster of predicted data
    print "Categorize Time : ", time.time() - start

    # find knn with period and shape
    cluster_centers_indices = model.cluster_centers_indices_
    labels = model.labels_

    values = []
    for i in range(1):
        # get all bursts in this label(cluster)
        class_members = labels == indice
        distance_sim = []
        for x in X[class_members]:
            x = x[1:]
            x = x[st_point:ed_point+1]
            distance_sim.append(norm(x - np.array(r_seq)))

        index_sorted = np.argsort(distance_sim)
        knns = [X[class_members][j] for j in index_sorted[-Number:]]
        values.append(np.mean(knns, axis=0)[1:])

    # get the real next time value
    value = seq[-1] + (values[0][ed_point+1] - seq[-1]) * (1 - (len(seq)-1-seq_last_point)/len(seq))
    
    return value

"""
def predict_burst_peak_v2(model, new_seq, st_point, ed_point):
    start = int(time.time())
    values = model.predict_value(np.array(new_seq).reshape(-1,1), st_point, ed_point)[0] # euclidean distance
    peak = values.max()
    print "Peak Predict Time : ", time.time() - start
    return peak
"""

def gaussian_prediction(st_point, st_value, burst_period, burst_value):
    # sigma = 1 / (math.sqrt(2*math.pi)*(burst_value - st_value))
    peak = 1 / (math.sqrt(2*math.pi)*1.0)

    x = np.arange(-(burst_period/2), burst_period-burst_period/2, 1.0)
    y = stats.norm.pdf(x, 0, 1)
    y = [i*(burst_value-st_value)/peak+st_value for i in y]
    
    return y

def knn_prediction(ap_model, X, new_seq, period, peak, ed_value, st_point, ed_point):
    """
    1. categorize
    2. use T and shape to find knn
    3. avg
    """
    r_seq = [x - new_seq[0] for x in new_seq] # let seq[0] == X[:][0]
    X = np.array(X)
    start = int(time.time())
    indices,_ = ap_model.predict(np.array(r_seq).reshape(-1,1), st_point, ed_point)
    indice = indices[0] # get the label/cluster of predicted data
    print "Categorize Time : ", time.time() - start, "cluster: ", indice

    
    if indice == 0:
        return None

    # find knn with period and shape
    cluster_centers_indices = ap_model.cluster_centers_indices_
    labels = ap_model.labels_

    predictions = []
    for i in range(1):
        # get all bursts in this label(cluster)
        class_members = labels == indice
        distance_sim = []
        period_sim = []
        peak_sim = []
        ed_sim = []
        for x in X[class_members]:
            # count T/period similarity
            # x[0]: burst len
            period_sim.append(abs(1-float(x[0])/period))
            if ed_value - 0.0 <= 0.0001:
                ed_sim.append(x[-1])
            else:
                ed_sim.append(abs(1-x[-1]/ed_value))

            # count similarity of shape
            x = x[1:] # delete burst len
            x = x[st_point:ed_point+1]
            distance_sim.append(norm(x - np.array(r_seq)))
            peak_sim.append(abs(1-max(x)/peak))

        rank = np.argsort(distance_sim).argsort() # weight: 2
        rank += rank
        rank += np.argsort(period_sim).argsort()
        rank += np.argsort(peak_sim).argsort()
        rank += np.argsort(ed_sim).argsort()

        knns = [X[class_members][j] for j in rank.argsort()[0:Number]]
        predictions.append(np.mean(knns, axis=0)[1:])
    
    return predictions


def stretch_burst(seq, period, st_value):
    """ 
    use burst period to stretch the burst
    """
    if period < 1 or type(seq) is not np.ndarray:
        return None

    new_seq = np.empty((period,), dtype=float)
    per = float(length) / period
    for i in xrange(0, period):
        i_ = int(i * per)
        if i_ + 1 >= length:
            new_seq[i] = st_value + seq[i_] + (seq[i_]-seq[i_-1])*(i*per - i_)
        else:
            new_seq[i] = st_value + seq[i_] + (seq[i_+1]-seq[i_])*(i*per - i_)

    return new_seq
