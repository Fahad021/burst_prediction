#!/usr/bin/env python
# coding=utf-8

# classifier for detecting a burst

import numpy as np
import time

from scipy.signal import argrelextrema
from sklearn import svm
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


n_feature = 19

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
    features['id_min'] = np.argmin(seq)
    # min value
    features['id_max'] = np.argmax(seq)
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
    tmp_l = [seq[i+1] - seq[i] for i in range(end)]
    features['max_fod_point'] = np.argmax(tmp_l)
    # minimum point of the first-order derivative
    features['min_fod_point'] = np.argmin(tmp_l)
    # d-value between positive and negative first-order derivative
    tmp_l = [1 if seq[i+1] - seq[i] >= 0 else 0 for i in range(end)]
    features['d_pfod_nfod'] = tmp_l.count(1) - tmp_l.count(0)

    return features.values()


def prepare_svm_input(pos_samples, neg_samples):
    """
    samples: [seq1, seq2, ...]
    the sample should include the rising and also the falling part of the burst
    """
    print "prepare features for classfier"
    total = len(pos_samples) + len(neg_samples)
    X = np.empty([total, n_feature], dtype=float)
    y = np.empty((total,), dtype=int)
    i = 0
    for pos_seq in pos_samples:
        X[i] = get_features(pos_seq)
        y[i] = 1
        i += 1
        print i
    for neg_seq in neg_samples:
        X[i] = get_features(neg_seq)
        y[i] = 2
        i += 1
        print i

    return X, y

def prepare_svm_input_v2(pos_samples, neg_samples, pos_features, neg_features):
    """
    samples: [seq1, seq2, ...]
    the sample should include the rising and also the falling part of the burst
    """
    print "prepare features for classfier"
    total = len(pos_samples) + len(neg_samples)
    f_n = n_feature + len(pos_features[0])
    X = np.empty([total, f_n], dtype=float)
    y = np.empty((total,), dtype=int)
    i = 0
    for pos_seq in pos_samples:
        X[i] = get_features(pos_seq) + pos_features[i]
        y[i] = 1
        i += 1
    p = X.shape[0]
    for neg_seq in neg_samples:
        X[i] = get_features(neg_seq) + neg_features[i-p]
        y[i] = 2
        i += 1

    return X, y


def build_svm_model(X, y, kernel="rbf"):
    # here not consider unbalanced problems
    # we can get same number of pos samples and neg samples
    start = int(time.time())
    clf = svm.SVC(kernel=kernel) # classify
    
    clf.fit(X,y)
    print "Classifier Fit Time : ", time.time() - start
    
    return clf


def build_clf_model(X, y, model="gaussian"):
    # here not consider unbalanced problems
    # we can get same number of pos samples and neg samples
    start = int(time.time())
    if model == "neighbor":
        clf = KNeighborsClassifier(2)
    elif model == "decision":
        clf = DecisionTreeClassifier(max_depth=5)
    elif model == "adaboost":
        clf = AdaBoostClassifier()
    elif model == "randomforest":
        clf = RandomForestClassifier(max_depth=5, n_estimators=10)
    else:
        clf = GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True) # classify

    clf.fit(X,y)
    print "Classifier Fit Time : ", time.time() - start
    
    return clf

def classfy_burst(clf, seq, seq_len):
    length = len(seq)
    if length != seq_len:
        seq = [0.0]*(seq_len - length) + seq

    # start = argrelextrema(seq, np.less)[0][-1]
    # end = seq_len - 1

    features = np.array(get_features(seq)).reshape(1,-1)

    start = int(time.time())
    rst = clf.predict(features)
    print "Classifier Predict Time : ", time.time() - start
    
    return rst
