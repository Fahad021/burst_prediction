#!/usr/bin/env python
# coding=utf-8

# get training data for the burst refinement part
# including the classfier and three predict model

import numpy as np
import pandas
import time
import csv
import random


def get_features_by_history(seq, has_burst):
    """
    use at most past two window size series data
    """
    end = len(seq) - 1
    if end == -1:
        return [0.0] * 10

    features = dict()
    s_max = np.amax(seq)
    s_min = np.amin(seq)

    features["mean"] = np.mean(seq)
    features["std"] = np.std(seq)
    features["has_burst"] = 1 if has_burst else 0
    features["id_max"] = np.argmax(seq)
    features["id_min"] = np.argmin(seq)
    features['d_last_first'] = seq[-1] - seq[0]
    features['d_last_max'] = seq[end] - s_max
    e_fod = 1/float(end+1) * sum([abs(seq[i+1] - seq[i]) for i in range(end)])
    features['e_fod'] = e_fod
    features['std_fod'] = np.sqrt(1/float(end+1) * sum([np.square(abs(seq[i+1] - seq[i]) - e_fod) for i in range(end)]))
    tmp_l = [1 if seq[i+1] - seq[i] >= 0 else 0 for i in range(-1)]
    features['d_pfod_nfod'] = tmp_l.count(1) - tmp_l.count(0)

    return features.values()


size = 40000 # tag
def get_category_bursts(file1, file2, file3, file4, pid, seq_len):
    """
    get all bursts in the category of predicting series
    """
    print "get category bursts"
    series = np.genfromtxt(file1, delimiter=",") # series
    r2 = csv.reader(open(file2, "r"), delimiter=";") # bursts
    r3 = csv.reader(open(file3, "r")) # category_info
    # stpoint = pandas.read_csv(file4, header=None).values[pid][0] #  start point
    # series = series[stpoint:]

    # get all products in the category
    pid_lines = None
    for line in r3:
        if str(pid) in line:
            pid_lines = [eval(i) for i in line]
            break

    if pid_lines is None:
        return None

    # get all bursts info
    i = 0
    bursts = []
    for line in r2:
        if i not in pid_lines:
            i += 1
            continue

        bursts.append([eval(item) for item in line])
        i += 1

    # get all burst series (add N days before and after)
    burst_series = []
    non_burst_series = []
    features = [] # for bursts
    history = [] # for bursts
    for i in range(series.shape[0]):
        if i not in pid_lines:
            non_burst_series.append(series[i])
            continue

        j = pid_lines.index(i)
        st = 0
        ik = 0
        for item in bursts[j]:
            if ik == 0:
                tag = False
            else:
                tag = True
            # get whole burst series, without other date
            # boundary = (max(item[0]-seq_len, 0), min(item[1]+seq_len, len(series[i])))
            burst_series.append(series[i][item[0]:item[1]])
            start = max(0, item[0]-2*seq_len)
            history.append([tag] + series[i][start:item[0]])
            features.append(get_features_by_history(history[-1], tag))
            st = item[1]
            ik += 1

    print "bursts, burst_series, non_burst_series", len(bursts), len(burst_series), len(non_burst_series)

    return bursts, burst_series, non_burst_series, features, history


def train_test_split(bursts, burst_series, non_burst_series, features, history):
    # bursts = bursts[:2] # tag
    tag = []
    tag2 = []
    for i in range(len(bursts)):
        if i % 3 == 0:
            tag.append(False)
            tag2 += [False]*len(bursts[i])
        else:
            tag.append(True)
            tag2 += [True]*len(bursts[i])

    b_train = [bursts[i] for i in range(len(tag)) if tag[i]]
    s_train = [burst_series[i] for i in range(len(tag2)) if tag2[i]]
    n_train = [non_burst_series[i] for i in range(len(tag2)) if tag2[i]]
    f_train = [features[i] for i in range(len(tag2)) if tag2[i]]
    h_train = [history[i] for i in range(len(tag2)) if tag2[i]]

    b_test = [bursts[i] for i in range(len(tag)) if not tag[i]]
    s_test = [burst_series[i] for i in range(len(tag2)) if not tag2[i]]
    n_test = [non_burst_series[i] for i in range(len(tag2)) if not tag2[i]]
    f_test = [features[i] for i in range(len(tag2)) if not tag2[i]]
    h_test = [history[i] for i in range(len(tag2)) if not tag2[i]]

    return b_train, s_train, n_train, b_test, s_test, n_test, f_train, f_test, h_train, h_test


N = 7 # 可调参数
def get_samples_for_classfier(series, n_series, features, seq_len):
    """
    get samples of burst/non burst in the category of predicting series
    include rising and falling part, for timely detecting
    """
    print "get samples for classfier"
    pos_samples = []
    neg_samples = []
    pos_features = []
    neg_features = []
    start = int(time.time())

    # get positive samples from burst series
    count_i = 0
    for i, seq in enumerate(series):
        # for i in xrange(N, len(seq)-N-seq_len):
        for i in xrange(0, len(seq)-seq_len):
            pos_samples.append(seq[i:i+seq_len])
            pos_features.append(features[i])
            count_i += 1
            if count_i >= size:
                break
        if count_i >= size:
            break

    # get negative samples from burst series
    count_i = len(pos_samples)
    count = max(len(pos_samples)/len(n_series),1)
    # count = 10
    for i, seq in enumerate(n_series):
        for i in range(count):
            start = random.randint(0, len(seq)-seq_len)
            neg_samples.append(seq[start:start+seq_len])
            st = max(0, start-2*seq_len)
            neg_features.append(get_features_by_history(seq[st:start], False))
            count_i -= 1
            if count_i <= 0:
                 break
        if count_i <= 0:
             break

    print "pos_samples count, neg_samples count: ", len(pos_samples), len(neg_samples)
    print "prepare clf time: ", int(time.time()) - start
    return pos_samples, neg_samples, pos_features, neg_features


def get_samples_for_classfier_v2(series, n_series, seq_len):
    """
    get samples of burst/non burst in the category of predicting series
    only include the rising part, for detecting burst start
    """
    print "get samples for classfier"
    pos_samples = []
    neg_samples = []

    # get positive samples from burst series
    count_i = 0
    for seq in series:
        if len(seq) == 0:
            continue

        # for i in xrange(N, np.argmax(seq)-seq_len):
        for i in xrange(0, np.argmax(seq)-seq_len):
            pos_samples.append(seq[i:i+seq_len])
            count_i += 1
            if count_i >= size:
                break
        if count_i >= size:
            break

    # get negative samples from burst series
    count_i = len(pos_samples)
    # count = 10
    for seq in n_series:
        if len(seq) == 0:
            continue
        for i in xrange(0, len(seq)-seq_len):
            neg_samples.append(seq[i:i+seq_len])
            count_i -= 1
            if count_i <= 0:
                break
        if count_i <= 0:
            break

    print "pos_samples count, neg_samples count: ", len(pos_samples), len(neg_samples)
    
    return pos_samples, neg_samples


def get_samples_for_predict_value(bursts, series, features, seq_len):
    """
    get seq and burst_value samples of burst in the category of predicting series
    """
    print "get samples for predict value"
    samples = []
    values = []
    new_features = []
    k = 0
    for burst in bursts:
        if burst is None or len(burst) == 0:
            continue

        for item in burst:
            # for i in xrange(N, len(series[k])+N-seq_len):
            for i in xrange(0, len(series[k])-seq_len):
                samples.append(series[k][i:i+seq_len])
                new_features.append(features[k])
                values.append(item[3])
            k += 1
            if k >= size:
                break
        if k >= size:
            break

    return samples, new_features, values,


def get_samples_for_predict_period(bursts, series, features, seq_len):
    """
    get seq and period samples of burst in the category of predicting series
    """
    print "get samples for predict period"
    samples = []
    periods = []
    peaks = []
    new_features = []
    k = 0
    for burst in bursts:
        if burst is None or len(burst) == 0:
            continue

        for item in burst:
            # for i in xrange(N, len(series[k])+N-seq_len):
            for i in xrange(0, len(series[k])-seq_len):
                samples.append(series[k][i:i+seq_len])
                periods.append(item[1]-item[0])
                new_features.append(features[k])
                peaks.append(item[3])
            k += 1
            if k >= size:
                break
        if k >= size:
            break

    return samples, peaks, new_features, periods


def get_samples_for_predict_end_value(bursts, series, features, seq_len):
    """
    get seq and burst_value samples of burst in the category of predicting series
    """
    print "get samples for predict end value"
    samples = []
    values = []
    periods = []
    st_values = []
    peaks = []
    new_features = []
    k = 0
    for burst in bursts:
        if burst is None or len(burst) == 0:
            continue

        for item in burst:
            # for i in xrange(N, len(series[k])+N-seq_len):
            for i in xrange(0, len(series[k])-seq_len):
                samples.append(series[k][i:i+seq_len])
                periods.append(item[1]-item[0])
                st_values.append(series[k][0])
                peaks.append(item[3])
                values.append(series[k][-1])
                new_features.append(features[k])
            k += 1
            if k >= size:
                break
        if k >= size:
            break

    return samples, periods, st_values, peaks, new_features, values


def get_samples_for_refine(file1, file2, file3, pid):
    """
    get reshaped bursts in the category of predicting series
    """
    print "get samples for refine"
    r1 = csv.reader(open(file1, "r")) # category_info
    r2 = csv.reader(open(file2, "r")) # category burst index
    series = np.genfromtxt(file3, delimiter=",") # reshaped bursts
    cid = 0

    # get category id
    i = 0
    for line in r1:
        if str(pid) in line:
            cid = i
            break
        i += 1

    # get category burst index and count
    index = 0
    count = 0
    for line in r2:
        if eval(line[0]) == cid:
            index = eval(line[1])
            count = eval(line[2])
            break

    # get all reshaped bursts in this category
    # samples = series[index:index+count, :] # samples:[burst_len, burst series]
    samples = series[index:index+count, :] # tag
    samples = np.apply_along_axis(lambda a:[b-a[1] if type(b) is float else b for b in a], 1, samples) # make all x[0] = 0

    print "refine sample shape: ", samples.shape

    # get 2/3 data as training data
    data = []
    for i in range(samples.shape[0]):
        if i % 3 == 0:
            data.append(samples[i,:])
    
    print "refine dataset shape: ", len(data)
    return data
