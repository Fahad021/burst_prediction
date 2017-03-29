#!/usr/bin/env python
# coding=utf-8

# get training data for the burst refinement part
# including the classfier and three predict model

import numpy as np
import pandas
import time
import csv
import random


size = 20000
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
    for i in range(series.shape[0]):
        if i not in pid_lines:
            non_burst_series.append(series[i])
            continue

        j = pid_lines.index(i)
        for item in bursts[j]:
            # get whole burst series, without other date
            # boundary = (max(item[0]-seq_len, 0), min(item[1]+seq_len, len(series[i])))
            burst_series.append(series[i][item[0]:item[1]])

    print "bursts, burst_series, non_burst_series", len(bursts), len(burst_series), len(non_burst_series)

    return bursts, burst_series, non_burst_series


N = 7 # 可调参数
def get_samples_for_classfier(series, n_series, seq_len):
    """
    get samples of burst/non burst in the category of predicting series
    include rising and falling part, for timely detecting
    """
    print "get samples for classfier"
    pos_samples = []
    neg_samples = []

    # get positive samples from burst series
    count_i = 0
    for seq in series:
        # for i in xrange(N, len(seq)-N-seq_len):
        for i in xrange(0, len(seq)-seq_len):
            pos_samples.append(seq[i:i+seq_len])
            count_i += 1
            if count_i >= size:
                break
        if count_i >= size:
            break

    # get negative samples from burst series
    count_i = len(pos_samples)
    count = max(len(pos_samples)/len(n_series),1)
    # count = 10
    for seq in n_series:
        for i in range(count):
            start = random.randint(0, len(seq)-seq_len)
            neg_samples.append(seq[start:start+seq_len])
            count_i -= 1
            if count_i <= 0:
                 break
        if count_i <= 0:
             break

    print "pos_samples count, neg_samples count: ", len(pos_samples), len(neg_samples)
    
    return pos_samples, neg_samples


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
    count = max(len(pos_samples)/len(n_series),1)
    # count = 10
    for seq in n_series:
        for i in range(count):
            start = random.randint(0, len(seq)-seq_len)
            neg_samples.append(seq[start:start+seq_len])
            count_i -= 1
            if count_i <= 0:
                break
        if count_i <= 0:
            break

    print "pos_samples count, neg_samples count: ", len(pos_samples), len(neg_samples)
    
    return pos_samples, neg_samples


def get_samples_for_predict_period(bursts, series, seq_len):
    """
    get seq and period samples of burst in the category of predicting series
    """
    print "get samples for predict period"
    samples = []
    periods = []
    k = 0
    for burst in bursts:
        if burst is None or len(burst) == 0:
            continue

        for item in burst:
            # for i in xrange(N, len(series[k])+N-seq_len):
            for i in xrange(0, len(series[k])-seq_len):
                samples.append(series[k][i:i+seq_len])
                periods.append(item[1]-item[0])
            k += 1

    return samples, periods


def get_samples_for_predict_value(bursts, series, seq_len):
    """
    get seq and burst_value samples of burst in the category of predicting series
    """
    print "get samples for predict value"
    samples = []
    values = []
    k = 0
    for burst in bursts:
        if burst is None or len(burst) == 0:
            continue

        for item in burst:
            # for i in xrange(N, len(series[k])+N-seq_len):
            for i in xrange(0, len(series[k])-seq_len):
                samples.append(series[k][i:i+seq_len])
                values.append(item[3])
            k += 1

    return samples, values


def get_samples_for_predict_end_value(bursts, series, seq_len):
    """
    get seq and burst_value samples of burst in the category of predicting series
    """
    print "get samples for predict end value"
    samples = []
    values = []
    periods = []
    st_values = []
    peaks = []
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
            k += 1

    return samples, periods, st_values, peaks, values


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
    samples = series[index:index+count, :] # samples:[burst_len, burst series]
    samples = np.apply_along_axis(lambda a:a[0] + [b-a[1] for b in a[1:]], 1, samples) # make all x[0] = 0
    print "refine sample shape: ", samples.shape
    return samples
