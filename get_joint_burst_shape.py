#!/usr/bin/env python
# coding=utf-8

# get all burst shape in each category

import csv
import numpy as np
import pandas as pd
import json
import sys


def get_category_pid(filename):
    # first number = line number = category id
    # following numbers are product id
    rd = csv.reader(open(filename, "r"), delimiter=",")
    ids = [] # element: str
    for row in rd:
        ids.append(row[1:])
    return ids


def get_product_ids(filename):
    rd = csv.reader(open(filename, "r"), delimiter=",")
    ids = [] # element: str
    for row in rd:
        ids.append(row[0])
    return ids


def resample_series(series, start, end, sample_n):
    """
    resample burst <=> change the frequency of burst to a fixed time
    sample_n: get sample_n samples
    """
    # print type(series)
    samples = []
    t = float(end - start) / sample_n # time interval to get sample, float
    if (end - start) % sample_n == 0:
        for i in range(sample_n):
            samples.append(series[start + i * int(t)])

    else:
        # treat series k to series k+1 as a straight line
        # get the corresponding value on the line
        for i in range(sample_n):
            k = int(start + i * t)
            sample = series[k] + (series[k+1] - series[k]) * (start + i * t - k)
            samples.append(sample)

    return samples

# load data
bursts = csv.reader(open(sys.argv[1], "r"), delimiter=";")
pid_list = get_product_ids(sys.argv[2])
series = np.genfromtxt(sys.argv[3], delimiter=",") # get smoothed series

# reshape all bursts to a fixed time length
print "reshape bursts"
length = 2*30 # variable!
i = 0
new_bursts = {} # key:value = pid:burst_list
burst_len = {} # key:value = pid:burst_len_list
for line in bursts:
    pid = pid_list[i]
    new_bursts[pid] = []
    burst_len[pid] = []
    for j in range(len(line)):
        burst = eval(line[j])
        # print type(burst)
        resampled_burst = resample_series(series[i], burst[0], burst[1], length)
        new_bursts[pid].append(resampled_burst)
        burst_len[pid].append(burst[1] - burst[0] + 1)
    i += 1

# categorize all bursts
print "categorize"
c_pids = get_category_pid(sys.argv[4]) # get from "grouped_pid.txt"
w1 = csv.writer(open(sys.argv[5], "w"), delimiter=",") # category burst index file
w2 = csv.writer(open(sys.argv[6], "w"), delimiter=",") # burst series file

line_number = 0
for c_id in xrange(len(c_pids)):
    count = 0
    for pid in c_pids[c_id]:
        if pid not in new_bursts:
            continue
        count += len(new_bursts[pid])
        for i in range(len(new_bursts[pid])):
            w2.writerow([burst_len[pid][i]] + new_bursts[pid][i])
    
    #if count == 0:
    #    continue

    w1.writerow([c_id, line_number, count])
    line_number += count
