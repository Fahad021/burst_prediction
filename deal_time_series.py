#!/usr/bin/env python
# coding=utf-8

import numpy as np
import math
import sys
import scipy
import csv


# load data from file
series = np.genfromtxt(sys.argv[1], delimiter=" ")

m, n = series.shape
print m,n

# get row and column mean
s_c_mean = series.mean(axis=1) # s_c_mean, all in one column
s_t = series.sum(axis=0) # s^t, all in one row
s_mean = s_t.mean() # s_mean, number

# print s_c_mean

# replace s_c^t with (s_c^t / s_c_mean) / (s^t / s_mean)
for i in xrange(0, m):
    # keep id column for following operations
    series[i][0] = i
    for j in xrange(1, n):
        series[i][j] = (series[i][j]/s_c_mean[i])/(s_t[j]/s_mean)

# save
# new tragectory has no line number in the first one
np.savetxt(sys.argv[2], series, delimiter=" ")
# series.tocsv(sys.argv[2])