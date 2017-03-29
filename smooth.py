#!/usr/bin/env python
# coding=utf-8

import sys
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl

# using exponential smoothing with kalman filter

def predict(pos, movement):
    return (pos[0] + movement[0], pos[1] + movement[1])


def gaussian_multiply(g1, g2):
    mu1, var1 = g1
    mu2, var2 = g2
    mean = (var1 * mu2 + var2 * mu1) / (var1 + var2)
    variance = (var1 * var2) / (var1 + var2)
    return (mean, variance)


def update(prior, likelihood):
    posterior = gaussian_multiply(likelihood, prior)
    return posterior


series_matrix = np.genfromtxt(sys.argv[1], delimiter=" ")
(m, n) = series_matrix.shape

smooth_series = np.ndarray(shape=(m, n-1), dtype=float)
start_points = []
for i in xrange(0, m):
    start_points.append(next((j for j, x in enumerate(series_matrix[i, 1:]) if int(x) > 0), None))

    zs = pd.ewma(series_matrix[i, 1:], span=14)

    voltage_std = 1.8
    process_var = .1**2

    x = (25, 1000) # initial state
    process_model = (0., process_var)

    estimates = []

    for z in zs:
        prior = predict(x, process_model)
        x = update(prior, (z, voltage_std**2))

        estimates.append(x[0])
        
    smooth_series[i] = estimates

np.savetxt(sys.argv[2], smooth_series, delimiter=",")

# get series start point
np.savetxt(sys.argv[3], start_points, fmt='%d', delimiter=",")