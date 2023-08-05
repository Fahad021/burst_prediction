#!/#!/usr/bin/env python
# coding=utf-8

# draw series graph

import datetime
import sys
import csv

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )

# matplotlib.style.use('ggplot')

data = np.genfromtxt(sys.argv[1], delimiter=",")
dates = get_dates(sys.argv[2])
key = 2
print(len(data[key]),len(dates))
plt.plot_date(dates,data[key],'-')

# get bursts
bursts = csv.reader(open(sys.argv[3],"r"), delimiter=";")
iii = 0
for l in bursts:
    if iii > key:
        break
    if iii != key:
        iii += 1
        continue
    for i in l:
        i = eval(i)
        plt.plot_date(dates[int(i[2])], float(i[3]), "or")
        plt.plot_date(dates[int(i[1])], data[key][int(i[1])], "og")
        plt.plot_date(dates[int(i[0])], data[key][int(i[0])], "og")
    iii += 1

plt.show()
