#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys


def plot_results(our_pred, pred, msq, series, scores, seq_len):
    for id_ in range(10,11):
        plt.plot(series[id_], "b-")
        plt.plot(pred[id_], "r-")
        plt.plot(our_pred[id_], "y-")
        #plt.text(1, 1, "msq: %.4f" % msq[id_], style='italic',
        #    bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.show()

rst = np.load(sys.argv[1])
plot_results(rst["our_pred"], 
             rst["pred"],
             rst["msq"],
             rst["series"],
             rst["scores"],
             rst["seq_len"])
