#!/usr/bin/env python
# coding=utf-8

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys


def plot_results(our_pred, pred, msq, test_set, scores, seq_len):
    id_ = 0
    plt.plot(test_set[id_], "b-")
    plt.plot(pred[id_], "r-")
    plt.plot(our_pred, "m-")
    plt.text(1, 1, "msq: %.4f" % msq[id_], style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})
    plt.text(0.01, 0.95, "clf, period, peak, end value" % scores,
        verticalalignment='bottom', horizontalalignment='right',
        color='green', fontsize=15)
    plt.show()

rst = np.load(sys.argv[1])
plot_results(rst["our_pred"], 
             rst["pred"],
             rst["msq"],
             rst["test_set"],
             rst["scores"],
             rst["seq_len"])
