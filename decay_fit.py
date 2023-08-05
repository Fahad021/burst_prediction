#!/usr/bin/env python
# coding=utf-8

import numpy as np
import sys
from sklearn.metrics import mean_squared_error


def l(x,v1,v2,k):
    return (v2-v1)/k*x + v1


def get_y(value, value2, k):
    return [l(i,value,value2,k) for i in range(k)]


def get_alpha(k):
    return [np.exp((-1/3)*i) for i in range(k)]


def decay_fit(f,g,a,k):
    tmp = [a[i]*f[i]+(1-a[i])*g[i] for i in range(k)]
    return tmp + list(g[k:])


msq_set = []
pred_set = [] 

ratio = 0.3

for fid in range(64):
    a = np.load("rst/id%d/rst.npz" % fid)
    seq_len = a["seq_len"]
    our_pred = a["our_pred"]
    series = a["series"]

    msqs = []
    preds = []

    for i in range(our_pred.shape[0]):
        try:
            if series[i][seq_len-1] - our_pred[i][0] > 0.1:
                K = len(series[i]) - seq_len
                point = int(K*ratio)
                g = our_pred[i]
                #print K, point, len(g)
                f = get_y(series[i][seq_len-1], g[point-1], point)
                alpha = get_alpha(point)
                y = decay_fit(f,g,alpha,point)
                if i == 396 and fid == 0:
                    print g
                    print y
                    print f
                    print alpha
                msq = mean_squared_error(series[i][seq_len:], y)
            else:
                y = our_pred[i]
                msq = a["msq"][i][1]

            print i, a["msq"][i][1], msq

        except Exception,e:
            print "%d error: %s" % (i,e)
            msq = None
            y = None
        
        msqs.append(msq)
        preds.append(y)

    msq_set.append(msqs)
    pred_set.append(preds)


np.savez("rst/rst_our_decay.npz", msq=msq_set, our_pred=pred_set)