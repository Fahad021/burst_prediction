#!/usr/bin/env python
# coding=utf-8

from run_v5.py import *

pid_list = [118, 6921, 372, 661, 62, 107, 42, 76, 27, 
            24993, 60, 263, 102, 365, 13, 6949, 75, 46, 
            319, 238, 352, 207, 753, 30, 230, 0, 6, 126, 
            15, 237, 11, 20, 59, 108, 78, 196, 24994, 97, 
            10, 8, 104, 105, 385, 440, 61, 5, 135, 4, 438, 
            84, 173, 148, 98, 33, 79, 132, 20482, 119, 28, 
            2923, 377, 161, 6913, 9] # 64, threshold:100

final_msq = []
final_score = []
rsts = []

alpha = 1.5 # peak
beta = 10 # period
delta = 0.8 # end_value

for i, id_ in enumerate(pid_list):
    rst = task(sys.argv, id_, i, alpha, beta, delta)
    rsts.append(rst)

for rst in rsts:
    final_score.append([max(l) for l in rst[0]])
    final_msq.append(rst[1])

np.savez(argv[7], msq=final_msq, score=final_score)