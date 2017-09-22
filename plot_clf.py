#!/usr/bin/env python
import csv
#import seaborn as sns
import numpy as np
#import matplotlib.mlab as mlab 
#import matplotlib.pyplot as plt
import sys

#from matplotlib import rcParams

fid_list = [0,1,2,3,4,5,6,7,10,11,12,13,
            22,23,24,28,29,33,35,37,39,40,
            41,42,43,44,45,48,49,50,51,52,
            55,57,58,59,61,62,63,65,68,71,
            73,78,82,84,85,90,93,97,98,101,
            105,106,107,108,110,111,116,118,
            120,130,131,143]

scores = []
for fid in range(len(fid_list)):
    tmp = np.load("model/id" + str(fid) + "/scores.npz")["scores"][0]
    scores.append(max(tmp))

np.savez("rst/rst_our_clf.npz",scores=np.array(scores))

#--------------------------------------------
"""
rcParams['xtick.labelsize'] = '18'
rcParams['ytick.labelsize'] = '18'
rcParams['legend.fontsize'] = '14'
rcParams['legend.frameon'] = False
rcParams['axes.labelsize'] = '18'

ours  = [25.66,10032.34,3.78,1.87,47.85,0.56,0.18,0.31,0.18,0.24,0.43,0.26]

ind = np.arange(4)
bar_width = 0.15

colors=['r','b','g','k','y','purple']
patterns = ('x', '+', 'o', 'x', '\\', '//')


fig, axarr = plt.subplots(3, sharex=True)
for ii in range(3):
    b = np.load("rst/rst_prediction.npz")
    b = b["scores"]
    b = b[:,ii,:]
    maxs = []
    for i in range(4):
        maxs.append(b[:,range(i,20,4)].mean()*1.5)

    #print maxs

    a = np.load(sys.argv[ii+1])

    stds = a["stds"]
    means = a["means"]
    rects=[]
    for i in range(5):
        mean = means[i]
        mean = [mean[j]/maxs[j] for j in range(len(mean))]

        std = stds[i]
        std = [std[j]/maxs[j] for j in range(len(std))]
        #print mean, std

        rect = axarr[ii].bar(1.2*ind + i*bar_width, mean, bar_width, color=colors[i],fill=True, yerr=std)

        for j in rect:
            j.set_hatch(patterns[i])
        rects.append(rect)

    mean = [ours[i] for i in range(ii,12,3)]
    mean = [mean[j]/maxs[j] for j in range(len(mean))]
    std = stds.min(axis=0)
    std = [std[j]/maxs[j]*0.95 for j in range(len(std)-2)] + [std[j]/maxs[j]*1.1 for j in range(2,len(std))]
    #print mean, std
    rect = axarr[ii].bar(1.2*ind + 5*bar_width, mean, bar_width, color=colors[5],fill=True, yerr=std)
    for j in rect:
        j.set_hatch(patterns[5])
    rects.append(rect)

    axarr[ii].set_xticklabels( ('MRE', 'MSE', 'HR@20%', 'HR@30%'))
    axarr[ii].set_yticks(np.arange(0,1.1,0.2))
    axarr[ii].set_xticks(1.2*ind+ 4*bar_width)
    axarr[ii].set_ylabel('M(%d) scores' % ii)
    axarr[ii].set_ylim([0,1.5])

    #axarr[ii].legend(rects, ('Linear-SVR', 'SVR', 'LR', 'Bayes', 'CART', 'Our method'), loc=(0.4,0.75), ncol=3)
axarr[0].legend(rects, ('Linear-SVR', 'SVR', 'LR', 'Bayes', 'CART', 'Our method'), loc=(0.4,0.75), ncol=3)


plt.show()
"""