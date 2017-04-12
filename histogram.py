#!/usr/bin/env python
import csv
import seaborn as sns
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

r = csv.reader(open("p_data/dist.csv","r"))
x = []
for line in r:
    l = []
    for i in line:
        l.append(float(i))
    x.append(l)
# the histogram of the data
#n, bins, patches = plt.hist(x[0],bins=10, normed=1, facecolor='green')
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
sns.distplot(x[0],kde=False,norm_hist=True,ax=ax1)
bins = np.linspace(0, 100, 101)
sns.distplot(x[1],bins,kde=False,norm_hist=True,ax=ax2)

fontsize_title = 18
fontsize = 18

ax1.set_xlabel('Burst period',fontsize=fontsize)
ax1.set_ylabel('Proportion',fontsize=fontsize)
ax2.set_xlabel('Burst peak',fontsize=fontsize)
ax2.set_ylabel('Proportion',fontsize=fontsize)
plt.ticklabel_format(useOffset=False)
plt.grid(True)

plt.show()
