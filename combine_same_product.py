#!/usr/bin/env python
# coding=utf-8

# combine all product with same info
# used before get product tragectory(for threshold)

import csv
import numpy as np
import pandas as pd
import json
import sys


def get_product_info(filename):
	# get info from 'product_sorted.txt'
	reader = csv.reader(open(filename, "r"), delimiter="\t")
	product = {}
	for row in reader:
	    product[row[0]] = [row[1], row[2], row[3], row[4]]

	return product

# load pid from 'product_sorted.txt'
pid_array = [l[0] for l in csv.reader(open(sys.argv[1],"r"), delimiter="\t")]
m = len(pid_array)

p_info = get_product_info(sys.argv[1])

same_p_writer = csv.writer(open(sys.argv[2], "w"), delimiter=",")

# first round: get same product following -> cut list length
print "first round"
tmp_list = []
i = 0
while i < m:
	pid = pid_array[i]
	same_p = [pid]

	while True:
		if i + 1 >= m - 1:
			break
		next_pid = pid_array[i+1]
		if p_info[pid] != p_info[next_pid]:
			break
		same_p.append(next_pid)
		i += 1

	if len(same_p) > 1:
		tmp_list.append(same_p)
		# same_p_writer.writerow(same_p)

	i += 1

# second round: get all same product in the list
print "second round", len(tmp_list)
combined_index = []
same_p = {}
for i in range(len(tmp_list)):
	if i in combined_index:
		continue

	for j in xrange(i+1, len(tmp_list)):
		if j in combined_index:
			continue

		pid = tmp_list[i][0]
		next_pid = tmp_list[j][0]
		# print pid, next_pid
		if p_info[pid] == p_info[next_pid]:
			tmp_list[i] += tmp_list[j]
			combined_index.append(j)
			# print j

for i in range(len(tmp_list)):
	if i not in combined_index:
		same_p_writer.writerow(tmp_list[i])
		same_p[tmp_list[i][0]] = tmp_list[i][1:]


# combine product records
print "combine records"
rd  = csv.reader(open(sys.argv[3], "r"), delimiter=" ")
writer = csv.writer(open(sys.argv[4], "w"), delimiter=" ")
records = {}
for line in rd:
	records[line[0]] = line[1:]

delete_pid = []
for pid in records:
	if pid not in same_p:
		continue
	for spid in same_p[pid]:
		delete_pid.append(spid)
		records[pid] += records[spid]

for pid in records:
	if pid in delete_pid:
		continue
	writer.writerow([pid] + records[pid])
