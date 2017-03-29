#!/usr/bin/env python
# coding=utf-8

# get product number tragectory by time

import csv
import sys


def get_time_list(filename):
    rd = csv.reader(open(filename, "r"), delimiter=",")
    tl = []
    for row in rd:
        tl = row

    return tl


def get_product_trajectory(timelist, file1, file2):
    rd  = csv.reader(open(file1, "r"), delimiter=" ")
    writer = csv.writer(open(file2, "w"), delimiter=" ")

    i = 0
    for row in rd:
        pid = row[0]
        if len(row) <= 100:
            continue

        print pid
        i += 1

        l = []
        for t in timelist:
            l.append(row.count(t))

        writer.writerow([pid] + l)

    print i


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "file format: timelist record outputfile"
        exit(0)

    timelist = get_time_list(sys.argv[1])
    get_product_trajectory(timelist, sys.argv[2], sys.argv[3])
