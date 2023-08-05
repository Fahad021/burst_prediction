#!/usr/bin/env python
# coding=utf-8

# build lstm network

import numpy as np
import pandas
import time
import csv

from numpy import newaxis
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Activation
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# get dates
def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )

# convert an array of values into a dataset matrix
def create_dataset(dataset, seq_len=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - seq_len - 1):
        a = dataset[i:(i + seq_len), 0]
        dataX.append(a)
        dataY.append(dataset[i + seq_len, 0])
    return np.array(dataX), np.array(dataY)

def create_dataset_v2(dataset, seq_len=1):
    windows = []
    windows_y = []
    for sequence in dataset:
        len_seq = len(sequence)
        for window_start in range(0, len_seq - seq_len -1):
            window_end = window_start + seq_len
            window = sequence[window_start:window_end]
            windows.append(window)
            windows_y.append(sequence[window_end+1])
    return np.array(windows), np.array(windows_y)

# load the dataset
def load_data(filename, startpointfile, seq_len=1, id_=0):
    dataframe = pandas.read_csv(filename, header=None)
    dataframe = pandas.DataFrame({"numbers": np.array(dataframe.iloc[id_])})
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    
    stpoint = pandas.read_csv(startpointfile, header=None).values[id_][0]
    dataset = dataset[stpoint:]

    # normalize the dataset
    # scaler = MinMaxScaler(feature_range=(0, 1))
    # if scaler:
    #     print "scaler"
    #     dataset = scaler.fit_transform(dataset)
    
    # reshape into X=t and Y=t+1
    X, Y = create_dataset(dataset, seq_len)
 
    # reshape input to be [samples, time steps, features]
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    return dataset, X, Y#, scaler


def get_samples_for_lstm(file1, file2, file3, pid, seq_len):
    print "get samples for lstm"
    series = np.genfromtxt(file1, delimiter=",") # series
    r3 = csv.reader(open(file2, "r")) # category_info
    stpoints = pandas.read_csv(file3, header=None)

    # get all products in the category
    pid_lines = None
    for line in r3:
        if str(pid) in line:
            pid_lines = [eval(i) for i in line]
            break

    if pid_lines is None:
        return None

    pid_lines = [pid_lines[i] for i in range(len(pid_lines)) if i % 10 == 0]
    dataset = [series[i][stpoints.values[i][0]:] for i in pid_lines]

    X, Y = create_dataset_v2(dataset, seq_len)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    print "lstm sample shape: ", X.shape    
    return X, Y


# build LSTM network
def build_model(seq_len=1):
    model = Sequential()

    model.add(LSTM(10, input_dim=seq_len))
    model.add(Dense(1))

    start = int(time.time())
    model.compile(loss='mean_squared_error', optimizer='adam')
    print "LSTM Compilation Time : ", time.time() - start
    return model


# inverse tranform data
def inverse_data(scaler, data_list):
    return [scaler.inverse_transform(data) for data in data_list]

