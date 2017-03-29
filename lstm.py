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
    days = np.loadtxt(filename,
                      unpack=True,
                      converters={0: mdates.strpdate2num('%Y%m%d')})
    # print days[:5]

    return days

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
    for i, sequence in enumerate(dataset):
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


def get_samples_for_lstm(dataset, seq_len):
    print "use burst series as the input of lstm"

    dataset = [dataset[i] for i in range(len(dataset)) if i % 10 == 0] # size issue!
    X, Y = create_dataset_v2(dataset, seq_len)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

    print "lstm sample shape: ", X.shape
    return X, Y


# build LSTM network
def build_model(seq_len=1):
    model = Sequential()

    model.add(LSTM(50, input_dim=seq_len))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    start = int(time.time())
    model.compile(loss='mean_squared_error', optimizer='adam')
    print "LSTM Compilation Time : ", time.time() - start
    return model


# inverse tranform data
def inverse_data(scaler, data_list):
    new_list = []
    for data in data_list:
        new_list.append(scaler.inverse_transform(data))

    return new_list

