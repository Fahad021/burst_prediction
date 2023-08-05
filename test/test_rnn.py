import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas
import math
import sys
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
# get dates


def get_dates(filename):
    return np.loadtxt(
        filename, unpack=True, converters={0: mdates.strpdate2num('%Y%m%d')}
    )


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    """
    dataX: a list of sequences [t-look_back,t]
    dataY: a list of next time value t+1
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
dataframe = pandas.read_csv(sys.argv[1], header=None)
dataframe = pandas.DataFrame({"numbers": np.array(dataframe.iloc[1])})
# print dataframe
# numbers = np.array(df.iloc[0])
# print numbers.shape
# days = get_dates(sys.argv[2])
# print days.shape
# dataframe = pandas.DataFrame({"day":days, "number":numbers})
# print dataframe.shape
dataset = dataframe.values
dataset = dataset.astype('float32')
# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
# split into train and test sets
train_size = int(len(dataset) * 0.3)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
# reshape into X=t and Y=t+1
look_back = 15
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
# another way:
# reshape input to be [samples, time steps, features]
# trainX = numpy.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
# testX = numpy.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_dim=look_back))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
# print trainPredict.shape, trainY.shape
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f RMSE' % (testScore))

np.savez(sys.argv[2], dataset=scaler.inverse_transform(dataset),
         rnn_predict=testPredict, rnn_train=trainPredict)
