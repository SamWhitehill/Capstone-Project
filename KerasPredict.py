#import os
#print(os.path.expanduser('~'))
#https://github.com/anujgupta82/DeepNets/blob/master/Online_Learning/Online_Learning_DeepNets.ipynb
#http://philipperemy.github.io/keras-stateful-lstm/
# LSTM for international airline passengers problem with memory
#http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
#cite: http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from MainForecastModule import fnGetNaturalLogPrices
from MainForecastModule import fnComputeFeatures

lstrPath ="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"
#LOOKBACK window
look_back = 15
horizon =7
# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1,horizon=1, positionY=0):
        dataX, dataY = [], []
        #y =dataset[-1+horizon+look_back:len(dataset)-1]
        y =np.array([i[positionY] for i in dataset])
        #y =np.reshape(y,len(y),0)
        for i in range(len(dataset)-look_back-horizon):
                a = dataset[i:(i+look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])


		#X[:-horizon]
        return np.array(dataX), y#np.array(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
#dataframe = pd.read_csv(lstrPath+'international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataframe = pd.read_csv(lstrPath+'SPY with returns.csv',  engine='python', skipfooter=3)
dataframe = pd.read_csv(lstrPath+'SPY TRAIN ARIMA.csv',  engine='python')
dataframe.Date = pd.to_datetime(dataframe.Date)
dataframe.index =dataframe['Date']

dataframe=fnGetNaturalLogPrices(dataframe,25)
dataframe =dataframe[ pd.notnull(dataframe['Adj Close'])] #pDf =pDf[ pd.notnull(pDf['CloseSlope'])]
dataframe,lstCols=fnComputeFeatures(dataframe,look_back,look_back,horizon,1,1)



numFeatures =1
dataframe=dataframe[['Adj Close']] #,'CloseSlope','Volume','rollingMax20','rollingMin20']]

dataset = dataframe.values
dataset = dataset.astype('float32')


# normalize the dataset

if True:
	#print ('temporarily not scaling so we can debug')
	scaler = MinMaxScaler(feature_range=(0, 1))
	dataset = scaler.fit_transform(dataset)
else:
	print ('temporarily not scaling so we can debug')

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# reshape into X=t and Y=t+1


trainX, trainY = create_dataset(train, look_back,horizon)	
testX, testY = create_dataset(test, look_back,horizon)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], numFeatures))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], numFeatures))
# create and fit the LSTM network
batch_size = 1
model = Sequential()
#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
#batch_input_shape=(batch_size, time_steps, features)
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,return_sequences=True, unroll=True,consume_less='cpu'))
model.add(LSTM(4, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True))

model.add(Dense(1,activation ='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')
for i in range(15):
	model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
	model.reset_states()
# make predictions
trainPredict = model.predict(trainX, batch_size=batch_size)

model.save('KerasStockModel.h5') 
model.reset_states()


testPredict = model.predict(testX, batch_size=batch_size)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print ('R2 score on Returns:' +str(r2_score(testY[0] ,testPredict[:,0])))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
#testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

testPredictPlot[len(trainPredict)+(look_back*2)+(1+horizon-3):len(dataset)-(1+horizon+1), :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
