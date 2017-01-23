#import os
#print(os.path.expanduser('~'))
#https://github.com/CanePunma/Stock_Price_Prediction_With_RNNs/blob/master/stock_prediction_keras_FINAL.ipynb
#https://github.com/anujgupta82/DeepNets/blob/master/Online_Learning/Online_Learning_DeepNets.ipynb
#http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#http://philipperemy.github.io/keras-stateful-lstm/
# LSTM for international airline passengers problem with memory
#http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
#cite: http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

from keras.callbacks import Callback
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from MainForecastModule import fnGetNaturalLogPrices
from MainForecastModule import fnComputeFeatures
from keras.models import load_model
from MainForecastModule import organize_data
from MainForecastModule import window_stack

lstrPath ="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"
#LOOKBACK window
look_back = 10
horizon =7


class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()  
        print ('resetting')      
		#if self.counter % max_len == 0:
  #          self.model.reset_states()
  #      self.counter += 1

def fnTrainModelForMap(model, x,y,pBatchSize):
	model.fit(x, y,nb_epoch =1,  batch_size=pBatchSize, shuffle=False)
	model.reset_states()
	return model

def fnTrainViaCallback(model, x,y,pBatchSize):
	model.fit(x, y,nb_epoch =40, callbacks=[ResetStatesCallback()], batch_size=pBatchSize, shuffle=False)
	return model


# convert an array of values into a dataset matrix
def create_dataset(dataset, targetSet, look_back=1,horizon=1, positionY=0):
        dataX, dataY = [], []
        #y =dataset[-1+horizon+look_back:len(dataset)-1]
        #y =np.array([i[positionY] for i in targetSet])
        #y =np.reshape(y,len(y),0)
        for i in range(1+len(dataset)-look_back-horizon):
                #a = dataset[i:(i+look_back), 0] original code
                a = dataset[i:(i+look_back), :] # modified for multiple features
                dataX.append(a)
                #dataY.append(targetSet[i + look_back+horizon-1, positionY])
                dataY.append(targetSet[i + look_back+horizon-1])


		#X[:-horizon]
		#dataX =dataX[:-horizon]
        return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility
np.random.seed(7)
# load the dataset
#dataframe = pd.read_csv(lstrPath+'international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataframe = pd.read_csv(lstrPath+'SPY with returns.csv',  engine='python', skipfooter=3)
dataframe = pd.read_csv(lstrPath+'SPYRNN.csv',  engine='python')
dataframe.Date = pd.to_datetime(dataframe.Date)
dataframe.index =dataframe['Date']

ndaysNatLog=20

dataframe=fnGetNaturalLogPrices(dataframe,ndaysNatLog)
dataframe =dataframe[ pd.notnull(dataframe['Adj Close'])] #pDf =pDf[ pd.notnull(pDf['CloseSlope'])]
dataframe,lstColsdrop=fnComputeFeatures(dataframe,look_back,7,horizon,1,1)

#lstCols =['Adj Close','rollingStdev20','rollingMax20','rollingMin20','OBV','upper_band','lower_band']
lstCols =['Adj Close', 'High','Low','OBV','HighLowRange','rollingMean50','rollingMax20','rollingMin20','upper_band','lower_band']
print ('running with slope lookback =7, look_back =' + str(look_back)+ ', natlog =' +str(ndaysNatLog) +' days ahead, 2 total LSTMS')

#MACD
#lstCols =['Adj Close','CloseSlope','rollingMax20','rollingMin20','HighLowRange','rollingStdev20','rollingMean20'] -neg. r^2
#lstCols =['Adj Close','rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol','HighLowRange','rollingMean20','Volume']# .01 rsq
#lstCols =['Adj Close'] #.19 #,'rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol']
numFeatures =len(lstCols)	
print (lstCols)
dataframe=dataframe[lstCols] #,'Volume','rollingMax20','rollingMin20']]

targetSet =dataframe['Adj Close'].values
targetSet =targetSet.astype('float32')

dataset = dataframe.values
dataset = dataset.astype('float32')

# normalize the dataset
if True:
	#print ('temporarily not scaling so we can debug')
	#DO NOT scale the target, only features
	scaler = MinMaxScaler(feature_range=(0, 1))
	Targetscaler = MinMaxScaler(feature_range=(0, 1))
	
	dataset = scaler.fit_transform(dataset)
	
	#targetSet =Targetscaler.fit_transform(targetSet)

else:
	print ('temporarily not scaling so we can debug')


# split into train and test sets
train_size = int(len(dataset) * 0.67)

test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

trainTarget, testTarget =targetSet[0:train_size], targetSet[train_size:len(targetSet)]
# reshape into X=t and Y=t+1

#different method for obtaining train, test  data
#lEndIndex =(len(df)-horizon)
#discard X, use y
#X,y = organize_data(train, look_back, horizon)

#use this X
#X = window_stack(dataframe[0:lEndIndex].values, stepsize=1, width=look_back   ) 

trainX, trainY = create_dataset(train,trainTarget ,look_back,horizon)	
testX, testY = create_dataset(test,testTarget, look_back,horizon)
# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], numFeatures))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], numFeatures))
# create and fit the LSTM network
batch_size = 1

blnLoadModel =False
if blnLoadModel:
	model = load_model(lstrPath+'KerasStockModel.h5')
else:

	model = Sequential()
	#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, 1), stateful=True))
	#batch_input_shape=(batch_size, time_steps, features)
	
	#model.add(LSTM(4, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,return_sequences=True, consume_less='cpu'))
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,return_sequences=True,consume_less='cpu'))
	model.add(LSTM(4, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,return_sequences=True,consume_less='cpu'))
	model.add(LSTM(252, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,return_sequences=True,consume_less='cpu'))
	model.add(LSTM(252, batch_input_shape=(batch_size, look_back, numFeatures), stateful=True,consume_less='cpu'))

	model.add(Dense(1)) #,activation ='relu' -> gives WORSE results.
	# Compile model
	learn_rate=.001
	#momentum=0
	#optimizer = SGD(lr=learn_rate, momentum=momentum)
	optimizer = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-08, decay=0.0)
	#adam R sq of -2
	model.compile(loss='mean_squared_error',optimizer=optimizer) # optimizer='adam')

	#model=fnTrainViaCallback(model,trainX,trainY,batch_size)
	
	#tmp=list(map(fnTrainModelForMap(model,trainX,trainY,batch_size),range(10)))
	#test=[fnTrainModelForMap(model,trainX,trainY,batch_size) for x in range(20)]
	#model=model[len(tmp)-1]

	if True:
		for i in range(15): #10 ITERATIONs is best thus far
			history=model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
			#history.history['loss']
			model.reset_states()
	
	# make predictions
	trainPredict = model.predict(trainX, batch_size=batch_size)

	model.save(lstrPath+'KerasStockModelNew.h5') 

model.reset_states()


testPredict = model.predict(testX, batch_size=batch_size)


# invert predictions
if False:
	trainPredict = scaler.inverse_transform(trainPredict)
	trainY = scaler.inverse_transform([trainY])
	testPredict = scaler.inverse_transform(testPredict)
	testY = scaler.inverse_transform([testY])
else:
	pass
	#trainPredict = Targetscaler.inverse_transform(trainPredict)
	#trainY =  Targetscaler.inverse_transform(trainY)
	#testPredict =  Targetscaler.inverse_transform(testPredict)
	#testY =  Targetscaler.inverse_transform(testY)
# calculate root mean squared error
#trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict))

print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict))
#testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

print ('R2 score on Train Returns:' +str(r2_score(trainY,trainPredict)))

print ('R2 score on Returns:' +str(r2_score(testY,testPredict)))

#fig=plt.figure(figsize=(8,6))
#fig.plot(trainY,label='Actual Train ')
#fig.plot(trainPredict,label='Predicted Train ')
#fig.show()

#fig=plt.figure(figsize=(8,6))
plt.plot(testY,label='Actual Test ')
plt.plot(testPredict,label='*Predicted* Test ')
plt.show()

if False:
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
