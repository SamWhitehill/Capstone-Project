#References used to help build this module:
#http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
#https://\.com/CanePunma/Stock_Price_Prediction_With_RNNs/blob/master/stock_prediction_keras_FINAL.ipynb
#https://github.com/anujgupta82/DeepNets/blob/master/Online_Learning/Online_Learning_DeepNets.ipynb
#http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#http://philipperemy.github.io/keras-stateful-lstm/
# LSTM for international airline passengers problem with memory
#http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
# http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#http://machinelearningmastery.com/use-keras-deep-learning-models-scikit-learn-python/
#https://github.com/FreddieWitherden/ta/blob/master/ta.py
#http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/

''' IMPORTS '''
import os
from sklearn.ensemble import ExtraTreesRegressor
import datetime
from datetime import timedelta	
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

from keras.wrappers.scikit_learn import KerasRegressor

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import math


from keras.callbacks import Callback
from keras.models import Sequential
from keras.layers import Dense, Dropout,Activation
from keras.layers import LSTM
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adagrad
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from FeatureGeneration import fnGetNaturalLogPrices
from FeatureGeneration import fnComputeFeatures
from keras.models import load_model
from FeatureGeneration import organize_data
from FeatureGeneration import window_stack
from FeatureGeneration import fnGetYahooStockData

from statsmodels.tsa.stattools import adfuller

from IPython.display import display
from scipy.optimize import differential_evolution
#from modGetStockData import fnGetStockData 
#from PowerForecast import run_network
'''END IMPORTS '''

'''This snippet below is a workaround fix to Keras when grid searching'''
'''Without this custom fn, there will be a grid search error! '''
from keras.wrappers.scikit_learn import BaseWrapper
import copy

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params
'''End workaround fix to Keras grid search '''

#local path to stock history files
lstrPath ="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"

#global variables used throughout, which are set in the main function.
global lstrStock
global numFeatures
global look_back
global horizon
global batch_size
global learn_rate

#set the random number gen seed for reproducability
np.random.seed(7)


def reportStationarity(pDfTimeSeries):
    #Perform Dickey-Fuller test:
    #report on stationarity of the time series: pDfTimeSeries - a data frame containing time series
    print( 'Results of Dickey-Fuller Test:')
    dftest = adfuller(pDfTimeSeries, autolag='BIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    display(dfoutput)
    print ("")


def fnGetStationaryTimeSeries(pDf,pLstCols=None,pHorizon=1):
    #return the data frame with the LOG differences of log of open, hi, low , close prices
	#first the log of the prics is taken, then 1st differences are taken on these logged values
	# to generate a stationary time series from the non stationary stock prices
    #this should make it stationary,  for time series analysis
    lstFlds =['Adj Close','Close','Open','High','Low']
    if pLstCols !=None:
        lstFlds=pLstCols
    for fld in lstFlds:
        #pDf[fld] =np.log( pDf[fld] )
        #pDf[fld] = np.log(pDf[fld]/pDf[fld].shift(periods=Nperiods))
        pDf[fld] = np.log(pDf[fld])
        #pDf[fld] = np.sqrt(pDf[fld])
        #pDf[fld]  ,_ =boxcox(pDf[fld])
        #print ("USING sqrt INSTEAD OF LOG FOR STATIONARITY")
        #pDf['Natural Log'] = pDf['Close'].apply(lambda x: np.log(x))  
        pDf[fld] =pDf[fld] -pDf[fld].shift(periods=1) #pHorizon)
        pDf[fld] =pDf[fld] -pDf[fld].shift(periods=1)
        pDf[fld] =pDf[fld] -pDf[fld].shift(periods=1)

    return pDf
	
def fnComputeFeatureImportances(trainX,trainY,lstCols):
    #PRINT FEATURE IMPORTANCES
    #params -trainX - feature data
    #trainY -target data
    #lstCols -list of feature names
    ETmodel = ExtraTreesRegressor()
    trainX =np.reshape(trainX,(trainX.shape[0],trainX.shape[2]))
    ETmodel.fit(trainX, trainY)
    fi =ETmodel.feature_importances_
    print (fi)

    for item in lstCols:
            print(item)

def fnPrintPredVsActuals(pDfBenchmark, pPrediction,pActual, pStrHeader):
    #params:
    #pDfBenchmark - dataframe containing benchmark data
    #pPrediction - np array of predicted data
    #pActual - np array of actual data
    
    pDfBenchmark['Predicted']=pPrediction.tolist()
    pDfBenchmark['Actual']=pActual.tolist()

    print(pStrHeader)

    display(pDfBenchmark)


def fnGetStockData(pStrFileName,nDaysReturnLookBack, look_back, horizon,
                   pLstCols,pblnUseWeb=False,pRemoveAdjFromFeatures=True, pDataFrame=None):
    #retrieve stock data from web or  file
    #process it by calculating log returns over specified lookback
    #calculate features needed
    #normalizes data via feature scaling
    #finally pivot data into numpy arrays so it can be fed into RNN model.
    #returns:
    # trainX: features 
    # trainY: target data from original stock data 
    #benchmark: benchmark stock prediction for use in comparison, plots, evaluation,etc...
    #originalDF: original data frame of stock data for using plotting 
    
    #read in data
    if  pblnUseWeb==False:
		#read from a file, to save time of if no internet
            dataframe = pd.read_csv(pStrFileName,  engine='python')
    else:
            #assume data was read from web via yahoo finance
            #and is passed in via parm :pDataFrame
            dataframe	=pDataFrame
	
    dataframe.Date = pd.to_datetime(dataframe.Date) 
    dataframe.index =dataframe['Date']

    originalDF =dataframe.copy(deep=True)
    #ndaysNatLog=nDaysReturnLookBack #40

	#convert this dataframe of historical stock data into  stationary time series
	# by taking 1st differnces of log of prices
    dataframe =fnGetStationaryTimeSeries(pDf=dataframe, pHorizon=horizon)   
	
	#remove the first row as it will be null due to 1st differences
    dataframe =dataframe[ pd.notnull(dataframe['Adj Close'])]
    
	#calculation Dickey Fuller stats on the time series display the stationarity stats
    #reportStationarity(dataframe['Adj Close'])

    #set slope lookback
    
    slopeLB =look_back
    #print ('slope lookback ' +str(slopeLB))
	#retrieve the feature computed columns form this historical time series: dataframe
	#dataframe will consequently contain original data plus numerous computed features
	#lstColsSR -is not used any longer
    dataframe,lstColsSR=fnComputeFeatures(dataframe,look_back, slopeLB,horizon,22,4)

	#repor the lookback on the slope
    #print ('slope lookback at ' + str(slopeLB))
	

	#Remove FIRST n ROWS WHICH contain NaN
	#this is due the lookback window for rolling means, stdevs
    dataframe =dataframe[ pd.notnull(dataframe['MACD'])] #
    dataframe =dataframe[ pd.notnull(dataframe['SineFreq'])]
    dataframe =dataframe[ pd.notnull(dataframe['DiffercenceBtwnAvgVol'])]

	#make a copy of the list of feautre columns
    lstCols=pLstCols[:]
    #lstCols =['2DayNetPriceChange','CloseSlope','rollingMean20','Adj Close','MACD','Volume',#'upper_band','lower_band',
     #        'High','Low','rollingStdev20','rollingMax20','rollingMin20']#,'ZigZag']#,'fastk','fulld','fullk'] #,'upAroon','downAroon']#,'fastk','fulld','fullk'] # ['S1','S2','S3','R1','R2','R3']
    #lstCols =['Adj Close','CloseSlope', 'rollingMean20','rollingStdev20','rollingMax20','rollingMin20']
    #lstCols =['Adj Close'], .45 R sq.
    #print ('Running with slope lookback =4, look_back =' + str(look_back)+ ', natlog =' +str(ndaysNatLog) +' days ahead, 2 total LSTMS')

    #MACD adds value to R^2, ForceIndex does NOT! Volumeslope does not
    #lstCols =['Adj Close','CloseSlope','rollingMax20','rollingMin20','HighLowRange','rollingStdev20','rollingMean20'] -neg. r^2
    #lstCols =['Adj Close','rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol','HighLowRange','rollingMean20','Volume']# .01 rsq
    #lstCols =['Adj Close'] #.19 #,'rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol']
    numFeatures =len(lstCols)	
    print ('Using these features ',lstCols)
    print ('')
 
    #create the target dataset from the original stock data
    targetSet =dataframe['Adj Close'].values
    targetSet =targetSet.astype('float64')

    #create a naive benchmark, which is simply last known price before the prediction
    benchmark=dataframe[['Adj Close']][look_back-1: len(dataframe)-horizon]
    benchmark['Date'] =benchmark.index
    
    if pRemoveAdjFromFeatures:
        lstCols.remove('Adj Close')
        numFeatures=numFeatures-1
    dataframe=dataframe[lstCols] #,'Volume','rollingMax20','rollingMin20']]

    #create the features dataset from stock dataframe
    dataset = dataframe.values
    dataset = dataset.astype('float64')

    # normalize the dataset
    if True:
            
            #create a feature scaler as the data MUST be scaled.
            # for use in the RNN model
            scaler = MinMaxScaler(feature_range=(-1, 1))
            #Targetscaler = MinMaxScaler(feature_range=(0, 1))
            
            dataset = scaler.fit_transform(dataset)
            

    else:
            print ('Temporarily not scaling so we can debug')

    # split into train and test sets
    train_size = int(len(dataset)) # * 0.65)

    #test_size = len(dataset) - train_size
    #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = dataset[0:train_size,:]

    #trainTarget, testTarget =targetSet[0:train_size], targetSet[train_size:len(targetSet)]
    trainTarget =targetSet[0:train_size]
    # reshape into X=t and Y=t+1

    trainX, trainY = get_dataset(train,trainTarget ,look_back,horizon)

    #fnComputeFeatureImportances(trainX,trainY,lstCols)

    #testX, testY = get_dataset(test,testTarget, look_back,horizon)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], numFeatures))
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], numFeatures))

    return trainX, trainY,benchmark,originalDF

class DecayLR(Callback):
    #class is used to decay the learning rate in a customized way
    #def __init__(self):
    
    def on_epoch_end(self, epoch, logs={}):
        #lr=self.model.optimizer.lr #.get_value()
        lr =model.nLearnRate
        print ('lr is '+str(lr))
        lr=step_decay(epoch, lr)
        #self.model.optimizer.lr.set_value(lr)
        self.model.optimizer.lr =lr
        #rint ('updated learning rate to: '+ str(lr))
        
class ResetStatesCallback(Callback):
	#class used to reset the state of keras RNN when training stateful network
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()  
        print ('resetting')      
		#if self.counter % max_len == 0:
  #          self.model.reset_states()
  #      self.counter += 1

def fnPlotChart(pBenchmark=None, pPredictions=None,pActual=None, \
                pStrStock="", pStrTitle="", pBenchLabel="",pBenchMarkField='Adj Close', pBlnPlotBenchMark=True):
    #display chart of benchmark , predicted and actual stock data (returns, prices,etc..)
    #parms: pBenchmark - dataframe of benchmark Adj Closing prices or returns and dates
    #pPredictions - np array containing predicted stock returns or prices
    #pActual - np array containing actual stock returns or prices
    #pStrStock -stock ticker symbol
    #pStrTitle -title of plot chart
    #pBenchLabel -benchmark label on chart

    #np.savetxt("C:\\temp\\testY.csv", testY, delimiter=',')
    #np.savetxt("C:\\temp\\testX.csv", testX, delimiter=',')
    lstrStock =pStrStock

    # plot the benchmark
    if pActual is not None:
            plt.plot_date(pBenchmark['Date'], pActual, 'k-',label='Actual '+lstrStock,linewidth=2)

    benchLabel ='Benchmark '+lstrStock
    if pBenchLabel!="":
        benchLabel=pBenchLabel
    
    if pBlnPlotBenchMark:
        plt.plot_date(pBenchmark['Date'], pBenchmark[pBenchMarkField], 'b-',label=benchLabel,linewidth=2)
	
    if pPredictions is not None:
        plt.plot_date( pBenchmark['Date'], pPredictions, 'g-',label='*Predicted* '+lstrStock,linewidth=2,linestyle="--")

	#show dates on x axis
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    plt.gcf().autofmt_xdate()

    #plt.plot(testY,label='Actual '+lstrStock)
    #plt.plot(testPredict,label='*Predicted* '+lstrStock)

	#show title
    plt.suptitle(pStrTitle,
                    fontsize=13, fontweight='bold')

    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.show()



def step_decay(epoch):
    #function to decay the learning rate in a step wise or custom fashion
    #initial_lrate =lr #1.08E-02 # 0.0012 #The Alpha used was .95
    global learn_rate
    initial_lrate =learn_rate #0.0003 #9e-4
    #print ('initial learning rate is ' +str(learn_rate))
    #increasing the learning rate (initial appears to flatten out the prediction
    drop = 0.5
    epochs_drop = 10.0
    ParmA=-0.000065
    ParmB=0.006

    ParmC=600

    #print ('initial learn rate ' +str(initial_lrate))
    
    lrate=initial_lrate * math.exp(-ParmA * epoch)/( 1 + math.exp(ParmB * (epoch-ParmC)))
    #lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    #print ('using decay initial_lrate=.' +str(initial_lrate) +' * math.pow(drop, math.floor((1+epoch)/epochs_drop))')
    #print ('using tanh decay')
    #print ('learning rate decay fn '+str(lrate))
    return lrate

#def fnGetModelMultiLayer(blnStateful,batch_size, look_back, numFeatures):
def fnGetModelMultiLayer(nLayers=0,numNeurons=0, nDropout=0,nEpochs=3000,nLearnRate=.00001,numFeatures=15): # neuronsL1=0,neuronsL2=0,neuronsL3=0):
    '''build a multilayer LSTM RNN model according to parms specified
    parms:nLayers -number of layers in RNN model
    numNeurons -number neurons per layer
    nDropout -dropout percentage per layer
    nEpochs -deprecated and no longer used
    nLearnRate -deprecated, will be overridden by step_Decay function
    numFeatures -deprecated, now numNeurons is used instead
    
    '''
    model = Sequential()
    blnStateful=True
    
    print ('Statefulness is ' +str(blnStateful))
    #layers={'hidden2': numFeatures*12, 'input': 1, 'hidden1': numFeatures*8, 'hidden3': numFeatures*4, 'output': 1}
    #layers={'hidden2': numFeatures*18, 'input': 1, 'hidden1': numFeatures*4, 'hidden3': numFeatures*4, 'output': 1}
    #layers={'hidden2': numFeatures*2, 'input': 1, 'hidden1': numFeatures*6, 'hidden3': numFeatures*6, 'output': 1}
    #layers={'hidden2': neuronsL2, 'input': 1, 'hidden1': neuronsL1, 'hidden3': neuronsL3, 'output': 1}
    
    layers = {'input': 1, 'hidden1': numFeatures, 'hidden2': 128, 'hidden3': 100, 'output': horizon}
    #increASING hidden1 flattens!
    #decreasing hidden3 amplifies volatility!
    print ('network structure is:')
    print(nLayers, numNeurons, nDropout)
    #keras.layers.recurrent.LSTM(output_dim, init='glorot_uniform',
    #inner_init='orthogonal', forget_bias_init='one', activation='tanh',
    #inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
    ldropout =nDropout #.20
    print ('dropout at: ' + str(ldropout))
    print ('learning rate at: ' +str(nLearnRate))
    #high ldropout flattens it out
    #Available activations

    #softmax: Softmax applied across inputs last dimension. Expects shape either (nb_samples, nb_timesteps, nb_dims) or  (nb_samples, nb_dims).
    #softplus
    #softsign
    #relu
    #tanh
    #sigmoid
    #hard_sigmoid
    #linear
    
    #***********
    #using activation='tanh', inner_activation='relu' gives a flat line, when using features computed on stationary data
    #cannot use relu in activation, returns nan
    #sigmoid in activation produces flat line

    #cannot use inner_activation =relu with stateful=True, loss =Nan

    #model.add(Dropout(ldropout,batch_input_shape=(batch_size, look_back, numFeatures)))

    for i in range(nLayers):
        if i==(nLayers-1):
            model.add(LSTM(numNeurons ,activation='tanh',inner_activation='hard_sigmoid', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful)) #activation='tanh',inner_activation='tanh'
        else:
            model.add(LSTM(numNeurons ,activation='tanh',inner_activation='hard_sigmoid', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
           
        model.add(Dropout(ldropout,batch_input_shape=(batch_size, look_back, numFeatures)))

    #model.add(Dropout(ldropout))

    model.add(Dense( input_dim =numNeurons,
            output_dim=layers['output']))

    #model.add(Dense(layers['hidden3'], activation='relu'))
    #model.add(Dense(output_dim=layers['output'], activation='linear'))
    
    model.add(Activation("linear"))
    learn_rate=nLearnRate #0.0005242175
    #momentum=0
    #optimizer = SGD(lr=learn_rate, momentum=momentum)
    decay_rate =learn_rate/nEpochs
    optimizer = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-10,decay=decay_rate)
    #optimizer = Adagrad(lr=learn_rate, epsilon=1e-08, decay=0.00)
    #Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adam R sq of -2
    model.compile(loss='mean_squared_error',optimizer=optimizer) # optimizer='adam')

    #model.summary()
    #plot(model, to_file='RNNmodel.png')
    model.nLearnRate =nLearnRate
    #setattr(model, 'somefield', 'somevalue')
    return model

def fnRandomizedSearchModel(trainX, trainY):

    model = KerasRegressor(build_fn=fnGetModelMultiLayer, nb_epoch=180, batch_size=batch_size,
                            verbose=2)
        
    # specify parameters and distributions to sample from
    param_dist = {"numNeurons": sp_randint(15,80),
                  "nLayers": sp_randint(1, 6),
                  "nDropout":[.1,.2,.3,.4,.5],
                  "nLearnRate":[0.00001,0.00002, 0.00008, 0.0002,
    .0004,.0007,.001,.002,.003,.005,.008,.01,.02 ]}

    tscv = TimeSeriesSplit(n_splits=2)
    #CVData =[(train,test) for train, test in tscv.split(trainX)]
    CVData =[( fnSliceOffDataPerBatchSize(pFeatures=train,pBatch_Size =batch_size)[0],
	fnSliceOffDataPerBatchSize(pFeatures=test,pBatch_Size =batch_size)[0]  ) 
	for train,test in tscv.split(trainX)]

    #clf = GridSearchCV(clfReg, parameters, verbose=1,n_jobs=3, cv=CVData)
    param_grid = dict(nLayers=nLayers,numNeurons=numNeurons,nDropout=nDropout) #,nLearnRate=nLearnRate )

    #param_grid = dict(neuronsL1=neuronsL1,neuronsL2=neuronsL2,neuronsL3=neuronsL3 )

    #create a list of callbacks for model:
    #first is the learning rate decay
    #2nd is the resettign of the state since this is a stateful network
    
    lrate = LearningRateScheduler(step_decay)
    #callbacks_list = [lrate,ResetStatesCallback()]
    callbacks_list=[ResetStatesCallback()]
    
    #print (param_grid)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,
                        cv=CVData,fit_params={'callbacks': callbacks_list})

    print ('Random CV Searching '+lstrStock)
    grid_result = grid.fit(trainX, trainY)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    params = grid_result.best_params_
    return grid_result.best_score_, params



    
def fnGridSearchModel(trainX, trainY):
    #Reference:http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    model = KerasRegressor(build_fn=fnGetModelMultiLayer, nb_epoch=180, batch_size=batch_size,
                            verbose=2)
        
    # define the grid search parameters
    numNeurons =[i*numFeatures for i in range(1,4,1)] #range stops 1 short
    #numNeurons=[2*numFeatures]
    nLayers =[3,4,5,6]
    nDropout =[.2] #[.1,.2,.4]
    #batch_size =[1,2,3,4,look_back,look_back*2]

    tscv = TimeSeriesSplit(n_splits=2)
    #CVData =[(train,test) for train, test in tscv.split(trainX)]
    CVData =[( fnSliceOffDataPerBatchSize(pFeatures=train,pBatch_Size =batch_size)[0],
	fnSliceOffDataPerBatchSize(pFeatures=test,pBatch_Size =batch_size)[0]  ) 
	for train,test in tscv.split(trainX)]

    #clf = GridSearchCV(clfReg, parameters, verbose=1,n_jobs=3, cv=CVData)
    param_grid = dict(nLayers=nLayers,numNeurons=numNeurons,nDropout=nDropout) #,nLearnRate=nLearnRate )

    #param_grid = dict(neuronsL1=neuronsL1,neuronsL2=neuronsL2,neuronsL3=neuronsL3 )

    #create a list of callbacks for model:
    #first is the learning rate decay
    #2nd is the resettign of the state since this is a stateful network
    
    lrate = LearningRateScheduler(step_decay)
    callbacks_list = [lrate,ResetStatesCallback()]
    ##callbacks_list=[ResetStatesCallback()]
    
    #print (param_grid)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1,
                        cv=CVData,fit_params={'callbacks': callbacks_list})

    print ('Grid Searching '+lstrStock)
    grid_result = grid.fit(trainX, trainY)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    params = grid_result.best_params_
    return grid_result.best_score_, params

def fnGetModel(blnStateful=True, pLook_Back=7, nEpochs =3000, nLearnRate=.01, nDropout=.2, nLayers=1,nNeuronMultiplier=2):
    global learn_rate
    learn_rate=nLearnRate
    #model = Sequential()
    #numNeurons =100 # try 4*look_back  #increasing neuron sappears to increase volatility too much ?
    #numNeurons=2*numFeatures #4*look_back *numFeatures
    #model.add(Dropout(0.25,batch_input_shape=(batch_size, look_back, numFeatures)))
    #lDRRate =.2555  #increased dr for OIL improves
    #nLayers =3 #less layers from 6 to 2 is worse R^2

    
    #REMOVING dropout severely hurts score/fit

    #decreasing learning rate increases volatility
    model =fnGetModelMultiLayer(nLayers=nLayers,numNeurons=int(nNeuronMultiplier),\
                                nDropout=nDropout,nLearnRate= learn_rate, nEpochs=nEpochs)
                  


    return model

def fnSliceOffDataPerBatchSize(pFeatures=None,pTarget=None, pBatch_Size=1):
    #must slice off data to match batch_size per Keras requirement when training by batch
    #must slice off training data to be divisible by batch_size
    lLenData =len(pFeatures)
    lRemainder =lLenData % pBatch_Size
    lLenData =lLenData-lRemainder

    pFeatures =pFeatures[:lLenData]
    
    if pTarget !=None:
        pTarget=pTarget[:lLenData]

    return pFeatures, pTarget

def CalculateAccuracyRatio(pPredictions,pActuals):
	#calculate accuracy ratio to determine score and evaluate model
	# defined as sum of squares of log of ratio of prediction/actual
	#print (pPredictions)
	#print (np.log(pPredictions))
	accRatio =np.log(pPredictions/pActuals)
	
	accRatio = np.square(accRatio)
	accRatio =np.sum(accRatio)
	return accRatio

def fnComputePredictedPrices(pBenchMark, pOriginalDF, pPredictions):
	#compute forecasted stock prices (adj close) based off of predicted values
	#predicted values are 1 day differences in log of prices
        dateStart =pBenchMark[horizon-1:horizon]['Date'].values[0]

        #dateStart =pBenchMark[0:1]['Date'].values[0]
        dateEnd =pBenchMark[len(pBenchMark)-1:len(pBenchMark)]['Date'].values[0]
        lBenchMValues =pBenchMark[['Adj Close']].values
        pPredictions =np.reshape(pPredictions, (pPredictions.shape[0]))

        #filter on 1st predicted row
        forecastDF =pOriginalDF[pOriginalDF['Date']>=dateStart]
		# need to filter off back end too as this will be Sliced off due to batch size match up.
        #forecastDF =forecastDF[forecastDF['Date']<=dateEnd]
        forecastDF =forecastDF[:len(pBenchMark)+1] #+horizon]
        #calculate log of adj close
        forecastDF['LogAdjClose'] =np.log(forecastDF['Adj Close'])
        #TODO: forecastDF['PredictedAdjClose']=0, forecastDF['BenchmarkAdjClose']=0
        forecastDF['PredictedAdjClose']=0
        forecastDF['BenchmarkAdjClose']=0
        #add predicted change
        forecastDF['PredictedAdjClose'][:len(pBenchMark)]=forecastDF[:len(pBenchMark)]['LogAdjClose'] +pPredictions.tolist()            
        # add benchmark change
        forecastDF['BenchmarkAdjClose'][:len(pBenchMark)]=forecastDF[:len(pBenchMark)]['LogAdjClose'] +pBenchMark['Adj Close'].values

        #convert from log to original price
        forecastDF['PredictedAdjClose'] =np.exp(forecastDF['PredictedAdjClose'])
        forecastDF['BenchmarkAdjClose'] =np.exp(forecastDF['BenchmarkAdjClose'])
            
		#shift ahead 1 day since we predicted change from prior day.
        forecastDF['PredictedAdjClose']  =forecastDF['PredictedAdjClose'].shift(1)
        forecastDF['BenchmarkAdjClose']=forecastDF['BenchmarkAdjClose'].shift(1)

        #since we shifted forecasts up to horizon date, there will be nas to drop
        forecastDF.dropna(inplace=True)

        #slice off remaining prices due to batch size division
        #forecastDF=forecastDF[:len(pPredictions)]
        return forecastDF


# convert an array of values into a dataset matrix
def get_dataset(dataset, targetSet=None, look_back=1,horizon=1, positionY=0):
        dataX, dataY = [], []
        #y =dataset[-1+horizon+look_back:len(dataset)-1]
        #y =np.array([i[positionY] for i in targetSet])
        #y =np.reshape(y,len(y),0)
        for i in range(1+len(dataset)-look_back-horizon):
                #a = dataset[i:(i+look_back), 0] original code
                a = dataset[i:(i+look_back), :] # modified for multiple features
                dataX.append(a)
                #dataY.append(targetSet[i + look_back+horizon-1, positionY])
                #dataY.append(targetSet[i + look_back+horizon-1])

				#changed to allow for multiple output layer RNN - i.e., predict 5 results instead of 1
                dataY.append(targetSet[i + look_back: i + look_back+ horizon])


		#X[:-horizon]
		#dataX =dataX[:-horizon]
        return np.array(dataX), np.array(dataY)
# fix random seed for reproducibility

# load the dataset
#dataframe = pd.read_csv(lstrPath+'international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
#dataframe = pd.read_csv(lstrPath+'SPY with returns.csv',  engine='python', skipfooter=3)
#dataframe = pd.read_csv(lstrPath+'SPY_train.csv',  engine='python')

#REMOVED MACD as this reduces amount of data (needs 26 days prior)
#'RealBody','BarType',#'upper_band','lower_band',
           #'UpperShadow','LowerShadow','Color',
#lstCols =['2DayNetPriceChange','CloseSlope','Adj Close','Volume', 'rollingMean20','EMV','ForceIndex',
#	'MACD',  'High','Low','rollingStdev20','rollingMax20','rollingMin20','StdDevSlope','VolumeSlope',
#	'DiffercenceBtwnAvgVol',
#	'UpDownVolumeChange',
#	'SineFreq',
#	'SineAmp',
#	'SinePhase',
#	'SineOffset','RealBody','BarType', #,'upper_band','lower_band',
#           'UpperShadow','LowerShadow','Color'
#	]


def fnRunVXX(pBlnGridSearch =False,pLook_Back=10, pHorizon=1):
    '''Run the RNN predictions on the VXX ETF using dates below

    Parm: pBlnGridSearch -true if we perform grid searching, false if not
    pLook_Back -number of days to look back within the model
    pHorizon -days ahead to forecast
    '''
    global  learn_rate
    lStartDateTrain=datetime.date(2009, 3, 1)
    lEndDateTrain=datetime.date(2010,1,31)

    lStartDateTest=datetime.date(2010,2,1)
    lEndDateTest=datetime.date(2010,7,30)

    lstrStock="VXX"
    learn_rate=0.001
    
    fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
           pLearnRate=learn_rate, pDropout=.1, pLayers=1, pNeuronMultiplier=2,\
           pLook_Back=pLook_Back, pHorizon=pHorizon,pBatchSize =8, pEpochs=500)


def fnRunOIL(pBlnGridSearch =False,pLook_Back=10, pHorizon=1):
    '''Run the RNN predictions on the OIL ETF using dates below
    Parm: pBlnGridSearch -true if we perform grid searching, false if not
    pLook_Back -number of days to look back within the model
    pHorizon -days ahead to forecast
    '''
    global  learn_rate
    lStartDateTrain=datetime.date(2007, 2, 1)
    lEndDateTrain=datetime.date(2008, 3, 31)

    lStartDateTest=datetime.date(2008, 4, 1)
    lEndDateTest=datetime.date(2008, 8, 29)

    lstrStock="OIL"

    learn_rate=0.004
    
    #5.5 % r Sq
    fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
           pLearnRate=learn_rate, pDropout=.1, pLayers=1, pNeuronMultiplier=1,pLook_Back=pLook_Back, pHorizon=pHorizon,\
           pBatchSize =8, pEpochs=500)

    #fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
     #   pLearnRate=learn_rate, pDropout=.2, pLayers=3, pNeuronMultiplier=4,pLook_Back=pLook_Back, pHorizon=pHorizon)


def fnRunQQQ(pBlnGridSearch =False,pLook_Back=10, pHorizon=1):
    '''Run the RNN predictions on the QQQ ETF using dates below
    Parm: pBlnGridSearch -true if we perform grid searching, false if not
    pLook_Back -number of days to look back within the model
    pHorizon -days ahead to forecast
    '''
    #global lstrStock
    global learn_rate
    lStartDateTrain=datetime.date(1999, 3, 10)
    lEndDateTrain=datetime.date(2000, 1  , 31)

    lStartDateTest=datetime.date(2000, 2, 1)
    lEndDateTest=datetime.date(2000, 7  , 31)

    lstrStock="QQQ"

    learn_rate=0.0008
    
    
    fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
           pLearnRate=learn_rate, pDropout=.1, pLayers=1, pNeuronMultiplier=2,\
           pLook_Back=pLook_Back, pHorizon=pHorizon,pBatchSize =8, pEpochs=500)
    

def fnRunSPY2015(pBlnGridSearch =False,pLook_Back=10, pHorizon=1):
    '''Run the RNN predictions on the SPY ETF using dates below
    Parm: pBlnGridSearch -true if we perform grid searching, false if not
    pLook_Back -number of days to look back within the model
    pHorizon -days ahead to forecast
    '''
    #global lstrStock
    global learn_rate
    lStartDateTrain=datetime.date(2014 , 9, 3)
    lEndDateTrain=datetime.date(2015 , 12, 31)

    lStartDateTest=datetime.date(2016 ,1, 4)
    lEndDateTest=datetime.date(2016, 5, 31)

    lstrStock="SPY"

    learn_rate=.00015

    fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,pBlnGridSearch,\
           pLearnRate=learn_rate, pDropout=.2, pLayers=3, pNeuronMultiplier=3,pLook_Back=pLook_Back,\
           pHorizon=pHorizon,pBatchSize =8, pEpochs=300)


def fnConvertPredictionsTo1Day(pPredictions, pHorizon):
    '''convert the predictions from multi sequence predictions over multiple days to just the
    1 day of interest and return that 1 day. when using a horizon >1 there will be multiple outputs from model
    1 per day up to horizon
    parms: pPredictions -array of predictions, multidim.
    pHorizon -forecast horizon >=1
    '''
    result=[x[pHorizon-1] for x in pPredictions]
    result =np.array(result)
    return result

def fnMain(pSymbol="", pStartDateTrain=None, pEndDateTrain=None, pStartDateTest=None,
           pEndDateTest=None, pModelFile=None,pBlnGridSearch=False, pLearnRate=.01, pDropout=.2, pLayers=1,\
           pNeuronMultiplier=1, pLook_Back=10, pHorizon=1, pEpochs=300, pBatchSize=10):
        '''
        This function either loads a pretrained RNN model or creates one and trains it. 
        Following the training, the model is used to predict on the test data set and
        the accuracy is reported and plotted against a benchmark
        Parms: pSymbol symbol of stock/ETF
        pStartDateTrain -start date of training data
        pEndDateTrain -end date of training data
        pStartDateTest - start date of testing data
        pEndDateTest -end date testing data
        pModelFile -optional model file to preload instead of training a model ( to save time)
        pBlnGridSearch -true if performing grid search
        nLearnRate -learning rate for model to use
        nDropout -drop out term for model to use
        nLayers -number of layers in the model
        nNeuronMultiplier -number of neurons in the model will be created by numFeatures * nNeuronMultiplier
        
        returns nothing
        '''
        print ('Running model for : ' +pSymbol)
        #these parms are used throughout and made global to reduce passing around so much.
        global lstrStock, numFeatures,look_back,horizon,batch_size,learn_rate

        lstrStock =pSymbol #'SPY'
        #number of features
        numFeatures=0

        #LOOKBACK window
        look_back =pLook_Back #25 #7 #LOOK back at 30 had worse fit than 20s

        #flag for stateful network or not
        blnStateful =True
        #flag for downloading data from web
        if os.environ['COMPUTERNAME']!='NY01-5H7YH72':
            blnUseWeb =True
        else:
            blnUseWeb =False
        #flag to remove adj close price from features
        blnRemoveAdjClose =False

        #number of training epochs
        nEpochs =pEpochs
        #forecast horizon
        horizon =pHorizon #1

        print ('Forecast horizon ='+str(horizon)) 

        #features to use in RNN model
        lstCols =['VolumeSlope',
        'MACD',
        'EMV',
        #'rollingMean20',
        'rollingStdev20',
        'RealBody',
        'SinePhase',
        'Volume',
        'SineAmp',
        #'DiffercenceBtwnAvgVol',
        #'StdDevSlope', 
        'ForceIndex',
        'LowerShadow',
        'UpperShadow',
        'SineFreq',
        'SineOffset',
        'Adj Close',
        'OBV'
        ]

        #count number of features, which is needed for the batch input shape parm.
        
        numFeatures =len(lstCols)

        #if blnRemoveAdjClose:
        #numFeatures=numFeatures-1

        dfTrain=None
        dfTest=None

        #are we downloading from the web ?
        if blnUseWeb:
            #download the training data from yahoo finance
            dfTrain =fnGetYahooStockData(pStartDateTrain,pEndDateTrain,pSymbol)

            #download the TESTING data from yahoo finance
            dfTest =fnGetYahooStockData(pStartDateTest,pEndDateTest,pSymbol)

        #get the training data set , target, benchmark and original data for comparison
        trainX, trainY, trainBMark, dfStockTrain  = fnGetStockData( lstrPath+lstrStock+'dfQuotes.csv',
                                                40, look_back, horizon,lstCols,blnUseWeb,blnRemoveAdjClose,dfTrain)

        #get the test data set , target, benchmark and original data for comparison
        testX, testY,testBMark,dfStockTest   = fnGetStockData(lstrPath+lstrStock+'dfQuotesTest.csv',
                                                     40, look_back, horizon,lstCols,blnUseWeb,blnRemoveAdjClose,dfTest)

        if False:
                #plot stock charts to show volatility & trend,etc..
                fnPlotChart(pBenchmark =dfStockTrain,pStrStock=lstrStock,pStrTitle="Training Data: " + lstrStock +" Historical Prices",pBenchLabel=lstrStock+" Adjusted Closing Price")
                fnPlotChart(pBenchmark =dfStockTest,pStrStock=lstrStock,pStrTitle="TESTING Data: " +lstrStock +" Historical Prices",pBenchLabel=lstrStock+" Adjusted Closing Price")
        
        #plot stationary time series
        
        #fnPlotChart(pBenchmark =trainBMark, pStrStock=lstrStock,pStrTitle="Training Data: " + lstrStock +" Historical Prices AFTER Stationarity Transformation",
#		pBenchLabel=lstrStock+" " +str(horizon) +" differencing of Log Adjusted Closing Price")
        # SET THE batch size
        batch_size =pBatchSize # 2*look_back #*2#*5 #higher batch sz gives better results for SPY.
		
        print('batch_size ='+ str(batch_size))
        blnLoadModel =False
        #run_network(X_train=trainX,y_train =trainY,X_test =testX,y_test =testY)

        #slice off data which is not divisible by batch size
        testBMark,t =fnSliceOffDataPerBatchSize(testBMark,None ,batch_size)

        lenTestData =len(testX)
        lRemainder =lenTestData % batch_size

        lenTestData =lenTestData-lRemainder
        #lenTestData=lenTestData-1

        testX =testX[:lenTestData]
        testY =testY[:lenTestData]

        #lEndValidation =batch_size*2
        #validationX=testX[:lEndValidation]
        #validationY=testY[:lEndValidation]


        #testX =testX[batch_size:lenTestData]
        #testY =testY[batch_size:lenTestData]

        #must slice off training data to be divisible by batch_size
        lLenData =len(trainX)
        lRemainder =lLenData % batch_size
        lLenData =lLenData-lRemainder

        trainX =trainX[:lLenData]
        trainY=trainY[:lLenData]

        if pBlnGridSearch:
            #grid searching using different learning rates
            #the learning rate cannot (easily) be fed into the grid search so we
            #use different preset learning rates
            #and iteratively the grid search with them.
            #lrList =[ 0.02, .008, .004, .001, .0003]
            lrList =[ .00001,.0002, .0003,.0005, .001, .003, .007,.01,.03]
            parmList=[]
            for lr in lrList:
                learn_rate=lr
                #fnGridSearchModel returns best fit score and parms
                result, params =fnGridSearchModel(trainX, trainY)
                params['learn_rate']=lr
                parmList.append((result, params))
                parmList.sort()

            print('-----Grid Search Result Score and Parameters-----')
            for i in parmList:
                print(i)
            print('-----Grid Search Results-----')
            
        if blnLoadModel:
                lModelPath=os.path.dirname(os.path.abspath(__file__))
                model = load_model(lModelPath+'KerasStockModel.h5')
        else:

            model=fnGetModel(False, look_back,nEpochs, pLearnRate, pDropout, pLayers,pNeuronMultiplier)

            # learning schedule callback
            lrate = LearningRateScheduler(step_decay)
            callbacks_list = [lrate]
            callbacks_list=None
            
            print ('callbacks used are ' +str(callbacks_list))
            if blnStateful:
                    #validation_split must allow for samples  div. batch_size to be perfectly
                    #divisible, otherwise there is an error using stateful network.
                    for i in range(nEpochs): #10 ITERATIONs is best thus far
                                    history=model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size , callbacks =callbacks_list,
                                                      verbose=2, shuffle=False) #validation_split=.15,
                                    #history.history['loss']

                                    model.reset_states()
                                    print (i)
            
            else:
                    #produces MEMORY ERROR after 992 epochs
                    model.fit(trainX, trainY, nb_epoch=nEpochs,shuffle=True, #   callbacks=[ResetStatesCallback()], #shuffle=False,
                                    batch_size=batch_size, verbose=2)#, validation_split=.15) #validation_data=(testX, testY), verbose=2)
                    #print ('using linear on last layer, hard tanh and tanh on LSTMs')
                    #num samples for trainX and testX must be divisible by batch_size!!!

            # make predictions
            trainPredict = model.predict(trainX, batch_size=batch_size)

            model.save(lstrPath+'KerasStockModel_'+lstrStock+'.h5') 

        #must reset the state of the network
        if blnStateful:
                model.reset_states()

        #make predictions on test data using RNN model
        testPredict = model.predict(testX, batch_size=batch_size)

        # invert predictions
        if False:
                trainPredict = scaler.inverse_transform(trainPredict)
                trainY = scaler.inverse_transform([trainY])
                testPredict = scaler.inverse_transform(testPredict)
                testY = scaler.inverse_transform([testY])
        else:
                pass

        #convert predictions and train/test data back to 1 item per sample
        trainPredict=fnConvertPredictionsTo1Day(trainPredict,horizon)
        testPredict=fnConvertPredictionsTo1Day(testPredict,horizon)
        trainY=fnConvertPredictionsTo1Day(trainY,horizon)
        testY=fnConvertPredictionsTo1Day(testY,horizon)


        testR2 =r2_score(testY,testPredict)
        print ('R2 score on Train Returns:' +str(r2_score(trainY,trainPredict)))

        print ('R2 score on Test Returns:' +str(testR2))

        if False:
            print ('R2 score on Benchmark Returns:' +str(r2_score(testY,testBMark['Adj Close'])))

            print ('Total training data points ' + str(len(trainY)))
            print ('Total testing data points ' + str(len(testY)))
                   
            dfPredictions =fnComputePredictedPrices(testBMark, dfStockTest, testPredict)
            print ('Accuracy Ratio on Test Data Using RNN Model: ' 
            +str(CalculateAccuracyRatio(dfPredictions['PredictedAdjClose'],dfPredictions['Adj Close'])))

            
            print ('Accuracy Ratio on Test Data Using Benchmark: '+\
            str(CalculateAccuracyRatio(dfPredictions['BenchmarkAdjClose'],dfPredictions['Adj Close'])))

        #np.savetxt("C:\\temp\\testY.csv", testY, delimiter=',')
        #np.savetxt("C:\\temp\\testX.csv", testX, delimiter=',')

##        plt.plot(testY, 'b-',label='Actual '+lstrStock)
##        plt.plot(testPredict, 'g-', label='*Predicted* '+lstrStock)
##
##        legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
##        plt.show()
##
##        plt.plot(trainY,label='Train Actual '+lstrStock)
##        plt.plot(trainPredict,label='*Train Predicted* '+lstrStock)
##        legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
##        plt.show()
##        plt.clf()

        if False:
                #fnPrintPredVsActuals(testBMark,testPredict,testY,"Adjusted Closing Returns")
                #print the predicted, actual and benchmark prices
                pd.set_option('display.max_rows', 500)
                pd.set_option('display.height', 500)
                display(dfPredictions[['Adj Close','PredictedAdjClose', 'BenchmarkAdjClose']])
                
                fnPlotChart(testBMark, testPredict,testY, lstrStock,"Predictions On Test Data (log differences)",pBlnPlotBenchMark=False)
                
                fnPlotChart(testBMark, None,testY, lstrStock,"Benchmark Predictions On Test Data (log differences)")
                plt.clf()
                #mismatch here ?
                #fnPlotChart(trainBMark, trainPredict,trainY, lstrStock,"Predictions On Training Data",pBlnPlotBenchMark=False)
                #plt.clf()

                #plot predictions and adj close.
                fnPlotChart(dfPredictions, dfPredictions[['PredictedAdjClose']],\
                    dfPredictions[['Adj Close']], lstrStock,"Predictions On Test Data (Prices)",\
                    '','BenchmarkAdjClose',pBlnPlotBenchMark=False)

                #plot just benchmark and adj close
                fnPlotChart(dfPredictions, None,\
                    dfPredictions[['Adj Close']], lstrStock,"Predictions On Test Data (Prices)",\
                    'Naive Benchmark','BenchmarkAdjClose',pBlnPlotBenchMark=True)
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

        return testR2

def fnMainWrapper(*pArgs):
        pLB=int(pArgs[0][0])
        pBatch=int(pArgs[0][1])
        pLearnRate=np.float64(pArgs[0][2])
        pLayers=int(pArgs[0][3])
        pNeurons=int(pArgs[0][4])
        pDropout=int(pArgs[0][5])
        pHorizon=1

        lStartDateTrain=datetime.date(2014 , 9, 3)
        lEndDateTrain=datetime.date(2015 , 12, 31)

        lStartDateTest=datetime.date(2016 ,1, 4)
        lEndDateTest=datetime.date(2016, 5, 31)

        #pDropout=.3215
        lstDropouts=[.05,.08,.1,.15,.2,.22,.25,.3,.4,.45,.5]
        pDropout=lstDropouts[pDropout]
        
        print (pArgs)
        
        lstrStock="SPY"

        #learn_rate=.0003

        result=fnMain(lstrStock,lStartDateTrain,lEndDateTrain, lStartDateTest,  lEndDateTest,None,False,\
           pLearnRate=pLearnRate, pDropout=pDropout, pLayers=pLayers, pNeuronMultiplier=pNeurons,pLook_Back=pLB,\
           pHorizon=pHorizon,pBatchSize =pBatch, pEpochs=180)
        
        return -result

if __name__=='__main__':
    #initGuess=[numLayers,lNeurons]
    #lBounds=[(5,9),(20,700)]
##    def fnMainWrapper(*pArgs):
##        pLB=int(pArgs[0][0])
##        pBatch=int(pArgs[0][1])
##        pLearnRate=(pArgs[0][2])
##        pLayers=int(pArgs[0][3])
##        pNeurons=int(pArgs[0][4])
##        pDropout=int(pArgs[0][5])
    lBounds=[(10,25),(1,12),(.0000517,.00217),(3,3),(3,3),(0,10)]
    '''Each of these fnRunXXX functions will run a forecast for the associated ETF  -SPY, OIL, VXX, QQQ '''
    result=differential_evolution(func=fnMainWrapper,bounds=lBounds,disp=1)
    print(result)	
    #fnRunSPY2015(False,18,1)
    #fnRunOIL(False,9,5)
    #fnRunVXX(False,9,5)
    #fnRunQQQ(False,9,1)

