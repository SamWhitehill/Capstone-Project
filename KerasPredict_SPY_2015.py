#import os
#print(os.path.expanduser('~'))

#References used to help build this module:
#http://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/
#https://\.com/CanePunma/Stock_Price_Prediction_With_RNNs/blob/master/stock_prediction_keras_FINAL.ipynb
#https://github.com/anujgupta82/DeepNets/blob/master/Online_Learning/Online_Learning_DeepNets.ipynb
#http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
#http://philipperemy.github.io/keras-stateful-lstm/
# LSTM for international airline passengers problem with memory
#http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction
#cite: http://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
#https://github.com/FreddieWitherden/ta/blob/master/ta.py
#http://www.johnwittenauer.net/a-simple-time-series-analysis-of-the-sp-500-index/


from sklearn.ensemble import ExtraTreesRegressor
#from scipy.stats import boxcox
from fitSine import fnFitSine


from sklearn.model_selection import GridSearchCV
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
from MainForecastModule import fnGetNaturalLogPrices
from MainForecastModule import fnComputeFeatures
from keras.models import load_model
from MainForecastModule import organize_data
from MainForecastModule import window_stack

from statsmodels.tsa.stattools import adfuller

from IPython.display import display
#from modGetStockData import fnGetStockData 
#from PowerForecast import run_network


''' This is a workaround fix to Keras when grid searching'''
from keras.wrappers.scikit_learn import BaseWrapper
import copy

def custom_get_params(self, **params):
    res = copy.deepcopy(self.sk_params)
    res.update({'build_fn': self.build_fn})
    return res

BaseWrapper.get_params = custom_get_params
'''End workaround fix to Keras grid search '''


lstrPath ="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"
#LOOKBACK window
look_back =10 #LOOK back at 30 had worse fit than 20s
horizon =5
lstrStock ='SPY'

#set the random num gen seed for reproducability
np.random.seed(7)


print ('horizon ='+str(horizon)) 
def reportStationarity(pDfTimeSeries):
	#Perform Dickey-Fuller test:
    print( 'Results of Dickey-Fuller Test:')
    dftest = adfuller(pDfTimeSeries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    #print (dfoutput)
    display(dfoutput)
    print ("")


def fnGetStationaryTimeSeries(pDf,pLstCols=None):
    #return the data frame with differences of log of open, hi, low , close prices
    #this should make it better for time series analysis
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
        pDf[fld] =pDf[fld] -pDf[fld].shift()

    return pDf
	
def fnComputeFeatureImportances(trainX,trainY,lstCols):
    #PRINT FEATURE IMPORTANCES
    #******************************************************
    ETmodel = ExtraTreesRegressor()
    trainX =np.reshape(trainX,(trainX.shape[0],trainX.shape[2]))
    ETmodel.fit(trainX, trainY)
    fi =ETmodel.feature_importances_
    print (fi)

    for item in lstCols:
            print(item)

def fnPrintPredVsActuals(pDfBenchmark, pPrediction,pActual, pStrHeader):
    #pDfBenchmark - dataframe containing benchmark data
    #pPrediction - np array of predicted data
    #pActual - np array of actual data

    pDfBenchmark['Predicted']=pPrediction.tolist()
    pDfBenchmark['Actual']=pActual.tolist()
    
    print(pStrHeader)

    display(pDfBenchmark)


def fnGetStockData(pStrFileName,nDaysReturnLookBack, look_back, horizon,pLstCols,pRemoveAdjFromFeatures=True):
    #retrieve stock data from web or  file
    #process it by calculating log returns over specified lookback
    #calculate features needed
    #normalize data via feature scaling
    # pivot data into numpy arrays so it can be fed into RNN model.
    #return X features and Y target data from original stock data 
    
    dataframe = pd.read_csv(pStrFileName,  engine='python')

    dataframe.Date = pd.to_datetime(dataframe.Date) 
    dataframe.index =dataframe['Date']

    originalDF =dataframe.copy(deep=True)
    ndaysNatLog=nDaysReturnLookBack #40


    dataframe =fnGetStationaryTimeSeries(dataframe)   
    dataframe =dataframe[ pd.notnull(dataframe['Adj Close'])]
    #print ('Converting to stationary AFTER building features')
    
    slopeLB =20
    dataframe,lstColsSR=fnComputeFeatures(dataframe,look_back, slopeLB,horizon,22,4)


    print ('slope lookback at ' + str(slopeLB))
	# dataframe=fnGetNaturalLogPrices(dataframe,ndaysNatLog)

	#REMOVE FIRST ROWS WHICH contain NaN
    dataframe =dataframe[ pd.notnull(dataframe['MACD'])] #
    dataframe =dataframe[ pd.notnull(dataframe['SineFreq'])]
    dataframe =dataframe[ pd.notnull(dataframe['DiffercenceBtwnAvgVol'])]
    
    reportStationarity(dataframe['Adj Close'])

    #fnComputeFeatures(pDf,pNumDaysLookBack,pSlopeLookback, pDaysAhead=8,pSRLookback=11,pSegments=4)
    #dataframe,lstColsSR=fnComputeFeatures(dataframe,look_back, 3,horizon,22,4)

    #lstCols =['Adj Close','rollingStdev20','rollingMax20','rollingMin20','OBV','upper_band','lower_band']
    #'RealBody','Color','BarType','UpperShadow','LowerShadow'
    lstCols=pLstCols[:]
    #lstCols =['2DayNetPriceChange','CloseSlope','rollingMean20','Adj Close','MACD','Volume',#'upper_band','lower_band',
     #        'High','Low','rollingStdev20','rollingMax20','rollingMin20']#,'ZigZag']#,'fastk','fulld','fullk'] #,'upAroon','downAroon']#,'fastk','fulld','fullk'] # ['S1','S2','S3','R1','R2','R3']
    #lstCols =['Adj Close','CloseSlope', 'rollingMean20','rollingStdev20','rollingMax20','rollingMin20']
    #lstCols =['Adj Close'], .45 R sq.
    print ('running with slope lookback =4, look_back =' + str(look_back)+ ', natlog =' +str(ndaysNatLog) +' days ahead, 2 total LSTMS')

    #MACD adds value to R^2, ForceIndex does NOT! Volumeslope does not
    #lstCols =['Adj Close','CloseSlope','rollingMax20','rollingMin20','HighLowRange','rollingStdev20','rollingMean20'] -neg. r^2
    #lstCols =['Adj Close','rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol','HighLowRange','rollingMean20','Volume']# .01 rsq
    #lstCols =['Adj Close'] #.19 #,'rollingStdev20','RSI','MACD','DiffercenceBtwnAvgVol']
    numFeatures =len(lstCols)	
    print (lstCols)
 
    targetSet =dataframe['Adj Close'].values
    targetSet =targetSet.astype('float64')

	#create a naive benchmark, which is simply last known price before the prediction
    benchmark=dataframe['Adj Close'][look_back-1: len(dataframe)-horizon]
    benchmark['Date'] =benchmark.index
    
    if pRemoveAdjFromFeatures:
        lstCols.remove('Adj Close')
        numFeatures=numFeatures-1
    dataframe=dataframe[lstCols] #,'Volume','rollingMax20','rollingMin20']]

    dataset = dataframe.values
    dataset = dataset.astype('float64')

    # normalize the dataset
    if True:
            #print ('temporarily not scaling so we can debug')
            #DO NOT scale the target, only features
            scaler = MinMaxScaler(feature_range=(-1, 1))
            #Targetscaler = MinMaxScaler(feature_range=(0, 1))
            
            dataset = scaler.fit_transform(dataset)
            
            #targetSet =Targetscaler.fit_transform(targetSet)

    else:
            print ('temporarily not scaling so we can debug')




    # split into train and test sets
    train_size = int(len(dataset)) # * 0.65)

    #test_size = len(dataset) - train_size
    #train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
    train = dataset[0:train_size,:]

    #trainTarget, testTarget =targetSet[0:train_size], targetSet[train_size:len(targetSet)]
    trainTarget =targetSet[0:train_size]
    # reshape into X=t and Y=t+1

    trainX, trainY = create_dataset(train,trainTarget ,look_back,horizon)


    #fnComputeFeatureImportances(trainX,trainY,lstCols)

    #testX, testY = create_dataset(test,testTarget, look_back,horizon)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], numFeatures))
    #testX = np.reshape(testX, (testX.shape[0], testX.shape[1], numFeatures))

    return trainX, trainY,benchmark,originalDF


class ResetStatesCallback(Callback):
    def __init__(self):
        self.counter = 0

    def on_epoch_end(self, epoch, logs={}):
        self.model.reset_states()  
        print ('resetting')      
		#if self.counter % max_len == 0:
  #          self.model.reset_states()
  #      self.counter += 1

def fnPlotChart(pBenchmark=None, pPredictions=None,pActual=None, pStrStock="", pStrTitle="", pBenchLabel=""):
	#display chart of benchmark , predicted and actual stock data (returns, prices,etc..)
	
    #np.savetxt("C:\\temp\\testY.csv", testY, delimiter=',')
    #np.savetxt("C:\\temp\\testX.csv", testX, delimiter=',')
    lstrStock =pStrStock

    # plot the benchmark
    if pActual!=None:
            plt.plot_date(pBenchmark['Date'], pActual, 'k-',label='Actual '+lstrStock)

    benchLabel ='Benchmark '+lstrStock
    if pBenchLabel!="":
        benchLabel=pBenchLabel
    
    plt.plot_date(pBenchmark['Date'], pBenchmark['Adj Close'], 'r-',label=benchLabel)
	
    if pPredictions!=None:
            plt.plot_date(pBenchmark['Date'], pPredictions, 'g-',label='*Predicted* '+lstrStock)


    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=25))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
    plt.gcf().autofmt_xdate()

    #plt.plot(testY,label='Actual '+lstrStock)
    #plt.plot(testPredict,label='*Predicted* '+lstrStock)

    plt.suptitle(pStrTitle,
                    fontsize=11, fontweight='bold')

    legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
    plt.show()

def fnTrainModelForMap(model, x,y,pBatchSize):
	model.fit(x, y,nb_epoch =1,  batch_size=pBatchSize, shuffle=False)
	model.reset_states()
	return model

def fnTrainViaCallback(model, x,y,pBatchSize):
	model.fit(x, y,nb_epoch =40, callbacks=[ResetStatesCallback()], batch_size=pBatchSize, shuffle=False)
	return model

# learning rate schedule
#A little tweaking gives you a three parameters approach:
#The decaying function used for decaying epsilon was:
#1 x math.exp(-ParmA x trialNum)/( 1 + math.exp(ParmB x (trialNum-ParmC)))
#*To be clear, x means multiply, I cannot use asterisk in jupyter.
#The following parameters used in the function are below:
# ParmA= -5E-11 ParmB= 0.104 ParmC= 210 trialNum is the trial number (e.g., 1,2,3...).

#np.exp(-0.02*x)/( 1 + np.exp(2*(x-5)))

def step_decay(epoch):
	initial_lrate = 0.0012 #The Alpha used was .95 
	#increasing the learning rate (initial appears to flatten out the prediction
	drop = 0.5
	epochs_drop = 10.0
	ParmA= -5E-11 
	ParmB= 0.104 
	ParmC= 50 
	lrate=initial_lrate * math.exp(-ParmA * epoch)/( 1 + math.exp(ParmB * (epoch-ParmC)))
	#lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	#print ('using decay initial_lrate=.' +str(initial_lrate) +' * math.pow(drop, math.floor((1+epoch)/epochs_drop))')
	#print ('using tanh decay')
	return lrate

#def fnGetModelMultiLayer(blnStateful,batch_size, look_back, numFeatures):
def fnGetModelMultiLayer(neuronsL1,neuronsL2,neuronsL3):
    model = Sequential()
    blnStateful=False
    #layers={'hidden2': numFeatures*12, 'input': 1, 'hidden1': numFeatures*8, 'hidden3': numFeatures*4, 'output': 1}
    #layers={'hidden2': numFeatures*18, 'input': 1, 'hidden1': numFeatures*4, 'hidden3': numFeatures*4, 'output': 1}
    #layers={'hidden2': numFeatures*2, 'input': 1, 'hidden1': numFeatures*6, 'hidden3': numFeatures*6, 'output': 1}
    layers={'hidden2': neuronsL2, 'input': 1, 'hidden1': neuronsL1, 'hidden3': neuronsL3, 'output': 1}
    
    #layers = {'input': 1, 'hidden1': numFeatures, 'hidden2': 128, 'hidden3': 100, 'output': 1}
    #increASING hidden1 flattens!
    #decreasing hidden3 amplifies volatility!

    print(layers)
    #keras.layers.recurrent.LSTM(output_dim, init='glorot_uniform',
    #inner_init='orthogonal', forget_bias_init='one', activation='tanh',
    #inner_activation='hard_sigmoid', W_regularizer=None, U_regularizer=None, b_regularizer=None, dropout_W=0.0, dropout_U=0.0)
    ldropout =.12075
    print ('dropout at: ' + str(ldropout))
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
    
    #model.add(LSTM(numNeurons ,activation='sigmoid',
    #inner_activation='tanh', batch_input_shape=(batch_size, look_back, numFeatures),
    #unroll=True, stateful=blnStateful)) #activation='tanh',inner_activation='tanh'
    #***********
    #using activation='tanh', inner_activation='relu' gives a flat line, when using features computed on stationary data
    #cannot use relu in activation, returns nan
    #sigmoid in activation produces flat line
    #model.add(Dropout(ldropout))
    #model.add(Dropout(ldropout,batch_input_shape=(batch_size, look_back, numFeatures)))
    #cannot use inner_activation =relu with stateful=True, loss =Nan
    #with hard sigmoid got a -.4 R^2
    #with relue got -4.0 R^2
    
    model.add(LSTM( activation='tanh', inner_activation='hard_sigmoid',stateful=blnStateful, #activation='tanh',
            batch_input_shape=(batch_size, look_back, numFeatures),
            output_dim=layers['hidden1'],
            return_sequences=True))
    model.add(Dropout(ldropout))


    model.add(LSTM(activation='tanh', inner_activation='hard_sigmoid',stateful=blnStateful,
            output_dim=layers['hidden2'],
            batch_input_shape=(batch_size, look_back, numFeatures),
            return_sequences=True))
    model.add(Dropout(ldropout))

    model.add(LSTM( activation='tanh', inner_activation='hard_sigmoid',stateful=blnStateful,
            output_dim=layers['hidden3'],
            batch_input_shape=(batch_size, look_back, numFeatures),
            return_sequences=False))
    model.add(Dropout(ldropout))

    model.add(Dense(
            output_dim=layers['output']))

    #model.add(Dense(layers['hidden3'], activation='relu'))
    #model.add(Dense(output_dim=layers['output'], activation='linear'))
    
    model.add(Activation("linear"))
    learn_rate=0.0005242175
    #momentum=0
    #optimizer = SGD(lr=learn_rate, momentum=momentum)
    optimizer = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-10, decay=0.000001)
    #optimizer = Adagrad(lr=learn_rate, epsilon=1e-08, decay=0.00)
    #Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adam R sq of -2
    model.compile(loss='mean_squared_error',optimizer=optimizer) # optimizer='adam')
    return model
    
def fnGridSearchModel(trainX, trainY):
    #Reference:http://machinelearningmastery.com/grid-search-hyperparameters-deep-learning-models-python-keras/
    model = KerasRegressor(build_fn=fnGetModelMultiLayer, nb_epoch=100, batch_size=batch_size, verbose=2)
    
    # define the grid search parameters
    neuronsL1 = [numFeatures, 20, 25, 30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    neuronsL2 = [numFeatures, 20, 25, 30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    neuronsL3 = [numFeatures, 20, 25, 30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]


    
    param_grid = dict(neuronsL1=neuronsL1,neuronsL2=neuronsL2,neuronsL3=neuronsL3 )
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
    grid_result = grid.fit(trainX, trainY)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']

def fnGetModel(blnStateful=True, pLook_Back=7):
    model = Sequential()
    #numNeurons =100 # try 4*look_back  #increasing neuron sappears to increase volatility too much ?
    numNeurons=2*numFeatures #4*look_back *numFeatures
    #model.add(Dropout(0.25,batch_input_shape=(batch_size, look_back, numFeatures)))
    lDRRate =.2555  #increased dr for OIL improves
    nLayers =3 #less layers from 6 to 2 is worse R^2
    #tried 16 layers and 1*numfeatures neurons but not enough volatility in forecast
    print ('dropout at ' + str(lDRRate) )

    #model.add(LSTM(numNeurons ,activation='tanh',inner_activation='hard_tanh', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True,consume_less='cpu', stateful=blnStateful,return_sequences=True))
    #model.add(Dropout(0.25,batch_input_shape=(batch_size, look_back, numFeatures)))
    #tanh and tanh together is completely flat, tanh and sigmoid has good volat. but not enough
    #model.add(LSTM(numNeurons ,activation='tanh',inner_activation='tanh', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True,consume_less='cpu', stateful=blnStateful,return_sequences=True))
    #using sigmoid sigmoid for acti and inner flatlines
    
    model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
    #print ('inner activation is sigmoid')
    if True:
        for i in range(nLayers):
            if i==(nLayers-1):
                model.add(LSTM(numNeurons ,activation='tanh',inner_activation='hard_sigmoid', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful)) #activation='tanh',inner_activation='tanh'
            else:
                model.add(LSTM(numNeurons ,activation='tanh',inner_activation='hard_sigmoid', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
           
            model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))

    if False:
        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
       
        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
        
        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
       
        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
        

        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
       
        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
        
        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
       
        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
       
        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=True))
     

        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
        
        model.add(LSTM(numNeurons ,activation='tanh',inner_activation='relu', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful,return_sequences=False))
     
        model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))


    #model.add(LSTM(numNeurons,activation='tanh',inner_activation='tanh', batch_input_shape=(batch_size, look_back, numFeatures),unroll=True, stateful=blnStateful)) #consume_less='cpu'

    #model.add(Dropout(lDRRate,batch_input_shape=(batch_size, look_back, numFeatures)))
        #REMOVING dropout severely hurts score/fit

    model.add(Dense(1)) #,activation ='relu' -> gives WORSE results.
    model.add(Activation("linear"))

    print ('using fnGetModelMultiLayer  model')
    model =fnGetModelMultiLayer(blnStateful,batch_size, look_back, numFeatures)
                  
    print ('model ' +str(nLayers)+' layers ' + str(numNeurons) + ' neurons per layer, final activation is linear')
    # Compile model
    #learn_rate=0.00006215 #reducing the learning rate improves the fit and r squared!!!
    #learn_rate=0.00004242175
    learn_rate=0.0005242175
    #momentum=0
    #optimizer = SGD(lr=learn_rate, momentum=momentum)
    optimizer = RMSprop(lr=learn_rate, rho=0.9, epsilon=1e-10, decay=0.000001)
    #optimizer = Adagrad(lr=learn_rate, epsilon=1e-08, decay=0.00)
    #Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #adam R sq of -2
    model.compile(loss='mean_squared_error',optimizer=optimizer) # optimizer='adam')

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

# convert an array of values into a dataset matrix
def create_dataset(dataset, targetSet=None, look_back=1,horizon=1, positionY=0):
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

lstCols =['VolumeSlope',
'MACD',
'EMV',
'rollingMean20','rollingStdev20',
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
'SineOffset','Adj Close',
          'OBV'
]


blnRemoveAdjClose =False
numFeatures =len(lstCols)

if blnRemoveAdjClose:
	numFeatures=numFeatures-1

trainX, trainY, trainBMark, dfStockTrain  = fnGetStockData(lstrPath+lstrStock+'dfQuotes2015.csv',40, look_back, horizon,lstCols,blnRemoveAdjClose)
testX, testY,testBMark,dfStockTest   = fnGetStockData(lstrPath+lstrStock+'dfQuotesTest2015.csv',40, look_back, horizon,lstCols,blnRemoveAdjClose)

if True:
    fnPlotChart(pBenchmark =dfStockTrain,pStrStock=lstrStock,pStrTitle="Training Data: " + lstrStock +" Historical Prices",pBenchLabel=lstrStock+" Adjusted Closing Price")
    fnPlotChart(pBenchmark =dfStockTest,pStrStock=lstrStock,pStrTitle="TESTING Data: " +lstrStock +" Historical Prices",pBenchLabel=lstrStock+" Adjusted Closing Price")
# create and fit the LSTM network
batch_size = look_back #*2#*5 #1 tried increasing batch, worse fit for batch_size=1

print('batch_size ='+ str(batch_size))
blnLoadModel =False
#run_network(X_train=trainX,y_train =trainY,X_test =testX,y_test =testY)

testBMark,t =fnSliceOffDataPerBatchSize(testBMark,None ,batch_size)

lenTestData =len(testX)
lRemainder =lenTestData % batch_size

lenTestData =lenTestData-lRemainder
#lenTestData=lenTestData-1
validationX=testX[:batch_size]
validationY=testY[:batch_size]

testX =testX[:lenTestData]
testY =testY[:lenTestData]

#must slice off training data to be divisible by batch_size
lLenData =len(trainX)
lRemainder =lLenData % batch_size
lLenData =lLenData-lRemainder

trainX =trainX[:lLenData]
trainY=trainY[:lLenData]



if blnLoadModel:
	model = load_model(lstrPath+'KerasStockModel.h5')
else:

        model =fnGridSearchModel(trainX, trainY)
        #model=fnGetModel(False, look_back)
	
	#model=fnTrainViaCallback(model,trainX,trainY,batch_size)
	
	#tmp=list(map(fnTrainModelForMap(model,trainX,trainY,batch_size),range(10)))
	#test=[fnTrainModelForMap(model,trainX,trainY,batch_size) for x in range(20)]
	#model=model[len(tmp)-1]

	# learning schedule callback
        lrate = LearningRateScheduler(step_decay)
        callbacks_list = [lrate]

        if False:
            
            for i in range(350): #10 ITERATIONs is best thus far
                    history=model.fit(trainX, trainY, nb_epoch=1, batch_size=batch_size, verbose=2, shuffle=False)
                    #history.history['loss']
                    
                    model.reset_states()
                    print (i)
			
        else:
            #produces MEMORY ERROR after 992 epochs
            model.fit(trainX, trainY, nb_epoch=200,shuffle=True, #   callbacks=[ResetStatesCallback()], #shuffle=False,
                    batch_size=batch_size, verbose=2) #,validation_data=(validationX, validationY)) #validation_data=(testX, testY), verbose=2)
            print ('using linear on last layer, hard tanh and tanh on LSTMs')
            #num samples for trainX and testX must be divisible by batch_size!!!

        # make predictions
        trainPredict = model.predict(trainX, batch_size=batch_size)

        model.save(lstrPath+'KerasStockModel_'+lstrStock+'.h5') 

if False:
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

print ('R2 score on Test Returns:' +str(r2_score(testY,testPredict)))

print ('R2 score on Benchmark Returns:' +str(r2_score(testY,testBMark)))

#fig=plt.figure(figsize=(8,6))
#plt.plot(trainY,label='Actual Train ')
#plt.plot(trainPredict,label='Predicted Train ')
#plt.show()

#np.savetxt("C:\\temp\\testY.csv", testY, delimiter=',')
#np.savetxt("C:\\temp\\testX.csv", testX, delimiter=',')

plt.plot(testY, 'b-',label='Actual '+lstrStock)
plt.plot(testPredict, 'g-', label='*Predicted* '+lstrStock)

legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()

plt.plot(trainY,label='Train Actual '+lstrStock)
plt.plot(trainPredict,label='*Train Predicted* '+lstrStock)
legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')
plt.show()
plt.clf()

if False:
	fnPrintPredVsActuals(testBMark,testPredict,testY,"Adjusted Closing Returns")

	fnPlotChart(testBMark, testPredict,testY, lstrStock,"Predictions On Test Data")
	plt.clf()
	fnPlotChart(trainBMark, trainPredict,trainY, lstrStock,"Predictions On Training Data")
	plt.clf()

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
