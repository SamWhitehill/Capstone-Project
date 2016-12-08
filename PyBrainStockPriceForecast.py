from __future__ import print_function
#https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/

#import pandas_datareader.data as web
import math
from pandas.io import data as web 
import pandas as pd
from sklearn.metrics import mean_squared_error
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pybrain.datasets import SequentialDataSet,SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer, LinearLayer, SigmoidLayer, SoftmaxLayer,TanhLayer
from pybrain.structure.networks import RecurrentNetwork
from pybrain.structure.connections import FullConnection
from pybrain.supervised import RPropMinusTrainer
from sys import stdout

from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import minimize




print ("Cite Reference: https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/")

lTicker ="FSLR"

#data = web.get_data_yahoo('SPY', dt.datetime(2014,11,22), dt.datetime(2016,8,22))
global data
data =pd.read_csv(lTicker+".csv")
data[['Open','High','Low','Adj Close','Volume']] =data[['Open','High','Low','Adj Close','Volume']].astype(float)


#data = data.ix[:,5].values.tolist()

data = data[['Open','High','Low','Adj Close','Volume']].values.tolist()

# normalize the dataset

scaler =MinMaxScaler(feature_range=(0, 1)).fit(data)

data = scaler.transform(data).tolist()

#ds = SequentialDataSet(7, 1)


def fStockForecast(params ): #pMidLayer=2,numDaysLookBack=140,pconvergence_threshold=100 ):
    #force parms to integers
    np.random.seed(43)
    #pMidLayer=pMidLayer.astype(int)
    #numDaysLookBack=numDaysLookBack.astype(int)
    #pconvergence_threshold=pconvergence_threshold.astype(int)

    pMidLayer =int(params[0])
    numDaysLookBack =int(params[1])
    pconvergence_threshold =int(params[2])

    #print(pMidLayer,numDaysLookBack,pconvergence_threshold)
    
    global data 
    numDaysAhead =44
    #numDaysLookBack=140
    numDim =5
    #data=lstStockHistory
    #possibly try
    #assert(X.shape[0] == y.shape[0])
    #DS.setField('input', X)
    #DS.setField('target', y)

    #ds = SequentialDataSet(5, 1)
    ds = SupervisedDataSet(numDaysLookBack*numDim, 1)

    for i in range(0,len(data)-(numDaysAhead+numDaysLookBack+1)):
        # training: previous 10 days and same day last year
        #sample = data[i:(i+6)] #+ [data[i-252]]
        sample = data[i:(i+numDaysLookBack)]
        # target: 5 days ahead
        #target = data[i+10]
        target =  [data[i+numDaysLookBack+numDaysAhead-1][3]] #adj close is 3rd item
        flattenedSample =[item for sublist in sample for item in sublist]

        ds.addSample(flattenedSample,target)
        if False:
            #for each row of training data for this sample (window of trading days)
            for input in sample:
       
                #need to add samples in a loop 
                ds.appendLinked(input, target)

        #ds.newSequence()

    #ds.removeSequence(i+1)

    print ('done with adding sequences')

    #tstdata, trndata = ds.splitWithProportion( 0.2 )

    #ds=trndata
    #net = buildNetwork(7, 30, 1, hiddenclass=LSTMLayer, outputbias=False, recurrent=True)
    #reduced middle layer by Sam to 6, this must match number of past days used!
    ### BUILD recurrent network explicitly

    if False:
        n =RecurrentNetwork()
        inLayer = LinearLayer(6)
        hiddenLayer = LSTMLayer(3)
        outLayer = SigmoidLayer(1)

        #n.addInputModule(inLayer)
        #n.addModule(hiddenLayer)
        #n.addOutputModule(outLayer)

        n.addInputModule(LinearLayer(6, name='in'))
        n.addModule(LSTMLayer(4, name='hidden'))
        n.addOutputModule(SigmoidLayer(1, name='out'))
        n.addConnection(FullConnection(n['in'], n['hidden'], name='c1'))
        n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
        n.addRecurrentConnection(FullConnection(n['hidden'], n['hidden'], name='c3'))

        n.sortModules()

    #BUILD RNN using SHORTCUT
    lMiddleLayer =pMidLayer
    net = buildNetwork(numDim*numDaysLookBack,lMiddleLayer,1 , hiddenclass=LSTMLayer,  outputbias=False, recurrent=True) # outclass=TanhLayer)

    trainer = RPropMinusTrainer(net, dataset=ds)

    #print('training until convergence')
    trainer.trainUntilConvergence(dataset=ds,validationProportion=0.25,convergence_threshold=130) #convergence_threshold ? continueEpochs =800,
    #200 epochs per cycle, 3 cycles is horrible
    if False:                    
        train_errors = [] # save errors for plotting later
        EPOCHS_PER_CYCLE = 4                                                                  
        CYCLES = 400 # was 500
        EPOCHS = EPOCHS_PER_CYCLE * CYCLES
        for i in range(CYCLES): #xrange(CYCLES):
            trainer.trainEpochs(EPOCHS_PER_CYCLE)
            train_errors.append(trainer.testOnData())
            epoch = (i+1) * EPOCHS_PER_CYCLE
            print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
            stdout.flush()

        print()
        print("final error =", train_errors[-1])

    #target = np.zeros(ds.getSequenceLength(0))
    #prediction = np.zeros(ds.getSequenceLength(0))

    #target = np.zeros(ds.getNumSequences())
    #prediction = np.zeros(ds.getNumSequences())

    target = np.zeros(len(ds))
    prediction = np.zeros(len(ds))
         
    i = 0

    #for s, t in ds.getSequenceIterator(0):
    #    target[i] = t
    #    prediction[i] = net.activate(s)
    for inp, targ in ds:
        target[i] =scaler.inverse_transform(targ.tolist()*numDim)[3]
        prediction[i] = net.activate(inp)
        #prediction[i] = scaler.inverse_transform([prediction[i]]) 
        prediction[i] = scaler.inverse_transform([prediction[i]]*numDim)[3]
        i += 1

    if False:
            for SeqCtr in range(ds.getNumSequences()):
                    input, target =ds.getSequence(SeqCtr)
                    #for s, t in ds.getSequenceIterator(SeqCtr):
                    #target[i] = t
                    #target[i] = scaler.inverse_transform(t[0]) #scaler.inverse_transform(t.tolist()*numDim)[3]
                    target[i] =scaler.inverse_transform([x[0] for x in target[0:numDim].tolist()])[3]
        
                    #prediction[i] = net.activate(s)
                    ###TRANSFORM input into serialized data so it can be activated
                    input=input.flatten()
                    prediction[i] = net.activate(input)
                    #prediction[i] = scaler.inverse_transform([prediction[i]]) 
                    prediction[i] = scaler.inverse_transform([prediction[i]]*numDim)[3]
                    i += 1



    #compute RMSE
    testScore = math.sqrt(mean_squared_error(target, prediction))
    print('Test Score: %.2f RMSE' % (testScore))
    
    blnPlot=False

    if blnPlot:

        plt.plot(target)
        plt.plot(prediction)

        # prediction error
        print(target - prediction)




        plt.suptitle(lTicker+' - Predictions. Middle layer: ' + str(lMiddleLayer) + ' RMSE: '+str(testScore)
                     +' num days ahead ' +str(numDaysAhead)+' look back ' +str(numDaysLookBack), fontsize=12, fontweight='bold')
        plt.show()


    print ('RMSE ' + str(testScore) + ' parms '+ str(params))
    return testScore
    #Test Score: 2.41 RMSE
    #trainer.trainUntilConvergence(dataset=ds,validationProportion=0.35,convergence_threshold=250) #convergence_threshold ? continueEpochs =800,



    # 3.5 RMSE
        #EPOCHS_PER_CYCLE = 2                                                                  
        #CYCLES = 450 # was 500



    #Test Score: 4.89 RMSE
        #EPOCHS_PER_CYCLE = 1                                                                  
        #CYCLES = 450 # was 500


    #net = buildNetwork(numDim*numDaysLookBack,3,1 , hiddenclass=LSTMLayer,  outputbias=False, recurrent=True)
    #RMSE =3.32
    #trainer.trainUntilConvergence(dataset=ds,validationProportion=0.35,convergence_threshold=250) 



    #net = buildNetwork(numDim*numDaysLookBack,2,1 , hiddenclass=LSTMLayer,  outputbias=False, recurrent=True) # outclass=TanhLayer)
    #numDaysAhead =7
    #numDaysLookBack=110
    #numDim =5
    #trainer.trainUntilConvergence(dataset=ds,validationProportion=0.25,convergence_threshold=50) #convergence_threshold ? continueEpochs =800,
    #RMSE


#
# fStockForecast( pMidLayer=2,numDaysLookBack=140,pconvergence_threshold=100 )
#fStockForecast( 44,170 )

x0 = [3,140,100]
lBounds =[(2,10),(10,200),(10,400)]

res = minimize(fStockForecast, x0, method='L-BFGS-B', bounds=lBounds,
                options={'disp': True,'eps':1})


print(res)
