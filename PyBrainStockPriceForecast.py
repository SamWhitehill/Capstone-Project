from __future__ import print_function
#https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/

#import pandas_datareader.data as web
import math
from pandas.io import data as web 
from sklearn.metrics import mean_squared_error
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
from pybrain.datasets import SequentialDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure.modules import LSTMLayer, LinearLayer, SigmoidLayer, SoftmaxLayer,TanhLayer
from pybrain.structure.networks import RecurrentNetwork
from pybrain.structure.connections import FullConnection
from pybrain.supervised import RPropMinusTrainer
from sys import stdout

from sklearn.preprocessing import MinMaxScaler

print ("Cite Reference: https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/")

data = web.get_data_yahoo('SPY', dt.datetime(2015,10,22), dt.datetime(2016,06,22))
#data = data.ix[:,5].values.tolist()

data = data[['Open','High','Low','Adj Close','Volume']].values.tolist()

# normalize the dataset

scaler =MinMaxScaler(feature_range=(0, 1)).fit(data)

data = scaler.transform(data).tolist()

#ds = SequentialDataSet(7, 1)

numDaysAhead =1
numDaysLookBack=20
numDim =5

ds = SequentialDataSet(5, 1)

for i in range(0,len(data)-(numDaysAhead+numDaysLookBack+1)):
    # training: previous 10 days and same day last year
    #sample = data[i:(i+6)] #+ [data[i-252]]
    sample = data[i:(i+numDaysLookBack)]
    # target: 5 days ahead
    #target = data[i+10]
    target =  [data[i+numDaysLookBack+numDaysAhead-1][3]] #adj close is 3rd item

    #for each row of training data for this sample (window of trading days)
    for input in sample:
           
        #need to add samples in a loop 
        ds.appendLinked(input, target)

    ds.newSequence()

ds.removeSequence(i+1)

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
net = buildNetwork(numDim,3,1 , hiddenclass=LSTMLayer,  outputbias=False, recurrent=True) # outclass=TanhLayer)

trainer = RPropMinusTrainer(net, dataset=ds)

print('training until convergence')
trainer.trainUntilConvergence(dataset=ds,validationProportion=0.25)

if False:
    train_errors = [] # save errors for plotting later
    EPOCHS_PER_CYCLE = 60
    CYCLES = 40 # was 500
    EPOCHS = EPOCHS_PER_CYCLE * CYCLES
    for i in xrange(CYCLES):
        trainer.trainEpochs(EPOCHS_PER_CYCLE)
        train_errors.append(trainer.testOnData())
        epoch = (i+1) * EPOCHS_PER_CYCLE
        print("\r epoch {}/{}".format(epoch, EPOCHS), end="")
        stdout.flush()

    print()
    print("final error =", train_errors[-1])

#target = np.zeros(ds.getSequenceLength(0))
#prediction = np.zeros(ds.getSequenceLength(0))

target = np.zeros(ds.getNumSequences())
prediction = np.zeros(ds.getNumSequences())
     
i = 0

for SeqCtr in range(ds.getNumSequences()):
    input, target =ds.getSequence(SeqCtr)
    #for s, t in ds.getSequenceIterator(SeqCtr):
    #target[i] = t
    #target[i] = scaler.inverse_transform(t[0]) #scaler.inverse_transform(t.tolist()*numDim)[3]
    target[i] =scaler.inverse_transform([x[0] for x in target[0:numDim].tolist()])[3]
    
    #prediction[i] = net.activate(s)
    prediction[i] = net.activate(input)
    #prediction[i] = scaler.inverse_transform([prediction[i]]) 
    prediction[i] = scaler.inverse_transform([prediction[i]]*numDim)[3]
    i += 1

plt.plot(target)
plt.plot(prediction)
plt.show()
# prediction error
print(target - prediction)


testScore = math.sqrt(mean_squared_error(target, prediction))
print('Test Score: %.2f RMSE' % (testScore))
