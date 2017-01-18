from __future__ import print_function
#Modified from https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/

#import pandas_datareader.data as web
from MainForecastModule import fnGetNaturalLogPrices
from MainForecastModule import fnGetHistoricalStockDataForSVM
import math
from pandas_datareader import data as web
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
from pybrain.tools.validation    import testOnSequenceData
from sys import stdout
import cPickle

from sklearn.preprocessing import MinMaxScaler

print ("Cite Reference: https://stavrossioutis.wordpress.com/2016/06/25/long-short-term-memory-rnns-for-stock-price-forecasting/")


lstrPath="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"
#data = web.get_data_yahoo('SPY', dt.datetime(2016,1,3), dt.datetime(2016,06,22))
data = cPickle.load(open(lstrPath+'dfQuotes.p', 'rb'))
numDaysAhead =45       
numDaysLookBack=7
 
data['Adj Close Price']=data['Adj Close']
#data = data.ix[:,5].values.tolist()
data=fnGetNaturalLogPrices(data,numDaysAhead)

XTrain=None
lst_Y=None
lstPreviousDayPrices=None
df=None
XTrain, lst_Y,  lstPreviousDayPrices,df= fnGetHistoricalStockDataForSVM(data,numDaysAhead,numDaysLookBack,45,1,1)


#data = data[['Open','High','Low','Adj Close']].values.tolist()

# normalize the dataset
if False:
    scaler =MinMaxScaler(feature_range=(0, 1)).fit(data)

    data = scaler.transform(data).tolist()

#ds = SequentialDataSet(7, 1)



numDim =13

ds = SequentialDataSet(numDim*numDaysLookBack, 1) # SupervisedDataSet.__init__(self, indim, targetdim)
#ds=SupervisedDataSet( 3, 2 )

#def activateOnDatasetSAM(dataset):
#    """Run the module's forward pass on the given dataset unconditionally
#    and return the output."""        
#    dataset.reset()
#    self.reset()
#    out = zeros((len(dataset), self.outdim))
#    for i, sample in enumerate(dataset):
#        # FIXME: Can we always assume that sample[0] is the input data?
#        out[i, :] = self.activate(sample[0])
#    self.reset()
#    dataset.reset()
#    return out
if False:

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


#for inp, targ in samples:
#    DS.appendLinked(inp, targ)
## or alternatively, with  ia  and  ta  being arrays:
#assert(ia.shape[0] == ta.shape[0])
ds.setField('input', XTrain)

lst_Y=np.expand_dims(lst_Y, axis=1) 
ds.setField('target',lst_Y )

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
net = buildNetwork(numDim*numDaysLookBack,3,1 , hiddenclass=LSTMLayer,  outputbias=False, recurrent=True) # outclass=TanhLayer)


trainer = RPropMinusTrainer(net, dataset=ds)

print('training until convergence')
#trainer.trainUntilConvergence(dataset=ds) #,validationProportion=0.15)

if True:
    train_errors = [] # save errors for plotting later
    EPOCHS_PER_CYCLE = 5
    CYCLES = 20 # was 500
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

#target = np.zeros(ds.getNumSequences())
#prediction = np.zeros(ds.getNumSequences())
     
target = np.zeros(len(XTrain))
prediction =np.zeros(len(XTrain))
i = 0

#net.offset =0
#for SeqCtr in range(ds.getNumSequences()):
    #input, output =ds.getSequence(SeqCtr)
    #for s, t in ds.getSequenceIterator(SeqCtr):
    #target[i] = t
    #target[i] = scaler.inverse_transform(t[0]) #scaler.inverse_transform(t.tolist()*numDim)[3]
    #target[i] =scaler.inverse_transform([x[0] for x in output[0:numDim].tolist()])[3]
    #target[i]=output
    #prediction[i] = net.activate(input)
    #prediction[i] = net.activateOnDataset(input)
    
    #prediction[i] = scaler.inverse_transform([prediction[i]]) 
    #prediction[i] = scaler.inverse_transform([prediction[i]]*numDim)[3]
    #i += 1

#input and output are entire arrays
input, output =ds.getSequence(0)
for i in range(len(input)):
    target[i]=output[i]
    prediction[i] = net.activate(input[i])


plt.plot(target)
plt.plot(prediction)
plt.show()
# prediction error
print(target - prediction)


testScore = math.sqrt(mean_squared_error(target, prediction))
print('Test Score: %.2f RMSE' % (testScore))
