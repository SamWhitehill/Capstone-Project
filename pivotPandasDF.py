import math
from sklearn import svm
import pandas as pd
import datetime
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc,\
    volume_overlay2,volume_overlay3
from pandas.io import data as web 
import datetime as dt
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
#from sklearn.neural_network import MLPRegressor
#from neon.data import IMDB
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cPickle
import numpy as np

def get_bollinger_bands(rm, rstd):

	upper_band=rm + (2*rstd)

	lower_band=rm + (-2*rstd)

	return upper_band, lower_band

def fnComputeCandleStickPattern(pDf):

    pDf['RealBody']=np.absolute(pDf['Close'] -pDf['Open'])/pDf['Open']*1
    pDf['Color']=np.where(pDf['Close'] >pDf['Open'] ,.5,-.5) #use numbers not colors
    
    return pDf

def fnComputeFeatures(pDf):
    #compute various historical features such as rolling mean, bollinger bands
    #candlestick patterns,etc.., 
    #returns dataframe with the features
    # compute rolling mean, stdev, bollinger
    df =pDf
    
    df=fnComputeCandleStickPattern(df)

    rollingMean =pd.rolling_mean(df['Close'],window=20)
    rollingMeanFifty =pd.rolling_mean(df['Close'],window=50)

    rollingStdev =pd.rolling_std(df['Close'],window=20)

    rollingMeanFifty.fillna(value=0,inplace=True)
    rollingMean.fillna(value=0,inplace=True)
    rollingStdev.fillna(value=0,inplace=True)

    #rollingMean =rollingMean+10
    #print(rollingMean)
    #upper /lower bands are pandas series
    upper_band, lower_band =get_bollinger_bands(rollingMean,rollingStdev )


    #append additional stats into original dataframe
    #first create dataframes
    upper_band =pd.DataFrame(upper_band)
    upper_band = pd.DataFrame(upper_band).reset_index()
    upper_band.columns = ['Date', 'upper_band']
    #name column
        
    lower_band =pd.DataFrame(lower_band)
    lower_band = pd.DataFrame(lower_band).reset_index()
    lower_band.columns = ['Date', 'lower_band']

    rollingMean=pd.DataFrame(rollingMean)
    rollingMean = pd.DataFrame(rollingMean).reset_index()
    rollingMean.columns = ['Date', 'rollingMean20']

    if True:
        df =df.merge(upper_band,how='inner',on=['Date'])
        df =df.merge(lower_band,how='inner',on=['Date'])
        #df =df.merge(rollingMean,how='inner',on=['Date'])
        #TRUNCATE dataframe until point when rolling stats start, otherwise we will have
        # zero for rolling mean , stdev
        df =df[df['upper_band']>0]
        
    #compute diff between price and up/low bollingers
    df['DiffercenceBtwn_upper_band'] =df['upper_band']-df['Close']
    df['DiffercenceBtwn_lower_band'] =df['Close']-df['lower_band']

    #remove bollingers now
    del df['upper_band']
    del df['lower_band']
    

    return pDf

def fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
                                   pNumDaysLookBack):

        #sort by date asc
        df=pDataFrameStockData
        #df.Date = pd.to_datetime(df.Date)
        df.sort(['Date'], inplace=True)
        lst_Y =[]
        lStrTicker ='TICKER'
        
        #add in calculated features
        df=fnComputeFeatures(df)

        iRowCtr =0
        #dfFilter =df[df['Date']<datetime.date(year=2015,month=9,day=6)]
        lEnd =len(df)
        lEndRow =1
        lRowPredictedPrice =pNumDaysAheadPredict
        df['Ticker'] =lStrTicker 
        result =[]
        ##list explicitly used columns for features
        #note that Date and Ticker are not used BUT needed for pivoting the dataframe
        lstCols=[ 'Open','Close','High','Low','Volume','RealBody' ,'Color','Ticker','Date' ]

        while (iRowCtr+pNumDaysLookBack+pNumDaysAheadPredict)<=lEnd:
                #for iRowCtr in range(0, lEnd):
                lEndRow =iRowCtr+pNumDaysLookBack
                #p = df[df['Date']<datetime.date(year=2015,month=9,day=6)].pivot(index='Ticker', columns='Date')
                p = df[iRowCtr:lEndRow][lstCols].pivot(index='Ticker', columns='Date')

                result.append(list(p.T[lStrTicker][:]))

                lRowPredictedPrice =lEndRow+pNumDaysAheadPredict-1
                lst_Y.append(df['Adj Close'][lRowPredictedPrice:lRowPredictedPrice+1].values[0])
                iRowCtr=iRowCtr+1



        return result, lst_Y


def fnGetYahooStockData(pStartDate, pEndDate, pSymbol):
    # (Year, month, day) tuples suffice as args for quotes_historical_yahoo
    #dateStart = (pStartDate.year, pStartDate.month, pStartDate.day)
    #dateEnd = (pEndDate.year, pEndDate.month, pEndDate.day)

    dateStart =dt.datetime(pStartDate.year, pStartDate.month, pStartDate.day)
    dateEnd = dt.datetime(pEndDate.year, pEndDate.month, pEndDate.day)

    sSymbol =pSymbol

    data = web.get_data_yahoo(sSymbol, dateStart,dateEnd)
    #'Need to create new column Date from index date as there is no column 'Date', yet.
    data['Date'] = data.index
    #quotes = quotes_historical_yahoo_ohlc(sSymbol, dateStart, dateEnd)
    quotes =data
    if len(quotes) == 0:
        raise SystemExit

    #dfQuotes =pd.DataFrame(quotes,columns=['Date','Open','Adj Close','High','Low','Volume'])
    dfQuotes =quotes[['Date','Open','Close','Adj Close','High','Low','Volume']]
    return dfQuotes
#df=pd.read_csv('..//StockData.csv')


def fnMain(pLookBackDays=60, pBlnUseSavedData=False):
    blnGridSearch =False
    lTicker ="XOM"

    lNumDaysLookBack=pLookBackDays
    lNumDaysAheadPredict=10
    #save data via pickle
    if pBlnUseSavedData==False:
        #train data
        lStartDate=datetime.date(2002, 6, 6)
        lEndDate=datetime.date(2003, 12, 25)

        dfQuotes =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #test data
        lStartDate=lEndDate #datetime.date(2003, 12, 25)
        lEndDate=datetime.date(2004, 7, 1)

        dfQuotesTest =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
         #                                  pNumDaysLookBack)

        train=fnGetHistoricalStockDataForSVM(dfQuotes,lNumDaysAheadPredict , lNumDaysLookBack)

        testingData=fnGetHistoricalStockDataForSVM(dfQuotesTest,lNumDaysAheadPredict , lNumDaysLookBack)

        #cPickle.dump(train, open('train.p', 'wb')) 
        #cPickle.dump(testingData, open('testingData.p', 'wb')) 
  
    else:
         #use previously saved data
        train = cPickle.load(open('train.p', 'rb'))
        testingData = cPickle.load(open('testingData.p', 'rb'))

    #11 day lookback, 10 day ahead, scores .75, SVR(C=1100000, cache_size=200, coef0=0.0, degree=3, epsilon=0.001,
    #gamma=1e-07, kernel='rbf', max_iter=-1, shrinking=True, tol=0.001,
    #verbose=False)
    #25 look back, 5 days out scores .42
    #18 look back, 10 days out scores .38
    # clf= svm.SVR(kernel='rbf', C=99999,gamma=1e-7,    epsilon =.001) continued...
    #18 look back, 10 days out scores  .55
    #15 look back, 10 days out svm.SVR(kernel='rbf', C=99999,gamma=1e-7,    epsilon =.001), scores .65


    # fit the model and calculate its accuracy
    #{'C': 500000, 'gamma': 1e-06}
    #{'C': 1100000, 'gamma': 1e-07}
    C=1400000
    gamma=.00002
    clfReg = svm.SVR(kernel='rbf', C=C,gamma=gamma,    epsilon =.001)
    #clfReg =MLPRegressor(activation='logistic')


    X_train =train[0]
    y_train =train[1]



    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #dataset = scaler.fit_transform(dataset)
    #must implement feature scaling, or else volume will dominate
    #scaler = preprocessing.StandardScaler().fit(X_train)
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_train) #.46 score

    X_train =scaler.transform(X_train)  

    parameters={'C':[2000,10000,45000,80000,120000, 160000,225000,500000,750000,1100000,15000000,25000000,80000000],
                'gamma':[.05,.04,.03,.02,.01,.001,0001,1e-5,1e-6,1e-7,1e-8,1e-10]}
    #12.17.2016 clf.best_params_{'C': 1100000, 'gamma': 1e-07}
    clf =clfReg

    if blnGridSearch:
        clf = GridSearchCV(clfReg, parameters, verbose=1,n_jobs=3)
    #clf =rbf_svm


    clf.fit(X_train, y_train)

    if blnGridSearch:
        print(clf.best_params_)

    X_test =testingData[0]
    y_test =testingData[1]

    X_test =scaler.transform(X_test)  
    

    score = clf.score(X_test, y_test)
    prediction =clf.predict(X_test)

    #reverse the scaler to get back original prices
    
    #prediction=scaler.inverse_transform(prediction)

    testScore = math.sqrt(mean_squared_error(y_test, prediction))
    print('Test Score: %.2f RMSE' % (testScore))

    print('SVM Score: ' +str(score))

        
    plt.plot(y_test,label='Actual ' + lTicker)
    plt.plot(prediction,label='Predicted')
    #lNumDaysLookBack=30
    #lNumDaysAheadPredict=5
    plt.suptitle(lTicker + ' SVR: ' + str(C) + ' gamma ' +str(gamma) + ' lookback: '+str(lNumDaysLookBack) +
                 ' daysAhead: ' + str(lNumDaysAheadPredict) + ' SVM Score: '+ str(score),
                fontsize=14, fontweight='bold')
    
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')
    
    plt.show()
    #print(test) result=  rbf_svm.predict(X_test)

    #?result[10:20]
    #y_test[10:20]
    #?rbf_svm.fit(X_train, y_train).score(X_test,y_test)


if __name__=='__main__':
        for i in range(40,100,2):
                fnMain(i,False) #25 look back was best
                print (i)


    #?clf.best_params_
#{'C': 40000, 'gamma': 1e-06}
