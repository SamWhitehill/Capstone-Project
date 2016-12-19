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
from sklearn.preprocessing import LabelEncoder  
import numpy as np
from textwrap import wrap
from scipy.stats import linregress


global lstCols
lstCols=[]
def get_bollinger_bands(rm, rstd):

	upper_band=rm + (2*rstd)

	lower_band=rm + (-2*rstd)

	return upper_band, lower_band

def fnConvertSeriesToDf(pdSeries, pLstColumns):
    #returns dataframe
    df = pd.DataFrame(pdSeries).reset_index()
    df.columns = pLstColumns
    return df

def fnCalcNDayNetPriceChange(pDf,N=2):
        #return dataframe pDf with N day net Adj closeprice change where N is numbrer days back
        pDf[str(N)+'DayNetPriceChange'] =pDf['Adj Close'] -pDf['Adj Close'].shift(N)
        return pDf

def fnCalcAvgVolumeStats(pDf,N=10):
        rollingAvgVol=pd.rolling_mean(pDf['Volume'],window=N)

        rollingAvgVol=fnConvertSeriesToDf(rollingAvgVol,['Date', 'AvgVolume'])

        pDf =pDf.merge(rollingAvgVol,how='left',on=['Date']).set_index(['Date'], drop=False)

        pDf['DiffercenceBtwnAvgVol'] =pDf['Volume']-pDf['AvgVolume']

        #compute whether stock went up on above avg volume or down on above avg

        #up on above avg
        pDf['UpDownVolumeChange']=''
        pDf['UpDownVolumeChange'] =np.where( np.logical_and(pDf['Adj Close'] -pDf['Adj Close'].shift(1)>=0,pDf['DiffercenceBtwnAvgVol']>0),
                                             'UpOnAboveAvg',pDf['UpDownVolumeChange'])
        #down on above avg
        pDf['UpDownVolumeChange'] =np.where( np.logical_and(pDf['Adj Close'] -pDf['Adj Close'].shift(1)<0,pDf['DiffercenceBtwnAvgVol']>0),
                                             'DownOnAboveAvg',pDf['UpDownVolumeChange'])

        #down on below avg
        pDf['UpDownVolumeChange'] =np.where( np.logical_and(pDf['Adj Close'] -pDf['Adj Close'].shift(1)<0,pDf['DiffercenceBtwnAvgVol']<0),
                                             'DownOnBelowAvg',pDf['UpDownVolumeChange'])
        #up on below avg
        pDf['UpDownVolumeChange'] =np.where( np.logical_and(pDf['Adj Close'] -pDf['Adj Close'].shift(1)>=0,pDf['DiffercenceBtwnAvgVol']<0),
                                             'UpOnBelowAvg',pDf['UpDownVolumeChange'])

        le = LabelEncoder()
        le.fit(  pDf['UpDownVolumeChange'])
        #encode publish time as it may not be able to fit in classifier otherwise.
        pDf['UpDownVolumeChange']=le.transform(pDf['UpDownVolumeChange'])
        
        return pDf

def fnWraplinregress(pValues):
      iLen =len(pValues)
      lXVals =range(iLen)
      slope_0, intercept, r_value, p_value, std_err=linregress(lXVals, pValues)
      return slope_0


def fnCalculateSlope(pDf,N=10):
     #calculate slope on a rolling basis    
     #slope_0, intercept, r_value, p_value, std_err =\
     #stats.linregress(pDf['Date'], pDf['Adj Close'])
     rollingSlopeLow =pd.rolling_apply(pDf[['Low']],N,fnWraplinregress)
     rollingSlopeLow=fnConvertSeriesToDf(rollingSlopeLow,['Date', 'LowSlope'])

     rollingSlopeHi =pd.rolling_apply(pDf[['High']],N,fnWraplinregress)
     rollingSlopeHi=fnConvertSeriesToDf(rollingSlopeHi,['Date', 'HighSlope'])

     pDf =pDf.merge(rollingSlopeLow,how='left',on=['Date']).set_index(['Date'], drop=False)
     pDf =pDf.merge(rollingSlopeHi  ,how='left',on=['Date']).set_index(['Date'], drop=False)
     #need a 
     #x.groupby('entity').apply(lambda v: linregress(v.year, v.value)[0])
     #[0] means slope only
     
     return pDf

def fnComputeCandleStickPattern(pDf):
   #If the high and low of a bar is higher than previous bar, then that bar is
   # called an 'up bar' or an 'up day'. If the high and low of a bar is lower
   # than previous bar, then that bar is called an 'down bar' or an 'down day'
   #sourc: http://www.stock-trading-infocentre.com/bar-charts.html

        pDf['Color']=np.where(pDf['Close'] >pDf['Open'] ,'WHITE','BLACK') #use numbers not colors
        
        pDf['RealBody']=np.absolute(pDf['Close'] -pDf['Open'])/pDf['Open']*1

        #upper shadow
        pDf['UpperShadow']=np.where(pDf['Close'] >pDf['Open'], (pDf['Close'] -pDf['Open'])/(pDf['High']- pDf['Open']),
        (pDf['Open'] -pDf['Close'])/(pDf['High']- pDf['Close']))

        pDf['UpperShadow']=pDf['UpperShadow']*1
        pDf['UpperShadow'].fillna(value=0,inplace=True)
        #lower shadow
        pDf['LowerShadow']=np.where(pDf['Close'] >pDf['Open'], (pDf['Close'] -pDf['Open'])/(pDf['Close']- pDf['Low']),
        (pDf['Open'] -pDf['Close'])/(pDf['Open']- pDf['Low']))

        pDf['LowerShadow']=pDf['LowerShadow']*1
        pDf['LowerShadow'].fillna(value=0,inplace=True)
        #'up bar
        pDf['BarType']=np.where(np.logical_and(pDf['High']>=pDf['High'].shift(1), pDf['Low']>=pDf['Low'].shift(1)),
                                'Up','')
        
        #down bar
        pDf['BarType']=np.where(np.logical_and(pDf['High']<=pDf['High'].shift(1), pDf['Low']<=pDf['Low'].shift(1)),
                                'Down',pDf['BarType'])

        #inside bar
        pDf['BarType']=np.where(np.logical_and(pDf['High']<=pDf['High'].shift(1), pDf['Low']>=pDf['Low'].shift(1)),
                                'Inside',pDf['BarType'])

        pDf['BarType']=np.where(np.logical_and(pDf['High']>pDf['High'].shift(1), pDf['Low']< pDf['Low'].shift(1)),
                                'Outside',pDf['BarType'])

        

        le = LabelEncoder()
        le.fit(  pDf['Color'])
        #encode publish time as it may not be able to fit in classifier otherwise.
        pDf['Color']=le.transform(pDf['Color'])

        le.fit(pDf['BarType'])
        pDf['BarType'] =le.transform(pDf['BarType'])

        return pDf

def fnComputeFeatures(pDf,pNumDaysLookBack):
        #compute various historical features such as rolling mean, bollinger bands
        #candlestick patterns,etc.., 
        #returns dataframe with the features
        # compute rolling mean, stdev, bollinger

        pDf=fnComputeCandleStickPattern(pDf)

        pDf =fnCalcNDayNetPriceChange(pDf,2)

        pDf=fnCalcAvgVolumeStats(pDf,12)

        pDf =fnCalculateSlope(pDf,pNumDaysLookBack)

        rollingMean =pd.rolling_mean(pDf['Adj Close'],window=10)
        rollingMeanFifty =pd.rolling_mean(pDf['Adj Close'],window=50)

        rollingStdev =pd.rolling_std(pDf['Adj Close'],window=10) # CHANGED TO 10 DAY FROM research paper

        rollingMeanFifty.fillna(value=0,inplace=True)
        rollingMean.fillna(value=0,inplace=True)
        rollingStdev.fillna(value=0,inplace=True)
        #df =df.merge(rollingMean,how='inner',on=['Date'])

        #rollingMean =rollingMean+10
        #print(rollingMean)
        #upper /lower bands are pandas series
        upper_band, lower_band =get_bollinger_bands(rollingMean,rollingStdev )

        

        #append additional stats into original dataframe
        #first create dataframes
        #name column
        upper_band=fnConvertSeriesToDf(upper_band,['Date', 'upper_band'])

        lower_band=fnConvertSeriesToDf(lower_band,['Date', 'lower_band'])
        rollingStdev=fnConvertSeriesToDf(rollingStdev,['Date', 'rollingStdev20'])
        rollingMean=fnConvertSeriesToDf(rollingMean,['Date', 'rollingMean20'])

        rollingMeanFifty=fnConvertSeriesToDf(rollingMeanFifty,['Date', 'rollingMean50'])

        pDf =pDf.merge(rollingMean,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingStdev,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingMeanFifty,how='inner',on=['Date'],right_index=True)

        pDf =pDf[pDf['rollingMean50']>0]

        if True:
                pDf =pDf.merge(upper_band,how='inner',on=['Date'],right_index=True)
                pDf =pDf.merge(lower_band,how='inner',on=['Date'],right_index=True)
                #df =df.merge(rollingMean,how='inner',on=['Date'])
                #TRUNCATE dataframe until point when rolling stats start, otherwise we will have
                # zero for rolling mean , stdev
                pDf =pDf[pDf['upper_band']>0]

                #compute diff between price and up/low bollingers
                pDf['DiffercenceBtwn_upper_band'] =pDf['upper_band']-pDf['Close']
                pDf['DiffercenceBtwn_lower_band'] =pDf['Close']-pDf['lower_band']

                #remove bollingers now
                #del pDf['upper_band']
                #del pDf['lower_band']
    

        return pDf

def fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
                                   pNumDaysLookBack):
        global lstCols
        #sort by date asc
        df=pDataFrameStockData
        #df.Date = pd.to_datetime(df.Date)
        df.sort(['Date'], inplace=True)
        lst_Y =[]
        lStrTicker ='TICKER'
        
        #add in calculated features
        df=fnComputeFeatures(df,pNumDaysLookBack)

        iRowCtr =0
        #dfFilter =df[df['Date']<datetime.date(year=2015,month=9,day=6)]
        lEnd =len(df)
        lEndRow =1
        lRowPredictedPrice =pNumDaysAheadPredict
        df['Ticker'] =lStrTicker 
        result =[]
        ##list explicitly used columns for features
        #note that Date and Ticker are not used BUT needed for pivoting the dataframe
        lstCols=[ 'Open','Close','High','Low','Volume','RealBody' ,'Ticker','Date',
                'BarType','Color','UpperShadow','LowerShadow','rollingMean50','rollingMean20','rollingStdev20' ]

        lstCols=['DiffercenceBtwnAvgVol','AvgVolume','2DayNetPriceChange','Volume','Ticker','Date','Adj Close', #'BarType' ,'Color',
                'rollingMean50','rollingMean20','rollingStdev20','Open','High','Low','UpDownVolumeChange']
                 #'UpDownVolumeChange'
                  #      ] # 'Open','High','Low', ,'upper_band','lower_band'
        
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
    global lstCols
    lTicker ="COP" #SBUX
        
    lNumDaysLookBack=pLookBackDays
    lNumDaysAheadPredict=10
    #save data via pickle
    if pBlnUseSavedData==False:
        #train data
        lStartDate=datetime.date(2001, 1, 6)
        lEndDate=datetime.date(2004, 7, 1)

        dfQuotes =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #test data
        lStartDate=lEndDate #datetime.date(2003, 12, 25)
        lEndDate=datetime.date(2005, 7, 1)

        dfQuotesTest =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
         #                                  pNumDaysLookBack)

        train=fnGetHistoricalStockDataForSVM(dfQuotes,lNumDaysAheadPredict , lNumDaysLookBack)

        testingData=fnGetHistoricalStockDataForSVM(dfQuotesTest,lNumDaysAheadPredict , lNumDaysLookBack)

        cPickle.dump(train, open('train.p', 'wb')) 
        cPickle.dump(testingData, open('testingData.p', 'wb')) 
  
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
    C=1100000 #SVM Score: -11.1354870245 #'C': 1100000, 'gamma': 1e-06}
    gamma=1e-06

    #{'C': 2000, 'gamma': 1e-05}
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

    parameters={'C':[500,1000,2000,10000,45000,80000,120000, 160000,225000,500000,750000,1100000,2500000,15000000],
                'gamma':[.01,.001,0001,.0004,.0007,1e-5,1e-6,1e-7,1e-8]}
    #12.17.2016 clf.best_params_{'C': 1100000, 'gamma': 1e-07}
    clf =clfReg

    if blnGridSearch:
        clf = GridSearchCV(clfReg, parameters, verbose=1,n_jobs=3)
    #clf =rbf_svm


    clf.fit(X_train, y_train)

    if blnGridSearch:
        print(clf.best_params_)  
        print('lookback: ' +str(lNumDaysLookBack))
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
    if 'lstCols' in globals()==False:
            lstCols=[]
    #lNumDaysAheadPredict=5
    lstrTitle ="\n".join(wrap(lTicker + ' SVR C: ' + str(C) + ' gamma ' +str(gamma) + ' lookback: '+str(lNumDaysLookBack) +
                 ' daysAhead: ' + str(lNumDaysAheadPredict) + ' SVM Score: '+ str(score) +' features: ' +str(lstCols)))
                              
    plt.suptitle(lstrTitle,
                fontsize=11, fontweight='bold')
    
    legend = plt.legend(loc='upper center', shadow=True, fontsize='x-large')

    if True:
            plt.show()
    #print(test) result=  rbf_svm.predict(X_test)

    #?result[10:20]
    #y_test[10:20]
    #?rbf_svm.fit(X_train, y_train).score(X_test,y_test)


if __name__=='__main__':
        for i in range(22,23):
                fnMain(i,False) #25 look back was best
                print (i)


    #?clf.best_params_
#{'C': 40000, 'gamma': 1e-06}
