import  BuildTrendLines as trendy
from sklearn.model_selection import TimeSeriesSplit
from SupportAndResistance  import fnGetSupportResistance
import math
from sklearn import svm
import pandas as pd
import datetime
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc,\
    volume_overlay2,volume_overlay3
from pandas_datareader import data as web
import datetime as dt
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPRegressor
#from neon.data import IMDB
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import cPickle
from sklearn.preprocessing import LabelEncoder  
import numpy as np
from textwrap import wrap
from scipy.stats import linregress
import matplotlib.dates as mdates
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import differential_evolution

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

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

def fnCalculateNewMACD():
    #source -https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code
    #https://github.com/panpanpandas/ultrafinance/blob/master/ultrafinance/pyTaLib/pandasImpl.py
    pass


def fnCalculateLogReturn(pDf,N=1):
    return np.log(pDf['Close']).diff()

def fnCalculateEMV(data, nDays):
    #Ease of movement
    #source: https://www.quantinsti.com/blog/build-technical-indicators-in-python/
     dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
     br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
     EVM = dm / br 
     EVM_MA = pd.Series(pd.rolling_mean(EVM, nDays), name = 'EMV') 
     data = data.join(EVM_MA) 
     return data

# Force Index 
def fnCalcForceIndex(data, ndays):
        
     FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
     data = data.join(FI) 
     return data


def fnCalcRSI(price, n=14):
    ''' rsi indicator '''
    gain = (price-price.shift(1)).fillna(0) # calculate price gain with previous day, first row nan is filled with 0

    def rsiCalc(p):
        # subfunction for calculating rsi for one lookback period
        avgGain = p[p>0].sum()/n
        avgLoss = -p[p<0].sum()/n 
        rs = avgGain/avgLoss
        return 100 - 100/(1+rs)

    # run for all periods with rolling_apply
    return pd.rolling_apply(gain,n,rsiCalc) 


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
      pValues=pValues/1000000
      slope_0, intercept, r_value, p_value, std_err=linregress(lXVals, pValues)
      return slope_0

def moving_average_convergence(pDf, nslow=26, nfast=12): #26,12
    pDf['emaslow'] = pd.ewma(pDf["Close"], span=nslow, freq="D",min_periods=nslow)
    pDf['emafast'] = pd.ewma(pDf["Close"], span=nfast, freq="D",min_periods=nfast)

    #emaslow = pd.ewma(group, span=nslow, min_periods=1)
    #emafast = pd.ewma(group, span=nfast, min_periods=1)
    #result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    pDf['MACD']=pDf['emafast'] - pDf['emaslow']

    return pDf

def fnCalculateSlope(pDf,N=10):
        #calculate slope on a rolling basis    
        #slope_0, intercept, r_value, p_value, std_err =\
        #stats.linregress(pDf['Date'], pDf['Adj Close'])
        #rollingSlopeLow =pd.rolling_apply(pDf[['Low']],N,fnWraplinregress)
        #rollingSlopeLow=fnConvertSeriesToDf(rollingSlopeLow,['Date', 'LowSlope'])

        rollingSlopeClose =pd.rolling_apply(pDf[['Close']],N,fnWraplinregress)
        rollingSlopeClose=fnConvertSeriesToDf(rollingSlopeClose,['Date', 'CloseSlope'])

        rollingSlopeVol=pd.rolling_apply(pDf[['Volume']],N,fnWraplinregress)
        rollingSlopeVol=fnConvertSeriesToDf(rollingSlopeVol,['Date', 'VolumeSlope'])


        #rollingSlopeHi =pd.rolling_apply(pDf[['High']],N,fnWraplinregress)
        #rollingSlopeHi=fnConvertSeriesToDf(rollingSlopeHi,['Date', 'HighSlope'])

        #pDf =pDf.merge(rollingSlopeLow,how='left',on=['Date']).set_index(['Date'], drop=False)
        #pDf =pDf.merge(rollingSlopeHi  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        pDf =pDf.merge(rollingSlopeVol  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        pDf =pDf.merge(rollingSlopeClose  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        
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

def fnComputeFeatures(pDf,pNumDaysLookBack,pSlopeLookback, pDaysAhead=8,pSRLookback=11,pSegments=4):
        #compute various historical features such as rolling mean, bollinger bands
        #candlestick patterns,etc.., 
        #returns dataframe with the features
        # compute rolling mean, stdev, bollinger
        lstCols=None
        
        #dfSR, lstCols =fnGetSupportResistance(pDf,pSRLookback,pDaysAhead,pSegments)

        #pDf =pDf.merge(dfSR,how='inner',on=['Date'],right_index=True)
        #not using candlestick patterns any longer
        #pDf=fnComputeCandleStickPattern(pDf)

        pDf =fnCalcNDayNetPriceChange(pDf,2)

        pDf=fnCalcAvgVolumeStats(pDf,20)

        pDf =fnCalculateSlope(pDf,pSlopeLookback) #pNumDaysLookBack try 32

        pDf=moving_average_convergence(pDf)

        pDf['RSI'] =fnCalcRSI(pDf['Close'])

        pDf=fnCalculateEMV(pDf, pNumDaysLookBack)

        pDf=fnCalcForceIndex(pDf,pNumDaysLookBack)

        #compute log return
        #pDf['DailyLogReturn'] =fnCalculateLogReturn(pDf,1)

        #RSI decision fields, rsi >= 70 overbought, rsi <= 30 oversold
        pDf['RSIDecision'] =np.where(pDf['RSI']  >=70,10*pDf['Adj Close'],0)
        pDf['RSIDecision'] =np.where(pDf['RSI'] <=30,-10*pDf['Adj Close'],pDf['RSIDecision'])


        rollingMean =pd.rolling_mean(pDf['Adj Close'],window=20)
        rollingMeanFifty =pd.rolling_mean(pDf['Adj Close'],window=50)

        #26 day std dev gives best results
        rollingStdev =pd.rolling_std(pDf['Adj Close'],window=10) # CHANGED TO 10 DAY FROM research paper

        rollingMeanFifty.fillna(value=0,inplace=True)
        rollingMean.fillna(value=0,inplace=True)
        rollingStdev.fillna(value=0,inplace=True)
        #df =df.merge(rollingMean,how='inner',on=['Date'])

        #rollingMean =rollingMean+10
        #print(rollingMean)
        #upper /lower bands are pandas series
        if True:
            upper_band, lower_band =get_bollinger_bands(rollingMean,rollingStdev )

        #append additional stats into original dataframe
        #first create dataframes
        #name column
        if True:
            upper_band=fnConvertSeriesToDf(upper_band,['Date', 'upper_band'])

            lower_band=fnConvertSeriesToDf(lower_band,['Date', 'lower_band'])

        rollingStdev=fnConvertSeriesToDf(rollingStdev,['Date', 'rollingStdev20'])
        rollingMean=fnConvertSeriesToDf(rollingMean,['Date', 'rollingMean20'])

        rollingMeanFifty=fnConvertSeriesToDf(rollingMeanFifty,['Date', 'rollingMean50'])

        pDf =pDf.merge(rollingMean,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingStdev,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingMeanFifty,how='inner',on=['Date'],right_index=True)

        #CANNOT HAVE any Nans, must all be valid number so cut off records with invalid Nans.
        if 'VolumeSlope' in pDf.columns:
            pDf =pDf[ pd.notnull(pDf['VolumeSlope'])]
        pDf =pDf[ pd.notnull(pDf['EMV'])]
        pDf =pDf[ pd.notnull(pDf['MACD'])]
        pDf =pDf[ pd.notnull(pDf['RSI'])]
        
        #ensure no NULL support /resistance levels, remove first n records to NOT cheat!
        lstSR =lstCols #['S1','S2','S3','S4','R1','R2','R3','R4']
        #any 1 of these SR items could be null
        if lstSR !=None:
            for item in lstSR:
                #fill blanks support with 0, blank resist with large num
                if item[0]=='S' or item[0]=='Min':
                    pDf[item].fillna(value=0,inplace=True)
                else:
                    pDf[item].fillna(value=10e4,inplace=True)
                #pDf =pDf[ pd.notnull(pDf[item])]


        if False:
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
    

        return pDf, lstCols

def fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
                                   pNumDaysLookBack,pSlopeLookback,pSRLookback,
                                   pSegments):
        global lstCols
        #sort by date asc
        df=pDataFrameStockData
        df.Date = pd.to_datetime(df.Date)
        df.sort(['Date'], inplace=True)
        lst_Y =[]
        lStrTicker ='TICKER'
        lstPreviousDayPrices =[]
        lstDates=[]
        #USING returns now, must start at first valid return
        #this shifts up the data set
        df =df[ pd.notnull(df['Adj Close'])]

        #add in calculated features
        #calculated features are NOT adding any value to prediction via SVR, or MLP ??
        if True:
                df, lstSRCols=fnComputeFeatures(df,pNumDaysLookBack,pSlopeLookback,
                                                pNumDaysAheadPredict,pSRLookback,pSegments)

        iRowCtr =0
        #dfFilter =df[df['Date']<datetime.date(year=2015,month=9,day=6)]
        lEnd =len(df)
        lEndRow =1
        lRowPredictedPrice =pNumDaysAheadPredict
        df['Ticker'] =lStrTicker 
        result =[]
        ##list explicitly used columns for features
        #note that Date and Ticker are not used BUT needed for pivoting the dataframe
        #lstCols=[ 'Open','Close','High','Low','Volume','RealBody' ,'Ticker','Date',
        #        'BarType','Color','UpperShadow','LowerShadow','rollingMean50','rollingMean20','rollingStdev20' ]

        #lstCols=['DiffercenceBtwnAvgVol','2DayNetPriceChange','Ticker','Date',  #'Adj Close',\
        #        'rollingStdev20','Adj Close','Open','High','Low','Volume','upper_band','lower_band',
        #        'MACD','RSI','RSIDecision','RealBody','Color' #realbody and color actually hurt the R^2 on SPY.
        #         ]'DailyLogReturn',CloseSlope 'Ticker','Date',
        lstCols=['2DayNetPriceChange', 'Adj Close','rollingMean20',# DiffercenceBtwnAvgVol
                'rollingStdev20', #'High','Low','Open', #'Volume','VolumeSlope', # 'DiffercenceBtwnAvgVol',
                 #'EMV','ForceIndex', #'CloseSlope',
                 'RSI'] # +lstSRCols # 'MACD']+lstSRCols # ['S1','S2','S3','S4','R1','R2','R3','R4']
                 #['Min1','Min2','Min3','Min4','Min5','Max1','Max2','Max3','Max4','Max5'] #lstSRCols #,'RSIDecision',
        #lstCols=['Ticker','Date', 'Adj Close']       
        #'LowSlope','HighSlope'] #,'LowSlope'] rollingMean50,rollingStdev20
                #,'LowSlope', 'HighSlope'
                # ]
                 #'UpDownVolumeChange'
                  #      ] # 'Open','High','Low', ,'upper_band','lower_band'
        print ('Features - ' +str(lstCols))
        lRowPrevDayPrice =0


        #new method for pivoting
        #**************************************
        lEndIndex =(len(df)-pNumDaysAheadPredict)
        X,y = organize_data(df['Adj Close'].values, pNumDaysLookBack, pNumDaysAheadPredict)
        X = window_stack(df[lstCols][0:lEndIndex].values, stepsize=1, width=pNumDaysLookBack   ) 
        lst_Y=y
        result =X

        while (iRowCtr+pNumDaysLookBack+pNumDaysAheadPredict)<=lEnd:
                #for iRowCtr in range(0, lEnd):
                lEndRow =iRowCtr+pNumDaysLookBack
                #p = df[df['Date']<datetime.date(year=2015,month=9,day=6)].pivot(index='Ticker', columns='Date')
                #p = df[iRowCtr:lEndRow][lstCols].pivot(index='Ticker', columns='Date')

                #result.append(list(p.T[lStrTicker][:]))

                lRowPredictedPrice =lEndRow+pNumDaysAheadPredict-1
                #using log returns now, adj close is just field name.
                #lst_Y.append(df['Adj Close'][lRowPredictedPrice:lRowPredictedPrice+1].values[0])
                
                lRowPrevDayPrice =lRowPredictedPrice -pNumDaysAheadPredict
                #pNumDaysAheadPredict=lEndRow-
                #use this list to get back original price from predicted prices
                lstPreviousDayPrices.append(df['Adj Close Price'][lRowPrevDayPrice:lRowPrevDayPrice+1].values[0])
                
                #lstDates.append(df['Date'][lRowPredictedPrice:lRowPredictedPrice+1].values[0])

                iRowCtr=iRowCtr+1



        #need to process the outputs as numpy arrays so sklearn handles them correctly 
        #FINAL_X =np.array(  [  np.array(item) for item in result])
        #FINAL_Y =np.array( [item for item in lst_Y] )

        return result, lst_Y,  lstPreviousDayPrices, df


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

def fnMainWrapper(*pArgs):
        pNumLayers=pArgs[0][0]
        pNeurons=pArgs[0][1]
        pNumLayers=int(pNumLayers)
        pNeurons=int(pNeurons)
        print ("layers, neurons " ,pNumLayers,pNeurons)
        result=fnMain(8,True,pNumLayers,pNeurons)
        return result

def fnMainWrapperSVRPoly(*pArgs):
        pC=int(pArgs[0][0])
        pGamma=pArgs[0][1]
        pDegrees=int(pArgs[0][2])

        
       
        print ("C, gamma, degreees " ,pC, pGamma, pDegrees)
        result=fnMain(8,True,pC, pGamma, pDegrees)
        return result

def fnMainWrapperSVRRBF(*pArgs):
        pC=int(pArgs[0][0])
        if pC<=0:
            pC=.1
        pGamma=pArgs[0][1]
        pSlopeLookback=int(pArgs[0][2])
        pLookback=int(pArgs[0][3])
        pDaysAhead=int(pArgs[0][4])
        pSRLookback=1 #int(pArgs[0][5])
        pSegments=1#int(pArgs[0][6])

        lBlnSavedData = True
        print ("Using Saved Data =" + str(   lBlnSavedData))    
        print ("pC, pGamma,pSlopeLookback,pLookback,pDaysAhead,pSRLookback,pSegments" ,
               pC, pGamma,pSlopeLookback,pLookback,pDaysAhead,pSRLookback,pSegments)
        result=fnMain(pLookback,lBlnSavedData,pC, pGamma, 1,pSlopeLookback,pDaysAhead,pSRLookback,pSegments)
        return result

def fnGetNaturalLogPrices(pDf,Nperiods=1):
    #return the data frame with natural log of open, hi, low , close prices
    #this should make it better for time series analysis
    lstFlds =['Adj Close','Close','Open','High','Low']
    for fld in lstFlds:
        #pDf[fld] =np.log( pDf[fld] )
        pDf[fld] = np.log(pDf[fld]/pDf[fld].shift(periods=Nperiods))

    return pDf

def window_stack(a, stepsize=1, width=3):
        n = a.shape[0]
        return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def organize_data(to_forecast, window, horizon):
        """
         Input:
          to_forecast, univariate time series organized as numpy array
          window, number of items to use in the forecast window
          horizon, horizon of the forecast
         Output:
          X, a matrix where each row contains a forecast window
          y, the target values for each row of X
        """
    
        shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
        strides = to_forecast.strides + (to_forecast.strides[-1],)
        X = np.lib.stride_tricks.as_strided(to_forecast, 
                                            shape=shape, 
                                            strides=strides)
        y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
        return X[:-horizon], y

def fnGetCVIndices(pDf):
    #return the CV param for use in GridSearchCV
    #this is needed as time series cannot have random shuffling
    tscv = TimeSeriesSplit(n_splits=3)

    #reducing size as this can cause index overflow error
    maxLen =min(len(pDf),550)
    pDf=pDf[0:maxLen]

    #cv requires identical number o
    groups = pDf.groupby(pDf['Date'].dt.year).groups
    # {2012: [0, 1], 2013: [2], 2014: [3], 2015: [4, 5]}
    sorted_groups = [value for (key, value) in sorted(groups.items())] 
    # [[0, 1], [2], [3], [4, 5]]

    #cv = [(sorted_groups[i] + sorted_groups[i+1], sorted_groups[i+2])
    #      for i in range(len(sorted_groups)-2)]
    #This looks like this:
    tup1=( [sorted_groups[0]] ,[sorted_groups[2]] )
    tup2=( [sorted_groups[1]] ,[sorted_groups[2]] )
    tup3=( [sorted_groups[0]] ,[sorted_groups[1]] )
    #tup2=( [sorted_groups[0],sorted_groups[1],[sorted_groups[2]] )
    cv =[tup1,tup2,tup3]
    return cv
    #[([0, 1, 2], [3]),  # idxs of first split as (train, test) tuple
    # ([2, 3], [4, 5])]  # idxs of second split as (train, test) tuple

#def fnMain(pLookBackDays=8, pBlnUseSavedData=True,pNumLayers=1, pNeurons=1):
def fnMain(pLookBackDays=8, pBlnUseSavedData=True,pC=1, pGamma=1, pDegrees=1, 
           pSlopeLookback=10,pDaysAhead=10,pSRLookback=11,pSegments=4):
    blnGridSearch =False
    global lstCols
    lTicker ="SPY" #SBUX
    lRandomState =89
    lBlnUseNaturalLog =True

    lNumDaysLookBack=pLookBackDays
    lNumDaysAheadPredict=pDaysAhead
    lstrPath="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"

    if pBlnUseSavedData==False:
        #GET DATA FROM WEB
        #train data
        lStartDate=datetime.date(2008, 9, 6)
        lEndDate=datetime.date(2012, 12, 1)

        dfQuotes =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #test data
        lStartDate=lEndDate #datetime.date(2003, 12, 25)
        lEndDate=datetime.date(2016, 12, 30)

        dfQuotesTest =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        cPickle.dump(dfQuotes, open(lstrPath+'dfQuotes.p', 'wb')) 
        cPickle.dump(dfQuotesTest, open(lstrPath+'dfQuotesTest.p', 'wb')) 
        
    else:
        #GET DATA FROM PICKLE/SAVED FILES
        #use previously saved data
        
        dfQuotes = cPickle.load(open(lstrPath+'dfQuotes.p', 'rb'))
        dfQuotesTest = cPickle.load(open(lstrPath+'dfQuotesTest.p', 'rb'))

            
    #save data via pickle
    if True: #pBlnUseSavedData==False:
        #train data
        #lStartDate=datetime.date(2001, 1, 6)
        #lEndDate=datetime.date(2004, 7, 1)

        #add a column for original ADJ close so we can go back to prices
        #after running SVM
        dfQuotes['Adj Close Price'] =dfQuotes['Adj Close']
        dfQuotesTest['Adj Close Price'] =dfQuotesTest['Adj Close']

        ############################################
        #use natural log of prices to stabilize variance
        dfQuotes =fnGetNaturalLogPrices(dfQuotes,pDaysAhead)
        dfQuotesTest =fnGetNaturalLogPrices(dfQuotesTest,pDaysAhead)


        #dfQuotes =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #test data
        #lStartDate=lEndDate #datetime.date(2003, 12, 25)
        #lEndDate=datetime.date(2005, 7, 1)

        #dfQuotesTest =fnGetYahooStockData(lStartDate,lEndDate , lTicker)

        #fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
         #                                  pNumDaysLookBack)

        #3rd item is a data frame of dates.
        train=fnGetHistoricalStockDataForSVM(dfQuotes,lNumDaysAheadPredict , lNumDaysLookBack,pSlopeLookback,pSRLookback,pSegments)

        testingData=fnGetHistoricalStockDataForSVM(dfQuotesTest,lNumDaysAheadPredict , lNumDaysLookBack,pSlopeLookback,pSRLookback,pSegments)

        #cPickle.dump(train, open('train.p', 'wb')) 
        #cPickle.dump(testingData, open('testingData.p', 'wb')) 
  
    else:
        pass
         #use previously saved data
        #train = cPickle.load(open('train.p', 'rb'))
        #testingData = cPickle.load(open('testingData.p', 'rb'))


    #{'C': 45000, 'gamma': 1e-05} for CAT (caterpillar stock, 12.19.2016
    C=135000#SVM Score: -11.1354870245 #'C': 1100000, 'gamma': 1e-06}
    gamma=1e-05

    #{'C': 2000, 'gamma': 1e-05}
    clfReg = svm.SVR(kernel='rbf', C=pC,gamma=pGamma,    epsilon =.01)
    #clfReg = svm.SVR(kernel='poly', C=pC,gamma=pGamma,degree=pDegrees,    epsilon =.001)  
    #clfReg =MLPRegressor(activation='logistic')
 
    #clfReg =MLPRegressor(hidden_layer_sizes=tupHiddenLayers,
    # activation='relu', solver='adam', alpha=0.0001,random_state=lRandomState,
    # batch_size='auto', learning_rate='constant', learning_rate_init=0.01)
        #cannot use tanh activation!

    #clfReg.out_activation_='tanh'
    #print(clfReg.out_activation_) -NOT supported

        #notes for MLP: reducing learning_rate_init improves SVR score greatly, -1.16
        
    X_train =train[0]
    y_train =train[1]



    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

    #dataset = scaler.fit_transform(dataset)
    #must implement feature scaling, or else volume will dominate
    scaler = preprocessing.StandardScaler().fit(X_train)
    
    #scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(X_train) #.46 score
    blnScale =False
    if blnScale:
        X_train =scaler.transform(X_train)  

    listC =list(range(10000,750000,35000))
    lStep =(.01-.0000001)/30
    listGamma =list(np.arange(.0000001,.01,lStep))

    parameters={'C':listC,
                'gamma':listGamma}
    #12.17.2016 clf.best_params_{'C': 1100000, 'gamma': 1e-07}
    clf =clfReg

    #CVData =fnGetCVIndices(dfQuotes)
    tscv = TimeSeriesSplit(n_splits=3)
    CVData =[(train,test) for train, test in tscv.split(X_train)]
    if blnGridSearch:
        clf = GridSearchCV(clfReg, parameters, verbose=1,n_jobs=3, cv=CVData)
    #clf =rbf_svm


    clf.fit(X_train, y_train)

    if blnGridSearch:
        print(clf.best_params_)  
        print('lookback: ' +str(lNumDaysLookBack))
    X_test =testingData[0]
    y_test =testingData[1]
    
    #grab prev. days' prices so we can use forecasted return to predict next price.
    y_PrevDayPrices=testingData[2]

   #grab the data frame to use the dates for plotting on graph.
    pDataFrame=testingData[3]

    if blnScale:
        X_test =scaler.transform(X_test)  
    

    score = clf.score(X_test, y_test)
    prediction =clf.predict(X_test)

    #reverse the scaler to get back original prices

    #need to reverse the natural log on prices
    if lBlnUseNaturalLog:
        prediction =np.exp(prediction)*y_PrevDayPrices
        y_test =np.exp(y_test)*y_PrevDayPrices
        #y_test =np.exp(y_test)*dfQuotesTest['Adj Close Price']
        
    #prediction=scaler.inverse_transform(prediction)

    #CANNOT TAKE LOG OF A NEGATIVE NUMBER, the forecast prediction can be negative
    # especially if stock is in a downtrend. this will fail for negative predictions.
    AccRatio =np.log(prediction/y_test)
    SSqAccRatio =sum(i**2 for i in AccRatio)
    print ('Accuracy Ratio Sum Sq. ' + str(SSqAccRatio))
    testScore = math.sqrt(mean_squared_error(y_test, prediction))
    print('Test Score: %.2f RMSE' % (testScore))

    print('SVM Score: ' +str(score))
    #r2_score(y_true, y_pred
    print ('R2 score on prices:' +str(r2_score(y_test ,prediction)))

    blnPlot=False

    if blnPlot:
        #need to trim the data frame dates to match length 
        lStartDate =len(pDataFrame)-len(y_test)
        pDataFrame=pDataFrame[lStartDate:]

        #plt.plot(pDataFrame['Date'],y_test,label='Actual ' + lTicker)
        plt.plot_date(pDataFrame['Date'], y_test, 'b-',label='Actual ' + lTicker)
        plt.plot_date(pDataFrame['Date'], prediction, 'g-',label='Predicted')
        #plt.plot(pDataFrame['Date']) 
        #ax.xaxis.set_minor_locator(months)
        #fig.autofmt_xdate()
        #plt.plot(prediction,label='Predicted')
   
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=33))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y'))
        plt.gcf().autofmt_xdate()

        #lNumDaysLookBack=30
        if 'lstCols' in globals()==False:
                lstCols=[]
        #lNumDaysAheadPredict=5
        lStrClassifier =repr(clf)
        lstrTitle ="\n".join(wrap('Classifier ' + lStrClassifier +  ' Stock: ' +lTicker + ' SVR C: ' + str(pC) + ' gamma ' +str(pGamma) + ' lookback: '+str(lNumDaysLookBack) +
                     ' daysAhead: ' + str(lNumDaysAheadPredict) + ' SVM Score: '+ str(score) +' features: ' +str(lstCols)))


        if True:
            plt.suptitle(lstrTitle,
                    fontsize=11, fontweight='bold')
    
        legend = plt.legend(loc='upper left', shadow=True, fontsize='x-large')

    if blnPlot:
        #ax = plt.add_subplot(111)
        #for xy in zip(pDataFrame['Date'], y_test):                                       #data labels
        #    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') 
            
        plt.show()

    return -score #SSqAccRatio #testScore

if __name__=='__main__':
        lNeurons=285
        numLayers=6
        #tup =tuple(h*6)
        result =None
        #initGuess=[numLayers,lNeurons]
        #lBounds=[(5,9),(20,700)]
        lBounds=[(1,3000000),(.0000000001,10),(7,40),(3,40),(2,10)] #(8,25) , (2,7)]
        #def fnMainWrapperSVRRBF(*pArgs):
        #pC=int(pArgs[0][0])
        #pGamma=pArgs[0][1]
        #pSlopeLookback=int(pArgs[0][2])
        #pLookback=int(pArgs[0][3])
        #pDaysAhead=int(pArgs[0][4])
        #pSRLookback=int(pArgs[0][5])
        #pSegments=int(pArgs[0][6])


        #tried (10,23) on days aheAD, AND 10 had best R ^2, 22 days is not valid
        #*MUST USE *args when calling a function from fmin_l_bfgs_b
        #fnMainWrapperSVRRBF([502811,0.00011855044037783498, 7]) #('C, gamma,pSlopeLookback ', 502811, 0.00011855044037783498, 7)
        #result=fmin_l_bfgs_b(func=fnMainWrapperSVRPoly,x0=initGuess,approx_grad=True,disp=1,bounds=lBounds,epsilon=1)
        #('C, gamma,pLookback,pDaysAhead', 60984, 0.00023225165833289416, 8, 34)
        #Best peformance is longer lookback on overall and shorter on S&R.
        #fnMainWrapperSVRRBF([343338, 7.1987125404345681e-08, 10, 30, 7])
        #fnMainWrapperSVRRBF([ 45000, 9.9999999999999995e-08, 22, 24, 9, 24, 6])
        #fnMainWrapperSVRRBF([ 60984, 0.00023225165833289416, 8, 34])

        #('pC, pGamma,pSlopeLookback,pLookback,pDaysAhead,pSRLookback,pSegments', 120, 0.22875079972240062, 11, 38, 21, 1, 1)
        
        #fnMainWrapperSVRRBF([120, 0.22875079972240062, 11, 38, 21])
        #fnMainWrapperSVRRBF([309948, 0.537111816058, 23, 15, 24])
        print ('no feature scaling')
        lBounds=[(70000,500000),(.000001,.9),(7,50),(40,150),(20,90) ]
        result=differential_evolution(func=fnMainWrapperSVRRBF,bounds=lBounds,disp=1)
        print(result)

        for i in range(8,9):
                #fnMain(i,True,h,numLayers) #25 look back was best
                #fnMainWrapper(initGuess)
                print (i)


    #?clf.best_params_
#{'C': 40000, 'gamma': 1e-0
#('layers	 neurons '	9	 224) using nearly all features, with MLP gives SVM of .14

#6 layers, 285 neurons
