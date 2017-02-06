
''' IMPORTS '''
from sklearn.model_selection import TimeSeriesSplit
#from SupportAndResistance  import fnGetSupportResistance
import math
from sklearn import svm
import pandas as pd
import datetime
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc,\
    volume_overlay2,volume_overlay3
#from pandas_datareader import data as web
import datetime as dt
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from sklearn.neural_network import MLPRegressor
#from neon.data import IMDB
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
#import cPickle
from sklearn.preprocessing import LabelEncoder  
import numpy as np
from textwrap import wrap
from scipy.stats import linregress
import matplotlib.dates as mdates
from scipy.optimize import fmin_l_bfgs_b
from scipy.optimize import curve_fit
from scipy.optimize import differential_evolution


from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

'''END IMPORTS '''

def get_bollinger_bands(rm, rstd):
	#function calculates bollinger bands (technical analysis) using 
	#rolling mean and rolling standard deviation panda series (params rm, rstd)
	#returns upper and lower bands as pandas series
	upper_band=rm + (2*rstd)

	lower_band=rm + (-2*rstd)

	return upper_band, lower_band

def fnConvertSeriesToDf(pdSeries, pLstColumns):
    #returns dataframe from the series:pdSeries, using the pLstColumns as dataframe columns
    df = pd.DataFrame(pdSeries).reset_index()
    df.columns = pLstColumns
    return df


def fnCalculateLogReturn(pDf,N=1):
	#calculate the log return of a closing stock price
	#return the log return data frame column
    return np.log(pDf['Close']).diff()

def fnCalculateEMV(data, nDays):
    #function calculates the Ease of movement -EMV technical volume indicator
	#Ease of movement
    #Reference: https://www.quantinsti.com/blog/build-technical-indicators-in-python/
	#returns pandas data
     dm = ((data['High'] + data['Low'])/2) - ((data['High'].shift(1) + data['Low'].shift(1))/2)
     br = (data['Volume'] / 100000000) / ((data['High'] - data['Low']))
     EVM = dm / br 
     EVM_MA = pd.Series(pd.rolling_mean(EVM, nDays), name = 'EMV') 
     data = data.join(EVM_MA) 
     return data


def fnCalcForceIndex(data, ndays):
    #calculate the technical indicator force index (basd on volume and price) 
	# Force Index 
	#return data frame containg force index from params:
	#data - dataframe and ndays lookback
     FI = pd.Series(data['Close'].diff(ndays) * data['Volume'], name = 'ForceIndex') 
     data = data.join(FI) 
     return data



def OBV(pDf, n):  
	#Calculate On-balance Volume  indicator
	#params: pDf - datafame containing adj closing prices
	#n is number of days lookback
	#returns dataframe with OBV indicator
	#Reference: https://www.quantopian.com/posts/technical-analysis-indicators-without-talib-code

    pDf['OBV']=0
    pDf['OBV'] =np.where(pDf['Adj Close'] -pDf['Adj Close'].shift(1)>0,
                                                pDf['Volume'],pDf['OBV'])

    pDf['OBV'] =np.where(pDf['Adj Close'] -pDf['Adj Close'].shift(1)==0,0,pDf['OBV'])

    pDf['OBV'] =np.where(pDf['Adj Close'] -pDf['Adj Close'].shift(1)<0, -pDf['Volume'],pDf['OBV'])

    #OBV_ma = pd.Series(pd.rolling_mean(pDf['OBV'], n), name = 'OBV') #_' + str(n))  
    OBV_ma=pd.rolling_mean(pDf['OBV'],window=n)
    OBV_ma=fnConvertSeriesToDf(OBV_ma,['Date', 'OBV'])

    #pDf = pDf.join(OBV_ma) 
    del pDf['OBV']
    pDf =pDf.merge(OBV_ma,how='left',on=['Date']).set_index(['Date'], drop=False)

    return pDf

def fnCalcRSI(price, n=14):
        #calculate RSI (Relative Strength Indicator)
        #params: price is a dataframe column, n is the lookback
        #return dataframe with RSI indicator
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
		#params: pDf - dataframe containing Adj Close, N is number of days back
        #return dataframe pDf with N day net Adj closeprice change where N is number days back
        pDf[str(N)+'DayNetPriceChange'] =pDf['Adj Close'] -pDf['Adj Close'].shift(N)
        return pDf

def fnCalcAvgVolumeStats(pDf,N=10):
		#Calculate various avg. volume statistics
		#return dataframe with these vol calculations
		#params: pDf - dataframe containing Volume 
		#N - number of days lookback for avg calc
		
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
	#wrapper function to calculate the slope (via linregress below)
	#of a pVlues - this would be a price series or returns series
	#returns slope over the PValues series
      iLen =len(pValues)
      lXVals =range(iLen)
      pValues=pValues/1000000
	
      slope_0, intercept, r_value, p_value, std_err=linregress(lXVals, pValues)
      return slope_0

#create the function we want to fit
def fnSin(x, freq, amplitude, phase, offset):
	#custom sine function used to fit over stock returns data
	#the sine function tries approximate the shape of a stationary stock time series
	#the parms freq, amplitude, phase, offset) are fit via curve_fit
		return np.sin(x * freq + phase) * amplitude + offset

def fnWrapCurve_Fit(pValues, pOutputCol=0):
	#wrapper function for curve_fit to be used in pandas rolling apply
	#this function fits the above sine wave fn to a series of values: pValues
	#these are prices or returns time series
		try:
			iLen =len(pValues)
			lXVals =np.arange(iLen) 
			#pValues=pValues/1000000
			guess_freq = 1
			guess_amplitude = 3*np.std(pValues)/(2**0.5)
			guess_phase = 0
			guess_offset = np.mean(pValues)

			p0=[guess_freq, guess_amplitude,
			guess_phase, guess_offset]

			# now do the fit
			fit = curve_fit(fnSin, lXVals, pValues, p0=p0,maxfev=5500)
			#*4 elements in fit[0]freq, amp, phase, offset
			#return fit[0][0], fit[0][1],fit[0][2],fit[0][3]
			return fit[0][pOutputCol]
		except Exception as e:
			print( str(e))
        #return pd.Series({'freq': fit[0][0], 'amplitude': fit[0][1]})

def moving_average_convergence(pDf, nslow=26, nfast=12): #26,12
	#Calculated MACD technical indicator using:
	#pDf - dataframe with Closing prices/returns
	#nslow - number of lookback periods on slow EMA
	#nfast - number of lookback periods on fast EMA
	#returns datafame containing emafast, emaslow, MACD
    pDf['emaslow'] = pd.ewma(pDf["Close"], span=nslow, freq="D",min_periods=nslow)
    pDf['emafast'] = pd.ewma(pDf["Close"], span=nfast, freq="D",min_periods=nfast)

    #emaslow = pd.ewma(group, span=nslow, min_periods=1)
    #emafast = pd.ewma(group, span=nfast, min_periods=1)
    #result = pd.DataFrame({'MACD': emafast-emaslow, 'emaSlw': emaslow, 'emaFst': emafast})
    pDf['MACD']=pDf['emafast'] - pDf['emaslow']

    return pDf

def fnCalculateSlope(pDf,N=10):
        #calculate slope on a rolling basis on a series of values - either prices or returns
		#params: pDf -datframe containing Adj Close, Volume,etc..
		#N - lookback period
		#returns dataframe with slope calculations
        #slope_0, intercept, r_value, p_value, std_err =\
        #stats.linregress(pDf['Date'], pDf['Adj Close'])
        #rollingSlopeLow =pd.rolling_apply(pDf[['Low']],N,fnWraplinregress)
        #rollingSlopeLow=fnConvertSeriesToDf(rollingSlopeLow,['Date', 'LowSlope'])
        fld ='Adj Close'

        rollingSineFreq=pd.rolling_apply(arg=pDf[[fld]],window=N,func=fnWrapCurve_Fit,args=(0,))
        rollingSineFreq=fnConvertSeriesToDf(rollingSineFreq,['Date', 'SineFreq'])

        rollingSineAmp=pd.rolling_apply(arg=pDf[[fld]],window=N,func=fnWrapCurve_Fit,args=(1,))
        rollingSineAmp=fnConvertSeriesToDf(rollingSineAmp,['Date', 'SineAmp'])

        rollingSinePhase=pd.rolling_apply(arg=pDf[[fld]],window=N,func=fnWrapCurve_Fit,args=(2,))
        rollingSinePhase=fnConvertSeriesToDf(rollingSinePhase,['Date', 'SinePhase'])

        rollingSineOffset=pd.rolling_apply(arg=pDf[[fld]],window=N,func=fnWrapCurve_Fit,args=(3,))
        rollingSineOffset=fnConvertSeriesToDf(rollingSineOffset,['Date', 'SineOffset'])

        rollingSlopeClose =pd.rolling_apply(pDf[['Close']],N,fnWraplinregress)
        rollingSlopeClose=fnConvertSeriesToDf(rollingSlopeClose,['Date', 'CloseSlope'])

        rollingSlopeVol=pd.rolling_apply(pDf[['Volume']],N,fnWraplinregress)
        rollingSlopeVol=fnConvertSeriesToDf(rollingSlopeVol,['Date', 'VolumeSlope'])

        rollingSlopeStdDev =pd.rolling_apply(pDf[['rollingStdev20']],N,fnWraplinregress)
        rollingSlopeStdDev=fnConvertSeriesToDf(rollingSlopeStdDev,['Date', 'StdDevSlope'])

        #rollingSlopeHi =pd.rolling_apply(pDf[['High']],N,fnWraplinregress)
        #rollingSlopeHi=fnConvertSeriesToDf(rollingSlopeHi,['Date', 'HighSlope'])

        #pDf =pDf.merge(rollingSlopeLow,how='left',on=['Date']).set_index(['Date'], drop=False)
        #pDf =pDf.merge(rollingSlopeHi  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        pDf =pDf.merge(rollingSlopeVol  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        pDf =pDf.merge(rollingSlopeClose  ,how='left',on=['Date']).set_index(['Date'], drop=False)
        pDf =pDf.merge(rollingSlopeStdDev  ,how='left',on=['Date']).set_index(['Date'], drop=False)

        pDf =pDf.merge(rollingSineFreq  ,how='left',on=['Date']).set_index(['Date'], drop=False)

        pDf =pDf.merge(rollingSineAmp  ,how='left',on=['Date']).set_index(['Date'], drop=False)

        pDf =pDf.merge(rollingSinePhase  ,how='left',on=['Date']).set_index(['Date'], drop=False)

        pDf =pDf.merge(rollingSineOffset  ,how='left',on=['Date']).set_index(['Date'], drop=False)

        #need a 
        #x.groupby('entity').apply(lambda v: linregress(v.year, v.value)[0])
        #[0] means slope only

        return pDf

def fnComputeCandleStickPattern(pDf):
   #Calculate various candlestick patterns (realbody, upper/lower shadow,etc..)
   # based on candlestick technical analysis
   #params: pDf - dataframe with open, high, low, close, adj close, volume stock data
   #returns dataframe with candlestick pattern info
   #If the high and low of a bar is higher than previous bar, then that bar is
   # called an 'up bar' or an 'up day'. If the high and low of a bar is lower
   # than previous bar, then that bar is called an 'down bar' or an 'down day'
   #Reference: http://www.stock-trading-infocentre.com/bar-charts.html

        pDf['Color']=np.where(pDf['Close'] >pDf['Open'] ,'WHITE','BLACK') #use numbers not colors
        
        pDf['RealBody']= np.where(pDf['Open']!=0, np.absolute(pDf['Close'] -pDf['Open'])/pDf['Open']*1, 0)

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
		#params: pDf - dataframe with open, high, low, close, adj close, volume stock data
		#pNumDaysLookBack - days to lookback when calculating
		#pSlopeLookback -days to lookback on slope
		#pDaysAhead  - horizon to forecast
		#pSRLookback -Support and resistance lookback - not currently used
		#pSegments -was used for Support and resistance- not currently used
        #returns dataframe with the features
        # compute rolling mean, stdev, bollinger
        lstCols=None
        
        #dfSR, lstCols =fnGetSupportResistance(pDf,pSRLookback,pDaysAhead,pSegments)

        #pDf =pDf.merge(dfSR,how='inner',on=['Date'],right_index=True)
        #not using candlestick patterns any longer
        pDf=fnComputeCandleStickPattern(pDf)
        pDf['HighLowRange'] =pDf['High'] -pDf['Low']

        pDf =fnCalcNDayNetPriceChange(pDf,2)

        pDf=fnCalcAvgVolumeStats(pDf,pNumDaysLookBack)

        pDf=moving_average_convergence(pDf) # pNumDaysLookBack,int(pNumDaysLookBack/3))

        pDf['RSI'] =fnCalcRSI(pDf['Close'])

        pDf=fnCalculateEMV(pDf, pNumDaysLookBack)

        pDf=fnCalcForceIndex(pDf,pNumDaysLookBack)

		#compute on balance volume
        pDf= OBV(pDf, pNumDaysLookBack) 

        #compute log return
        #pDf['DailyLogReturn'] =fnCalculateLogReturn(pDf,1)

        #RSI decision fields, rsi >= 70 overbought, rsi <= 30 oversold
        #pDf['RSIDecision'] =np.where(pDf['RSI']  >=70,10*pDf['Adj Close'],0)
        #pDf['RSIDecision'] =np.where(pDf['RSI'] <=30,-10*pDf['Adj Close'],pDf['RSIDecision'])

        rollingMax=pd.rolling_max(pDf['Adj Close'],window=pNumDaysLookBack)
        rollingMin =pd.rolling_min(pDf['Adj Close'],window=pNumDaysLookBack)

        rollingMean =pd.rolling_mean(pDf['Adj Close'],window=pNumDaysLookBack)
        rollingMeanFifty =pd.rolling_mean(pDf['Adj Close'],window=50)

        #26 day std dev gives best results
        rollingStdev =pd.rolling_std(pDf['Adj Close'],window=pNumDaysLookBack) # CHANGED TO 10 DAY FROM research paper

        rollingMeanFifty.fillna(value=0,inplace=True)
        rollingMean.fillna(value=0,inplace=True)
        rollingStdev.fillna(value=0,inplace=True)

        rollingMax.fillna(value=0,inplace=True)
        rollingMin.fillna(value=0,inplace=True)


        #df =df.merge(rollingMean,how='inner',on=['Date'])

        #rollingMean =rollingMean+10
        #print(rollingMean)
        #upper /lower bands are pandas series
        if False:
            upper_band, lower_band =get_bollinger_bands(rollingMean,rollingStdev )

        #append additional stats into original dataframe
        #first create dataframes
        #name column
        if False:
            upper_band=fnConvertSeriesToDf(upper_band,['Date', 'upper_band'])

            lower_band=fnConvertSeriesToDf(lower_band,['Date', 'lower_band'])

        rollingStdev=fnConvertSeriesToDf(rollingStdev,['Date', 'rollingStdev20'])
        rollingMean=fnConvertSeriesToDf(rollingMean,['Date', 'rollingMean20'])
		
        rollingMax=fnConvertSeriesToDf(rollingMax,['Date', 'rollingMax20'])
        rollingMin=fnConvertSeriesToDf(rollingMin,['Date', 'rollingMin20'])

        rollingMeanFifty=fnConvertSeriesToDf(rollingMeanFifty,['Date', 'rollingMean50'])

        pDf =pDf.merge(rollingMean,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingStdev,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingMeanFifty,how='inner',on=['Date'],right_index=True)

        pDf =pDf.merge(rollingMax,how='inner',on=['Date'],right_index=True)
        pDf =pDf.merge(rollingMin,how='inner',on=['Date'],right_index=True)

        #CANNOT HAVE any Nans, must all be valid number so cut off records with invalid Nans.
        #if 'VolumeSlope' in pDf.columns:
            #pDf =pDf[ pd.notnull(pDf['VolumeSlope'])]
			#pass
        #print ('skipping volume slope')
        

        pDf =fnCalculateSlope(pDf,pSlopeLookback) #pNumDaysLookBack try 32
		#pDf =pDf[ pd.notnull(pDf['EMV'])]
        pDf =pDf[ pd.notnull(pDf['MACD'])]
        #pDf =pDf[ pd.notnull(pDf['RSI'])]
        pDf =pDf[ pd.notnull(pDf['CloseSlope'])]
        
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
                #pDf =pDf[pDf['upper_band']>0]
                pDf =pDf[ pd.notnull(pDf['upper_band'])]

                #compute diff between price and up/low bollingers
                #pDf['DiffercenceBtwn_upper_band'] =pDf['upper_band']-pDf['Close']
                #pDf['DiffercenceBtwn_lower_band'] =pDf['Close']-pDf['lower_band']

                #remove bollingers now
                #del pDf['upper_band']
                #del pDf['lower_band']
    

        return pDf, lstCols


def fnGetYahooStockData(pStartDate, pEndDate, pSymbol):
	#Retrieve historical stock data from yahoo finance.
	
	#params:
    # (Year, month, day) tuples suffice as args for quotes_historical_yahoo
    #dateStart = (pStartDate.year, pStartDate.month, pStartDate.day)    
    #dateEnd = (pEndDate.year, pEndDate.month, pEndDate.day)
	#pSymbol - ticker of stock to retrieve
	#returns dataframe with O H L C V stock data

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
	
	#SORT dataframe by date in ascending order!
    dfQuotes.Date = pd.to_datetime(dfQuotes.Date)
    dfQuotes.sort(['Date'], inplace=True)

    return dfQuotes


def fnGetNaturalLogPrices(pDf,Nperiods=1):
    #return the data frame with natural log of open, hi, low , close prices
    #this should make it better for time series analysis
    lstFlds =['Adj Close','Close','Open','High','Low']
    for fld in lstFlds:
        #pDf[fld] =np.log( pDf[fld] )
        pDf[fld] = np.log(pDf[fld]/pDf[fld].shift(periods=Nperiods))

    return pDf

def window_stack(a, stepsize=1, width=3):
	#function arranges data for use in neural network by stacking it
	#according to stepsize and width params
	#a is the data to stack
	#returns np structure in desired format
        n = a.shape[0]
        return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

def organize_data(to_forecast, window, horizon):
		#function arranges data for use in neural network by 
        #
        #Parms
        # to_forecast is a time series organized as numpy array
        # window, number of items to use in the forecast window
        # horizon, days ahead of the forecast
        #returns :
        # X - a matrix where each row contains a forecast window
        # y - the target values for each row of X
        
    
        shape = to_forecast.shape[:-1] + (to_forecast.shape[-1] - window + 1, window)
        strides = to_forecast.strides + (to_forecast.strides[-1],)
        X = np.lib.stride_tricks.as_strided(to_forecast, 
                                            shape=shape, 
                                            strides=strides)
        y = np.array([X[i+horizon][-1] for i in range(len(X)-horizon)])
        return X[:-horizon], y


