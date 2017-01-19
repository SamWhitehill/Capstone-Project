
#from pandas_datareader import data as web #, wb
import pandas as pd
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
from scipy.optimize import differential_evolution

import statsmodels.api as sm
from scipy.stats import norm
from datetime import datetime

from MainForecastModule import fnGetNaturalLogPrices
from MainForecastModule import fnGetHistoricalStockDataForSVM

"""
A nonseasonal ARIMA model is classified as an ARIMA(p,d,q) model, where:

p is the number of autoregressive terms,
d is the number of nonseasonal differences needed for stationarity, and
q is the number of lagged forecast errors in the prediction equation.
"""

def mean_forecast_err(y, yhat):
    return y.sub(yhat).mean()

lstrPath="C:\\Udacity\\NanoDegree\\Capstone Project\\MLTrading\\"
#data = web.get_data_yahoo('SPY', dt.datetime(2016,1,3), dt.datetime(2016,06,22))
data =pd.read_csv(lstrPath+'SPY Long history.csv')
data.Date = pd.to_datetime(data.Date)
data.sort(['Date'], inplace=True)

numDaysAhead =47    
numDaysLookBack=7

XTrain=None
lst_Y=None
lstPreviousDayPrices=None
df=None

data['Adj Close Price']=data['Adj Close']
#data = data.ix[:,5].values.tolist()
data=fnGetNaturalLogPrices(data,numDaysAhead)

#XTrain, lst_Y,  lstPreviousDayPrices,df= fnGetHistoricalStockDataForSVM(data,numDaysAhead,numDaysLookBack,45,1,1)

#XTest, lst_YTest,  lstPreviousDayPricesTest,dfTest= fnGetHistoricalStockDataForSVM(dataTest,numDaysAhead,numDaysLookBack,45,1,1)
data =data[ pd.notnull(data['Adj Close'])]

data=data[['Adj Close','Date']]
data.index =data['Date']
#dta = sm.datasets.sunspots.load_pandas().data

#data.index = pd.Index(sm.tsa.datetools.dates_from_range('2006-03-20', '2010-08-10'))
#dta.index = pd.Index(sm.tsa.datetools.dates_from_range('1700', '2008'))
#del dta["YEAR"]
del data['Date']

#res = sm.tsa.arma_order_select_ic(data, ic=['aic', 'bic'], trend='nc'), data.to_csv("C:\\temp\\dataWithReturns.csv")
data.index.freq = 'D'

origData =data
stopIndex =2151
data=data[:stopIndex]

startIndex =7
arma_mod30 = sm.tsa.ARMA(data, (startIndex,0,3), ).fit(trend='nc',method = 'css-mle')
#arma_mod30 = sm.tsa.ARMA(data, (20,0,3)).fit()


#predict_sunspots = arma_mod30.predict('1990', '2012', dynamic=True) result=arma_mod30.forecast(3)[0]

#predict_returns=arma_mod30.predict(start='2006-01-18', end='2016-12-30',dynamic=True)
#predict_returns=arma_mod30.predict(start=startIndex, dynamic=True)
 
#predict_returns=arma_mod30.predict(start = data.index[10],dynamic=False)

if False:
	fig, ax = plt.subplots(figsize=(12, 8))

	ax = data.ix['2006-01-18':].plot(ax=ax)
	fig = arma_mod30.plot_predict(start=startIndex, dynamic=False, ax=ax, plot_insample=False)
#ax = data.ix['2006-03-20':].plot(ax=ax) 
#np.savetxt("C:\\temp\\predict_returns.csv", predict_returns, delimiter=',')
#fig = arma_mod30.plot_predict('1990', '2012', dynamic=True, ax=ax, plot_insample=False)
#fig = arma_mod30.plot_predict(start = data.index[100], dynamic=True,  plot_insample=False)
#predict_returns=arma_mod30.predict(start ='2014-05-21',end= len(data)+15,dynamic=True)
#np.savetxt("C:\\temp\\predict_returns.csv", predict_returns, delimiter=',')

#********************************************
lStop =len(origData) -stopIndex
predictions=[]
for i in range(lStop):
	arma_mod30 = sm.tsa.ARMA(origData[:i+1+stopIndex], (startIndex,0,3), ).fit(trend='nc',method = 'css-mle')
	result=arma_mod30.forecast(1)[0]
	predictions.append(result)
#SEPARATE PLOT
plt.figure(figsize=(8,6))
y =origData[stopIndex-1:]
plt.plot(y, label='True Return', color='#377EB8', linewidth=2)
plt.plot(predictions, '-',color='#EB3737', linewidth=1, label='Prediction')
plt.show()

#print(mean_forecast_err(dta.SUNACTIVITY, predict_sunspots))
