import  BuildTrendLines as trendy
import pandas as pdMain
# Download Apple price history and save adjusted close prices to numpy array
import pandas.io.data as pd
import datetime
from TestTrendlines import pdMain

#start = datetime.datetime(2009, 1, 1)

#end = datetime.datetime(2009, 9, 27)
#x = pd.DataReader("GLD", "yahoo",start,end)['Adj Close']

# Make some trendlines
#import trendy

# Generate general support/resistance trendlines and show the chart
# winow < 1 is considered a fraction of the length of the data set
#trendy.gentrends(x, window = .1, charts = True)

# Generate a series of support/resistance lines by segmenting the price history
#trendy.segtrends(x, segments = 2, charts = True)  # equivalent to gentrends with window of 1/2


def fnGetSRperDF(x,pNumDaysAheadPredict):

    #get LAST date
    lLastDate =x.index.tolist()
    lLastDate=lLastDate[len(lLastDate)-1]
    
    #seg trends requires series param
    x=x[['Adj Close']]
    x_maxima, maxima, x_minima, minima,lstMinLines ,lstMaxLines=\
        trendy.segtrends(x, segments =4, charts = False,pProjDaysAhead=pNumDaysAheadPredict)  # plots several S/R lines
    #num lstSupportLines is 1 less than # segments.
    lstSupportLines =[i[-1:][0] for i in lstMinLines] #S1, S2, etc.. field in DF!
    lstResistanceLines =[i[-1:][0] for i in lstMaxLines] #R1, R2S5
    #print ('inside')
    #ONLY need 1 data point per support line or resistance line since forecast is for a 
    # point in time

    lstSR =list(minima)+lstSupportLines+lstResistanceLines+list(maxima)
    #lstSR=list(minima)+list(maxima)
    lstMin =['Min'+str(i+1) for i,val in enumerate(minima)]#['Min1','Min2','Min3','Min4','Min5']
    lstMax =['Max'+str(i+1) for i,val in enumerate(maxima)]
    lstSupp =['S'+str(i+1) for i,val in enumerate(lstSupportLines)]
    lstResist=['R'+str(i+1) for i,val in enumerate(lstResistanceLines)]
    lstCols =lstMin+lstSupp+lstResist+lstMax
    dictSR =dict( zip( lstCols, lstSR))
        
    tmpFrame=pd.DataFrame(dictSR,columns=lstCols,index=[lLastDate])
    return tmpFrame, lstCols #1 #lstSupportLines+lstResistanceLines

#print(fnGetSR(x))

#pDf= pd.DataReader("GLD", "yahoo",start,end)
#pDf['Date'] = pDf.index

def fnGetSupportResistance(pDf,pNumDaysLookBack=30, pNumDaysAheadPredict=8):

    dfSR =None
    lEnd =len(pDf)
    iRowCtr=0
    #range is faster than while
    lEnd=lEnd-(pNumDaysLookBack)+1
    iRowCtr=0 #cannot start at 0, treat like SMA...
    for i in range(0,lEnd): #start, stop, step
    #while (iRowCtr+pNumDaysLookBack+pNumDaysAheadPredict)<=lEnd:
            #for iRowCtr in range(0, lEnd):
            lEndRow =i+pNumDaysLookBack
            #p = df[df['Date']<datetime.date(year=2015,month=9,day=6)].pivot(index='Ticker', columns='Date')
            p = pDf[iRowCtr:lEndRow]
            tmpFrame,lstCols =fnGetSRperDF(p,pNumDaysAheadPredict)

            if dfSR is None:
                 dfSR=tmpFrame #pd.DataFrame(dictSR,columns=lstCols,index=[lFirstDate])
            else:
                #tmpFrame=pd.DataFrame(dictSR,columns=lstCols,index=[lFirstDate])
                dfSR=dfSR.append(tmpFrame, ignore_index=False) 
             
            iRowCtr=iRowCtr+1
    
    dfSR['Date'] = dfSR.index               
    return dfSR, lstCols

#print(fnGetSupportResistance(pDf))
#rollingSlopeClose =pdMain.rolling_apply(pDf['Adj Close'],10,fnGetSR)
#rollingSlopeClose =	pDf['Adj Close'].rolling(center=False,window=10).apply(func=fnGetSR)

# Generate smaller support/resistance trendlines to frame price over smaller periods
#x=x[len(x)-100:]
#x=x[420:]
#return xMax, yMax, xMin, yMin ,projSupport,projResistance
#xMax, yMax, xMin, yMin ,projSupport,projResistance=\
#    trendy.minitrends(x, window = .2, charts = True) #try this!!

#print(projSupport)
#print(projResistance)

# Iteratively generate trading signals based on maxima/minima in given window
#trendy.iterlines(x, window = 30, charts = True)  # buy at green dots, sell at red dots
