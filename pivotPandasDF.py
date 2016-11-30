from sklearn import svm
import pandas as pd
import datetime
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc,\
    volume_overlay2,volume_overlay3

from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV


def fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
                                   pNumDaysLookBack):

    #sort by date asc
    df=pDataFrameStockData
    df.Date = pd.to_datetime(df.Date)
    df.sort(['Date'], inplace=True)
    lst_Y =[]
    lStrTicker ='TICKER'
    
    
    iRowCtr =0
    #dfFilter =df[df['Date']<datetime.date(year=2015,month=9,day=6)]
    lEnd =len(df)
    lEndRow =1
    lRowPredictedPrice =pNumDaysAheadPredict
    df['Ticker'] =lStrTicker 
    result =[]
    while (iRowCtr+pNumDaysLookBack+pNumDaysAheadPredict)<=lEnd:
    #for iRowCtr in range(0, lEnd):
        lEndRow =iRowCtr+pNumDaysLookBack
        #p = df[df['Date']<datetime.date(year=2015,month=9,day=6)].pivot(index='Ticker', columns='Date')
        p = df[iRowCtr:lEndRow].pivot(index='Ticker', columns='Date')

        result.append(list(p.T[lStrTicker][:]))

        lRowPredictedPrice =lEndRow+pNumDaysAheadPredict-1
        lst_Y.append(df['Close'][lRowPredictedPrice:lRowPredictedPrice+1].values[0])
        iRowCtr=iRowCtr+1



    return result, lst_Y


def fnGetYahooStockData(pStartDate, pEndDate, pSymbol):
    # (Year, month, day) tuples suffice as args for quotes_historical_yahoo
    dateStart = (pStartDate.year, pStartDate.month, pStartDate.day)
    dateEnd = (pEndDate.year, pEndDate.month, pEndDate.day)

    sSymbol =pSymbol

    quotes = quotes_historical_yahoo_ohlc(sSymbol, dateStart, dateEnd)
    if len(quotes) == 0:
        raise SystemExit

    dfQuotes =pd.DataFrame(quotes,columns=['Date','Open','Close','High','Low','Volume'])
    return dfQuotes
#df=pd.read_csv('..//StockData.csv')


def fnMain():
    blnGridSearch =False
    
    #train data
    lStartDate=datetime.date(2002, 1, 6)
    lEndDate=datetime.date(2003, 12, 25)

    dfQuotes =fnGetYahooStockData(lStartDate,lEndDate , "XOM")

    #test data
    lStartDate=lEndDate #datetime.date(2003, 12, 25)
    lEndDate=datetime.date(2004, 12, 24)

    dfQuotesTest =fnGetYahooStockData(lStartDate,lEndDate , "XOM")

    #fnGetHistoricalStockDataForSVM(pDataFrameStockData, pNumDaysAheadPredict,
     #                                  pNumDaysLookBack)

    lNumDaysLookBack=120
    lNumDaysAheadPredict=44
    train=fnGetHistoricalStockDataForSVM(dfQuotes,lNumDaysAheadPredict , lNumDaysLookBack)

    testingData=fnGetHistoricalStockDataForSVM(dfQuotesTest,lNumDaysAheadPredict , lNumDaysLookBack)



    # fit the model and calculate its accuracy
    #{'C': 500000, 'gamma': 1e-06}
    rbf_svm = svm.SVR(kernel='rbf', C=25000000,gamma=1e-06,    epsilon =.001)


    X_train =train[0]
    y_train =train[1]

    #must implement feature scaling, or else volume will dominate
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train =scaler.transform(X_train)  

    parameters={'C':[45000,80000,500000,750000,1200000,25000000,80000000],'gamma':[1e-03,1e-06,1e-10,1e-16]}
    
    clf =rbf_svm
    if blnGridSearch:
        clf = GridSearchCV(rbf_svm, parameters, verbose=1,n_jobs=3)
    #clf =rbf_svm


    clf.fit(X_train, y_train)

    if blnGridSearch:
        print(clf.best_params_)

    X_test =testingData[0]
    y_test =testingData[1]

    X_test =scaler.transform(X_test)  
    score = rbf_svm.score(X_test, y_test)
    print(score)
    #print(test) result=  rbf_svm.predict(X_test)

    #?result[10:20]
    #y_test[10:20]
    #?rbf_svm.fit(X_train, y_train).score(X_test,y_test)


if __name__=='__main__':
	fnMain()


    #?clf.best_params_
#{'C': 40000, 'gamma': 1e-06}