import pandas as pd
import datetime


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
        lst_Y.append(df['close'][lRowPredictedPrice:lRowPredictedPrice+1])
        iRowCtr=iRowCtr+1


    return result, lst_Y




df=pd.read_csv('..//StockData.csv')

test=fnGetHistoricalStockDataForSVM(df,6 , 5)
print(test)