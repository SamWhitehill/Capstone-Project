def OBV(pDf, n):  

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
