import pandas as pd
import datetime

lstrTicker ='IBM'
df=pd.read_csv('..//StockData.csv')

df.Date = pd.to_datetime(df.Date)
df['Ticker'] =lstrTicker  

df =df[df['Date']<datetime.date(year=2015,month=9,day=6)]

p = df.pivot(index='Ticker', columns='Date')
                
print(list(p.T[lstrTicker][:]))

#dfUnstack =df.unstack(level=-1)

#print(dfUnstack[:])
#table =pd.pivot_table(df, values='D', index=['Date'],
#                     columns=['open'], aggfunc=min)