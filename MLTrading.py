import pandas as pd


def get_bollinger_bands(rm, rstd):

	upper_band=rm + (2*rstd)

	lower_band=rm + (-2*rstd)

	return upper_band, lower_band




dfStockData = pd.read_csv("StockData.csv")

rollingMean =pd.rolling_mean(dfStockData['Close'],window=5)

rollingStdev =pd.rolling_std(dfStockData['Close'],window=5)

rollingMean.fillna(value=0,inplace=True)
rollingStdev.fillna(value=0,inplace=True)

#rollingMean =rollingMean+10
#print(rollingMean)

upper_band, lower_band =get_bollinger_bands(rollingMean,rollingStdev )

print( upper_band[5:10])



