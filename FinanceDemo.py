#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
from matplotlib.finance import quotes_historical_yahoo_ohlc, candlestick_ohlc,\
    volume_overlay2,volume_overlay3

from matplotlib import transforms
import datetime as date



# (Year, month, day) tuples suffice as args for quotes_historical_yahoo
date1 = (1995, 6, 1)
date2 = (1995, 11, 12)


mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
alldays = DayLocator()              # minor ticks on the days
weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
dayFormatter = DateFormatter('%d')      # e.g., 12

sSymbol ='IBM'

quotes = quotes_historical_yahoo_ohlc(sSymbol, date1, date2)
if len(quotes) == 0:
    raise SystemExit

ds, opens, closes, highs, lows, volumes = zip(*quotes)

tmpdt=date.date.fromordinal(int(ds[0]))
strdate=date.date.strftime(tmpdt,'%Y-%m-%d')

print(strdate)

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)
ax.xaxis.set_major_locator(mondays)
ax.xaxis.set_minor_locator(alldays)
ax.xaxis.set_major_formatter(weekFormatter)
#ax.xaxis.set_minor_formatter(dayFormatter)

#plot_day_summary(ax, quotes, ticksize=3)
candlestick_ohlc(ax, quotes, width=0.8,colorup='#53c156', colordown='#ff1717')

ax.xaxis_date()
ax.autoscale_view()
plt.setp(plt.gca().get_xticklabels(), rotation=45, horizontalalignment='right')
ax.set_title(sSymbol)

# Add a seconds axis for the volume overlay
#ax2t = ax.twinx()

#ax2t = plt.subplot2grid((5,4),(4,0),sharex=ax,rowspan=1,colspan=4)
# Plot the volume overlay

#volume = (closes*volumes)/1e6  # dollar volume in millions
if False:    
    vmax = max(list(volumes))
    poly = ax2t.fill_between(ds, volumes, 0, label='Volume', facecolor='red', edgecolor='black')
    ax2t.set_ylim(0, 5*vmax)
    ax2t.set_yticks([])

#bc = volume_overlay(ax2, opens, closes, volumes, colorup='g', alpha=0.9, width=.5)
#volume_overlay2(ax2t, closes, volumes, colorup='g',colordown='r',width=1,alpha=.2)
#ax2t.set_position(transforms.Bbox([[0.125,0.1],[0.9,0.32]]))
#bc=volume_overlay2(ax2t, quotes, colorup='g', colordown='r', width=1, alpha=.2)
#matplotlib.finance.volume_overlay3(ax, quotes, colorup='k', colordown='r', width=4, alpha=1.0)
#Add a volume overlay to the current axes. quotes is a list of (d, open, high, low, close, volume) and close-open is used to determine the color of the bar

#matplotlib.finance.volume_overlay2(ax, closes, volumes, colorup='k', colordown='r', width=4, alpha=1.0)
#Add a volume overlay to the current axes. 
#The closes are used to determine the color of the bar. -1 is missing. 
#If a value is missing on one it must be missing on all

#ax2t.bar(ds,volumes)

#matplotlib.finance.candlestick(ax, DOCHLV, width=0.6, colorup='g', colordown='r', alpha=1.0)
#nb: first point is not displayed - it is used only for choosing the right color
#ax2t.add_collection(bc)

#plt.hold(True)
plt.show()