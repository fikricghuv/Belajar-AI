# -*- coding: utf-8 -*-
"""Buy-sell-Bollingerband.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SSYmvX1mBdvMtvquO1fLvxhJ3w1mV4jW
"""

#Description this program uses the bolingerband strategy to determine when to buy and sell stock

#import the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import pandas_datareader as web

#Load the data
#Get the Stock
df = web.DataReader('ADRO.JK', data_source='yahoo', start='2020-06-17', end='2021-04-30')
#show the data
df

#Calculate the simple moving average, deviation, upper band and the lower band
#Get the time periode (20 days)
period = 20
#Calculate the simmple moving average (SMA)
df['SMA'] = df['Close'].rolling(window=period).mean()
#Get the standart deviation
df['STD'] = df['Close'].rolling(window=period).std()
#Calculate the upper bolinger band
df['Upper'] = df['SMA'] + (df['STD'] *2)
#Calculate the lowe bolinger band
df['Lower'] = df['SMA'] - (df['STD'] *2)

#Create a list of columns to keep
column_list = ['Close', 'SMA', 'Upper', 'Lower']
#Plot the data
df[column_list].plot(figsize=(12.2, 6.4))

plt.title('Bolinger Band for ADRO')
plt.ylabel('Price Rp')
#pltxlabel('Date')
plt.show()

#Plot and shade the area between the two Bollinger bands
#Get the figure and the figure size
fig = plt.figure(figsize=(12.2 , 6.4))
#Add the subplot
ax = fig.add_subplot(1,1,1)
#Get the index values of the data frame
x_axis = df.index
#Plot and shade the area between tho upper band and lower band grey
ax.fill_between(x_axis, df['Upper'], df['Lower'], color = 'grey')
#Plot the closing price and the moving average
ax.plot(x_axis, df['Close'], color = 'gold', lw =3, label = 'Close Price')
ax.plot(x_axis, df['SMA'], color = 'blue', lw =3, label = 'Simple Moving average')
#Set the title and show the image
ax.set_title('Bollinger Band for ADRO')
ax.set_xlabel('Date')
ax.set_ylabel('Price Rp')
plt.xticks(rotation = 45 )
ax.legend()
plt.show

#Creat a new data frame
new_df = df[period-1:]
#Show the new data
new_df

#Creat a function to get the buy and sell signals
def get_signal(data):
  
  buy_signal = []
  sell_signal = []

  for i in range(len(data['Close'])):
    if data['Close'][i] > data['Upper'][i]: #Then you should sell
      buy_signal.append(np.nan)
      sell_signal.append(data['Close'][i])
  
    elif data['Close'][i] < data['Lower'][i]: #Then you should buy
      buy_signal.append(data['Close'][i])
      sell_signal.append(np.nan)
    
    else :
      buy_signal.append(np.nan)
      sell_signal.append(np.nan)

  return (buy_signal, sell_signal)

#Create two new columns
new_df['Buy'] = get_signal(new_df)[0]
new_df['Sell'] = get_signal(new_df)[1]

#Plot all the data
#Get the figure and the figure size
fig = plt.figure(figsize=(12.2 , 6.4))
#Add the subplot
ax = fig.add_subplot(1,1,1)
#Get the index values of the data frame
x_axis = new_df.index
#Plot and shade the area between tho upper band and lower band grey
ax.fill_between(x_axis, new_df['Upper'], new_df['Lower'], color = 'grey')
#Plot the closing price and the moving average
ax.plot(x_axis, new_df['Close'], color = 'gold', lw =3, label = 'Close Price', alpha = 0.5)
ax.plot(x_axis, new_df['SMA'], color = 'blue', lw =3, label = 'Simple Moving average', alpha = 0.5)
ax.scatter(x_axis, new_df['Buy'], color = 'green', lw =3, label = 'Buy', marker= '^', alpha = 0.7)
ax.scatter(x_axis, new_df['Sell'], color = 'red', lw =3, label = 'Sell', marker= 'v', alpha = 0.7)
#Set the title and show the image
ax.set_title('Bollinger Band for ADRO')
ax.set_xlabel('Date')
ax.set_ylabel('Price Rp')
plt.xticks(rotation = 45 )
ax.legend()
plt.show

