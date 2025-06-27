# Importing Libraries

import pandas as pd
import numpy as np
import talib
import os
import matplotlib.pyplot as plt


# Load stock data from CSV file and set the date as index
def load_stock_data(path):
    df = pd.read_csv(path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    return df


def compute_indicators(df):
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['RSI'] = talib.RSI(df['Close'], timeperiod=14)
    df['MACD'], df['MACD_signal'], _ = talib.MACD(df['Close'])
    return df

def plot_indicators(df, title=''):
    """
    Plot key technical indicators for a stock: SMA, RSI, and MACD.
    Generates a 3-panel chart to visualize price trends and momentum signals.
    Helps in identifying market conditions like overbought/oversold or trend shifts.
    """
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(df.index, df['Close'], label='Close')
    axs[0].plot(df.index, df['SMA_20'], label='SMA 20', linestyle='--')
    axs[0].set_title(f'{title} - Price & SMA')
    axs[0].legend()

    axs[1].plot(df.index, df['RSI'], color='blue')
    axs[1].axhline(70, color='red', linestyle='--')
    axs[1].axhline(30, color='green', linestyle='--')
    axs[1].set_title('RSI')

    axs[2].plot(df.index, df['MACD'], label='MACD', color='purple')
    axs[2].plot(df.index, df['MACD_signal'], label='Signal Line', color='grey')
    axs[2].set_title('MACD')
    axs[2].legend()

    plt.tight_layout()
    plt.show()
