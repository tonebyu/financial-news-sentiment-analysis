
import pandas as pd
import talib
import os
import matplotlib.pyplot as plt

def load_stock_data(file_path):
    """Loads stock data CSV into a DataFrame with datetime index."""
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values('Date', inplace=True)
    df.set_index('Date', inplace=True)
    return df

def add_technical_indicators(df):
    """Add SMA, RSI, MACD indicators using TA-Lib."""
    df['SMA_20'] = talib.SMA(df['Close'], timeperiod=20)
    df['RSI_14'] = talib.RSI(df['Close'], timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macdsignal
    df['MACD_Hist'] = macdhist
    return df

def batch_process_stock_files(data_dir):
    """Load and process all stock CSVs in the given directory."""
    processed_data = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('_historical_data.csv'):
            stock_symbol = filename.split('_')[0].upper()
            filepath = os.path.join(data_dir, filename)
            df = load_stock_data(filepath)
            df = add_technical_indicators(df)
            processed_data[stock_symbol] = df
    return processed_data

def plot_stock_indicators(df, stock_name):
    """Plot Close price, SMA, RSI, MACD for a given stock."""
    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(df.index, df['Close'], label='Close', color='blue')
    axs[0].plot(df.index, df['SMA_20'], label='SMA 20', color='orange')
    axs[0].set_title(f'{stock_name} Price & SMA')
    axs[0].legend()

    axs[1].plot(df.index, df['RSI_14'], label='RSI (14)', color='purple')
    axs[1].axhline(70, color='red', linestyle='--', alpha=0.7)
    axs[1].axhline(30, color='green', linestyle='--', alpha=0.7)
    axs[1].set_title('Relative Strength Index (RSI)')
    axs[1].legend()

    axs[2].plot(df.index, df['MACD'], label='MACD', color='black')
    axs[2].plot(df.index, df['MACD_Signal'], label='Signal Line', color='magenta')
    axs[2].bar(df.index, df['MACD_Hist'], label='MACD Hist', color='grey')
    axs[2].set_title('MACD Indicator')
    axs[2].legend()

    plt.suptitle(f'Technical Analysis for {stock_name}', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()