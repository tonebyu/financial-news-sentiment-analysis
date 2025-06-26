# correlation_analysis.py

import os
import pandas as pd
import yfinance as yf
import numpy as np
from textblob import TextBlob

from scipy.stats import pearsonr
from datetime import timedelta
from dateutil.tz import tzoffset
import seaborn as sns
import matplotlib.pyplot as plt

# --- Date Alignment ---
def align_news_to_trading_day(news_df, datetime_col='date', market_open_time='09:30:00', market_tz='US/Eastern'):
    from datetime import timedelta

    # Step 1: Parse datetimes safely with UTC mode
    news_df[datetime_col] = pd.to_datetime(news_df[datetime_col], errors='coerce', utc=True)

    # Step 2: Drop invalid rows and copy to avoid SettingWithCopyWarning
    news_df = news_df.dropna(subset=[datetime_col]).copy()

    # Step 3: Convert to desired timezone (US/Eastern)
    news_df[datetime_col] = news_df[datetime_col].dt.tz_convert(market_tz)

    # Step 4: Map to trading day
    market_open = pd.to_datetime(market_open_time).time()

    def to_trading_day(timestamp):
        return timestamp.date() if timestamp.time() <= market_open else (timestamp + timedelta(days=1)).date()

    news_df['trading_date'] = news_df[datetime_col].apply(to_trading_day)

    return news_df

# --- Sentiment Analysis ---
def apply_sentiment_analysis(news_df, text_column='headline'):
    news_df['sentiment'] = news_df[text_column].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
    return news_df

def aggregate_daily_sentiment(news_df):
    daily_sentiment = news_df.groupby(['trading_date', 'stock'])['sentiment'].mean().reset_index()
    daily_sentiment.rename(columns={'trading_date': 'date'}, inplace=True)
    return daily_sentiment

# --- Stock Returns ---
def load_stock_data(stock_csv_path):
    df = pd.read_csv(stock_csv_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    df.set_index('Date', inplace=True)
    df['return'] = df['Close'].pct_change()
    df['date'] = df.index.date
    return df

def prepare_all_stocks(stock_folder_path):
    stock_data = {}
    for filename in os.listdir(stock_folder_path):
        if filename.endswith('_historical_data.csv'):
            stock_symbol = filename.split('_')[0].upper()
            full_path = os.path.join(stock_folder_path, filename)
            df = load_stock_data(full_path)
            df['stock'] = stock_symbol
            stock_data[stock_symbol] = df
    return stock_data

# --- Correlation Analysis ---
def correlate_sentiment_returns(sentiment_df, stock_df, stock_symbol):
    sentiment_stock = sentiment_df[sentiment_df['stock'] == stock_symbol]
    merged = pd.merge(sentiment_stock, stock_df.reset_index(), on=['date', 'stock'])
    merged = merged.dropna(subset=['sentiment', 'return'])
    if len(merged) < 2:
        return None, None, len(merged)  # Return count too
    corr, pval = pearsonr(merged['sentiment'], merged['return'])
    return corr, pval, len(merged)

def correlation_matrix(stock_data, sentiment_df):
    correlations = {}
    p_values = {}
    sample_counts = {}
    for stock, df in stock_data.items():
        corr, pval, count = correlate_sentiment_returns(sentiment_df, df, stock)
        correlations[stock] = corr
        p_values[stock] = pval
        sample_counts[stock] = count
    corr_df = pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])
    pval_df = pd.DataFrame.from_dict(p_values, orient='index', columns=['P-Value'])
    count_df = pd.DataFrame.from_dict(sample_counts, orient='index', columns=['N'])
    return corr_df, pval_df, count_df

# --- Visualization ---
def plot_correlation_heatmap(corr_df):
    plt.figure(figsize=(8, 5))
    sns.heatmap(corr_df.T, annot=True, cmap='coolwarm', center=0, cbar_kws={'label': 'Pearson Correlation'})
    plt.title('Sentiment-Return Correlation per Stock')
    plt.yticks(rotation=0)
    plt.show()