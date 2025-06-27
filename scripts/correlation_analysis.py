# correlation_analysis.py

import os
import sys
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
def convert_and_align_dates(news_data, stock_data, date):
    stock_data = stock_data.rename(columns={'Date': 'date'})
    news_data['date'] = pd.to_datetime(news_data['date'], errors='coerce', utc='True')

    # Convert date columns to datetime format, handling potential errors
    news_data['date'] = pd.to_datetime(news_data['date'], format='%Y-%m-%d').dt.date
    stock_data['date'] = pd.to_datetime(stock_data['date'], format='%Y-%m-%d').dt.date

    # Align the datasets by dates
    aligned_df = pd.merge(news_data, stock_data, on='date', how='inner')

    return aligned_df



def load_data(news_path, stock_path):

    # Load the news data
    news_data = pd.read_csv(news_path)

    # Load the stock price data
    stock_data = pd.read_csv(stock_path)

    aligned_data = convert_and_align_dates(news_data, stock_data, 'date')
        
    return aligned_data

# Perform sentiment analysis on the text data using TextBlob
def analyze_sentiment(df, text_column):
    
    # This function uses TextBlob to analyze the sentiment of the text
    def get_sentiment(text):
        analysis = TextBlob(text)
        return analysis.sentiment.polarity
    # Apply the sentiment analysis function to the specified text column
    df['sentiment'] = df[text_column].apply(get_sentiment)
    # Display the DataFrame with sentiment scores
    print(df[[text_column, 'sentiment']].head())
    
 # Plot the distribution of sentiment scores   
def sentiment_distribution_plot(aligned_data):
    
    plt.figure(figsize=(14, 8))
    sns.histplot
    plt.title('Sentiment Score Distribution')
    sns.histplot(aligned_data['sentiment'], bins=30, kde=True, color='blue', stat='density')
    plt.axvline(aligned_data['sentiment'].mean(), color='red', linestyle='dashed', linewidth=1, label='Mean Sentiment')
    plt.axvline(aligned_data['sentiment'].median(), color='green', linestyle    ='dashed', linewidth=1, label='Median Sentiment')
    plt.legend()    
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

# Calculate daily returns based on the 'Close' prices in the DataFrame
def daily_returns(aligned_data):
    """Calculate daily stock returns based on the 'Close' prices in the DataFrame.
    """
    # Calculate daily stock returns
    aligned_data['daily_return'] = aligned_data['Close'].pct_change()
    # Calculate daily sentiment returns
    aligned_data['sentiment_return'] = aligned_data['sentiment'].pct_change()      
    # Drop the first row with NaN values due to pct_change()
    aligned_data = aligned_data.dropna()

    return aligned_data

# plot daily returns
def plot_daily_returns(aligned_data):
   
    plt.figure(figsize=(14, 8))
    # Plot the daily returns over time  
    plt.plot(aligned_data['date'], aligned_data['daily_return'], marker='o', linestyle='-', color='blue', label='Daily Return')
    plt.xticks(rotation=45)
    plt.title('Daily Stock Returns Over Time')
    plt.xlabel('Date')
    plt.ylabel('Daily Return')
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

# Calculate daily sentiment and merge with stock data
def daily_sentiment_and_merge(aligned_data):
    
    # Group by date and calculate the mean sentiment score
    daily_sentiment = aligned_data.groupby('date')['sentiment'].mean().reset_index()

    # Merge with daily returns
    final_data = pd.merge(daily_sentiment, aligned_data[['date', 'daily_return']], on='date')

    return final_data

# Plot the correlation between sentiment scores and daily returns

def plot_sentiment_and_daily_return_correlation(merged_data):
  
    plt.figure(figsize=(14, 8))
    # Create a scatter plot of sentiment scores vs. daily returns
    sns.scatterplot(x='sentiment', y='daily_return', data=merged_data)
    plt.title('Correlation between Sentiment Scores and Daily Returns')
    plt.xlabel('Sentiment Score')
    plt.ylabel('Daily Return')
    plt.grid(True)
    plt.show()

# Plot a correlation heatmap

def correlation_heatmap(correlation_matrix):
    
    plt.figure(figsize=(10, 8))
    # Create a heatmap of the correlation matrix   
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap')
    plt.show()  