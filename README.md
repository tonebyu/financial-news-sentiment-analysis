Project Overview

The goal of this project is to analyze how financial news sentiment affects stock price behavior. The analysis focuses on seven major publicly traded technology companies: AAPL, AMZN, GOOG, META, MSFT, NVDA, and TSLA.

The methodology follows the CRISP-DM framework, covering data understanding, preparation, modeling, evaluation, and interpretation.

Tasks Completed

Task 1: Exploratory Data Analysis (EDA)

Examined publication trends, headline lengths, keyword frequency

Analyzed the distribution of articles by publisher and publishing day

Task 2: Technical Analysis

Applied TA-Lib indicators (SMA, RSI, MACD) to historical stock prices

Visualized price trends and technical patterns for each stock

Task 3: Sentiment and Return Correlation

Used VADER sentiment scoring on headlines

Calculated daily average sentiment and stock returns

Computed Pearson correlation to evaluate relationship (results were largely negligible)

Key Findings

Most headlines are published mid-week and clustered around certain publishers

Technical indicators like RSI and MACD offered more actionable signals than sentiment scores

Sentiment scores derived from VADER showed no significant correlation with short-term stock returns

Tools and Libraries Used

pandas, numpy, matplotlib, seaborn (data processing and visualization)

TA-Lib (technical indicators)

NLTK, TextBlob (text and sentiment analysis)

scikit-learn (feature processing and evaluation)

Git and GitHub (version control and collaboration)

How to Run the Project

Clone the repository:

git clone https://github.com/tonebyu/financial-news-sentiment-analysis.git
cd financial-news-sentiment-analysis

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate

Install the required dependencies:

pip install -r requirements.txt

Open the Jupyter notebooks:

jupyter notebook

Navigate to the notebooks/ folder and run the notebooks in order.

Author

Name: Neba Program: 10 Academy - AI Mastery Week 1 Challenge

License

This project is developed for educational use under the 10 Academy curriculum.