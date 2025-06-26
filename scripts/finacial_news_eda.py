import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Descriptive statistics

# 1.1 Basic statistics on headline length

def headline_length_stats(df, text_col='headline'):
    df['headline_length'] = df[text_col].astype(str).apply(len)
    print("\n--- Headline Length Stats ---")
    print(df['headline_length'].describe())
    return df

# 1.2 Article count by publisher

def article_count_by_publisher(df, publisher_col='publisher'):
    print("\n--- Top Publishers by Article Count ---")
    print(df[publisher_col].value_counts().head(20))
    sns.countplot(y=publisher_col, data=df, order=df[publisher_col].value_counts().iloc[:20].index)
    plt.title("Top 10 Publishers")
    plt.tight_layout()
    plt.show()

# 1.3 publication date pattern


def analyze_publication_trends(df, date_col='date', return_counts=True, plot=True):

   #Cleans, converts, and analyzes publication datetime data.
        
    #  Convert to pandas datetime format

    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    #  Drop rows where date couldn't be parsed
    df = df.dropna(subset=[date_col]).copy()

  
    #  Aggregate daily article counts
    #Extract useful components
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['weekday'] = df[date_col].dt.day_name()
    df['hour'] = df[date_col].dt.hour    
    df['date_only'] = df[date_col].dt.date

    # Group by day and count the number of articles published
   
    daily_counts = df.groupby('date_only').size().reset_index(name='article_count')
    
     

    # Plotting and analysis
    if plot:
        # Plot daily trend
        plt.figure(figsize=(12, 5))
        plt.plot(daily_counts['date_only'], daily_counts['article_count'], marker='o')
        plt.title('Publication Frequency Over Time')
        plt.xlabel('Date')
        plt.ylabel('Number of Articles')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Plot day-of-week distribution
        plt.figure(figsize=(12, 5))
        df['weekday'].value_counts().reindex([
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
        ]).plot(kind='bar', color='skyblue')
        plt.title('Article Count by Day of the Week')
        plt.ylabel('Total Articles')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    # Step 6: Return results
    return (df, daily_counts) if return_counts else df

# 2. Text analysis

import re
import nltk 
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')



import wordcloud
from wordcloud import WordCloud

import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from IPython.display import display
from collections import Counter

# 2.1 Sentiment analysis

def analyze_sentiment(df, text_col='headline'):
    """
    Perform sentiment analysis on the text column and add sentiment scores to the DataFrame.
    """
    sia = SentimentIntensityAnalyzer()
    
    # Apply sentiment analysis
    df['sentiment'] = df[text_col].apply(lambda x: sia.polarity_scores(str(x))['compound'])
    # Categorize sentiment
    df['Sentiment_Category'] = df['sentiment'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    
    # Display sentiment statistics
    print("\n--- Sentiment Analysis Stats ---")
    print(df['sentiment'].describe())
    
    # Plot sentiment distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['sentiment'], bins=30, kde=True, color='blue')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Score")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return df

def sentiment_distribution(df, sentiment_col='Sentiment_Category'):
    """
    Plot the distribution of sentiment categories.
    """
    plt.figure(figsize=(10, 5))
    sns.set_theme(style="whitegrid")
    sns.countplot(data=df, x=sentiment_col, palette='Set2')
    plt.title("Sentiment Distribution")
    plt.xlabel("Sentiment Category")
    plt.ylabel("Number of Articles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# 2.2 Word cloud generation
def generate_wordcloud(df, text_col='headline'):
    text = " ".join(df[text_col].dropna().astype(str).tolist()).lower()
    wordcloud = WordCloud(width=1000, height=500, background_color='white').generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title("WordCloud of Headlines")
    plt.tight_layout()
    plt.show()

def preprocess (text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove punctuation
    tokens = word_tokenize(text)
    filtered = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(filtered)

# Extract keywords and match known phrases
def extract_keywords_and_matches(df, text_column='headline'):
    df['cleaned'] = df[text_column].apply(preprocess)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words='english', max_df=0.85)
    X = vectorizer.fit_transform(df['cleaned'])

    avg_tfidf = X.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    tfidf_scores = pd.DataFrame({'term': terms, 'score': avg_tfidf})
    top_keywords = tfidf_scores.sort_values(by='score', ascending=False).head(20)

    important_phrases = ['fda approval', 'price target', 'earnings surprise', 'rating upgrade']
    df['matched_phrases'] = df[text_column].apply(
        lambda x: [phrase for phrase in important_phrases if phrase in x.lower()]
    )

    return top_keywords, df[[text_column, 'matched_phrases']]

### Time series analysis

def analyze_publication_frequency(df, date_col='date'):
    """
    Plot publication frequency over time and detect spikes.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    df['date_only'] = df[date_col].dt.date

    # Frequency plot
    daily_counts = df.groupby('date_only').size()
    plt.figure(figsize=(12, 5))
    daily_counts.plot()
    plt.title("Daily Publication Frequency")
    plt.xlabel("Date")
    plt.ylabel("Article Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    rolling_avg = daily_counts.rolling(window=7).mean()

    plt.figure(figsize=(12, 5))
    plt.plot(rolling_avg, label='7-Day Rolling Average', color='blue')
    plt.title("Publication Frequency with Rolling Average")
    plt.xlabel("Date")  
    plt.ylabel("Article Count")
    plt.legend()        
    plt.show()
    
   
    # Spike detection (1.5 std above mean)
    threshold = daily_counts.mean() + 1.5 * daily_counts.std()
    spikes = daily_counts[daily_counts > threshold]
    if not spikes.empty:
        print("Spikes detected on:")
        print(spikes)

    return spikes


def analyze_hourly_distribution(df, date_col='date'):
    """
    Analyze when during the day news is most published (in UTC-4 as-is).
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    df['hour'] = df[date_col].dt.hour

    plt.figure(figsize=(10, 4))
    sns.countplot(data=df, x='hour', palette='coolwarm')
    plt.title(" News Volume by Hour of Day (UTC-4)")
    plt.xlabel("Hour")
    plt.ylabel("Number of Articles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df[['hour']]

def spike_detection(df, date_col='date', threshold=1.5):
    """
    Detect spikes in publication frequency based on a threshold.
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce', format='mixed')
    df['date_only'] = df[date_col].dt.date

    daily_counts = df.groupby('date_only').size()
    mean_count = daily_counts.mean()
    std_count = daily_counts.std()

    spike_threshold = mean_count + threshold * std_count
    spikes = daily_counts[daily_counts > spike_threshold]

    if not spikes.empty:
        print("Spikes detected on:")
        print(spikes)

    return spikes

# 4. Publisher Analysis
def analyze_publishers(df, publisher_col='publisher', headline_col='headline'):
    """
    Analyze news distribution by publisher.
    """
    publisher_counts = df[publisher_col].value_counts()

    plt.figure(figsize=(12, 5))
    publisher_counts.head(10).plot(kind='bar')
    plt.title("Top 10 Publishers by Article Count")
    plt.xlabel("Publisher")
    plt.ylabel("Articles")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return publisher_counts
def publisher_sentiment(df, publisher_col='publisher', headline_col='headline'):
    publisher_news_types = df.groupby('publisher')['Sentiment_Category'].value_counts().unstack().fillna(0)

    # Plot the distribution of news types for top publishers
    publisher_counts = df[publisher_col].value_counts()
    top_publishers = publisher_counts.head(10).index
    filtered_data = publisher_news_types.loc[top_publishers]

     # Plot
    ax = filtered_data.plot(
        kind='bar',
        stacked=True,
        figsize=(14, 6),
        colormap='tab20',
        edgecolor='black')

    plt.title('News Type Distribution for Top 10 Publishers', fontsize=14)
    plt.xlabel('Publisher', fontsize=12)
    plt.ylabel('Number of Articles', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='News Type', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def extract_email_domains(df, publisher_col='publisher'):
    """
    Extract unique domains from email-style publishers.
    """
    email_publishers = df[df[publisher_col].str.contains('@', na=False)]
    email_publishers['domain'] = email_publishers[publisher_col].apply(lambda x: re.findall(r'@([\w\.-]+)', x)[0] if '@' in x else None)

    domain_counts = email_publishers['domain'].value_counts().copy()
    if not domain_counts.empty:
        print(" Most common email domains in publishers:")
        print(domain_counts.head(10))

    return domain_counts