import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from slang_dict import slang_dict
from textblob import Word
import string
import emoji
from api_key import YOUTUBE_API_KEY
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import os

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

api_key = YOUTUBE_API_KEY

def scrape_youtube_comments(url, max_comments):
    """Scrape comments from a YouTube video"""
    video_id = url[-11:]
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    youtube = build('youtube', 'v3', developerKey=api_key)
    df = pd.DataFrame(columns=['comment'])

    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100)  # Max results per page 
    
        while request and len(df) < max_comments:
            response = request.execute()
    
            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] 
                        for item in response['items']]
    
            # create new dataframe
            df2 = pd.DataFrame({"comment": comments})
            df = pd.concat([df, df2], ignore_index=True)
            request = youtube.commentThreads().list_next(request, response)
            time.sleep(2)

    except HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred: {e.content}")

    return df

def preprocess_text(text):
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)   # Remove URLs
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = emoji.replace_emoji(text, replace="")  # Remove emoji text
    
    tokens = word_tokenize(text)  # Tokenize the sentence
    
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    tokens = [slang_dict.get(token, token) for token in tokens]  # Handle short forms and slangs

    # Spelling correction and lemmatization using TextBlob
    lemmatizer = WordNetLemmatizer()
    corrected_tokens = []
    for token in tokens:
        word = Word(token)
        corrected_word = word.correct()  # This will handle misspellings like 'gud', 'osm'
        lemmatized_word = lemmatizer.lemmatize(token)
        corrected_tokens.append(lemmatized_word)

    corrected_tokens = [token for token in corrected_tokens if len(token) > 2]  # Select tokens of length > 2
    
    corrected_tokens = [word for word in corrected_tokens if word not in string.punctuation]   # Remove punctuation

    # Remove stopwords again after lemmatization and slang correction
    corrected_tokens = [word for word in corrected_tokens if word not in stop_words]
    text = ' '.join(corrected_tokens)

    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation again

    text = re.sub(r'\d+', '', text)  # Remove numbers again

    text = ' '.join(text.split()[:50])  # Limit the text to the first 50 words

    return text

def analyze_sentiment(text):
    """Analyze sentiment of text using VADER"""
    # Initialize VADER
    sia = SentimentIntensityAnalyzer()
    preprocessed_text = preprocess_text(text)
    sentiment_scores = sia.polarity_scores(preprocessed_text)
    return sentiment_scores['compound']

def analyze_comments(url, max_comments):
    """Scrape and analyze comments from a YouTube video"""
    # Scrape comments
    df = scrape_youtube_comments(url, max_comments)

    # Preprocess comments
    # When using apply(), Pandas automatically passes each element of the column as an argument to the function.
    df['cleaned_comment'] = df['comment'].apply(preprocess_text)

    # Analyze sentiment
    df['sentiment'] = df['cleaned_comment'].apply(analyze_sentiment)

    # Categorize sentiment
    df['sentiment_category'] = pd.cut(df['sentiment'], 
                                      bins=[-1, -0.1, 0.1, 1], 
                                      labels=['Negative', 'Neutral', 'Positive'])
    
    # Remove rows with empty cells in 'cleaned_comments' feature
    df['cleaned_comment'].replace('', np.nan, inplace=True)  # Replace empty strings with NaN
    df.dropna(subset=['cleaned_comment'], inplace=True)  # Drop rows where 'cleaned_comments' is NaN

    return df

# Disable interactive mode
plt.ioff()

def wordcloud(df):
    text = ' '.join(df['cleaned_comment'])
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(8, 4))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.title('Most Common Words in Comments')
    plt.savefig('static/wordcloud.png')
    plt.close()

def barchart(df):
    """Generates a bar plot of sentiment counts using Seaborn"""
    sentiment_counts = df['sentiment_category'].value_counts().reset_index(name='count')
    sentiment_counts.columns = ['sentiment_category', 'count']
    sentiment_counts = sentiment_counts.sort_values('count', ascending=False)

    colors = ['red', 'blue', 'green']
    sns.set_style("whitegrid")

    # Create bar plot
    plt.figure(figsize=(8, 4))
    ax = sns.barplot(x='sentiment_category', y='count', data=sentiment_counts, 
                    hue='sentiment_category', palette=colors, legend=False)

    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', fontsize=11, color='black', 
                    xytext=(0, 5), textcoords='offset points')   

    plt.title('Sentiment Distribution', fontsize=16)
    plt.xlabel('Sentiment Category', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('static/barchart.png')
    plt.close()

def piechart(df):
    sentiment_counts = df['sentiment_category'].value_counts()
    sentiment_percentages = sentiment_counts / sentiment_counts.sum() * 100

    colors = ['green', 'blue', 'red']

    # Create pie chart
    plt.figure(figsize=(8, 4))
    plt.pie(sentiment_percentages, labels=sentiment_percentages.index, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Sentiment Distribution by %', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    plt.tight_layout()
    plt.savefig('static/piechart.png')
    plt.close()

def get_summary(df):
    """Generate a summary of the sentiment analysis"""
    total_comments = len(df)
    sentiment_counts = df['sentiment_category'].value_counts()
    average_sentiment = df['sentiment'].mean()

    summary = f"""
    Total Comments: {total_comments}
    Positive Comments: {sentiment_counts.get('Positive', 0)}
    Neutral Comments: {sentiment_counts.get('Neutral', 0)}
    Negative Comments: {sentiment_counts.get('Negative', 0)}
    Average Sentiment: {average_sentiment:.2f}
    """
    return summary

# Clearing image files present due to the past searches made.
def clean_cache(directory=None):
    
    clean_path = directory
    # Only proceed if directory is not empty
    if os.listdir(clean_path):
        # Iterate over the files and remove each file
        files = os.listdir(clean_path)
        for fileName in files:
            print(fileName)
            os.remove(os.path.join(clean_path, fileName))
    print("cleaned!")

# The main function will be called from the Flask app
def main(url):
    clean_cache(directory='static')
    max_comments = 200
    df = analyze_comments(url, max_comments)
    summary = get_summary(df)
    return df, summary
        