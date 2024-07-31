# FeelTheVid - YouTube Video Sentiment Analysis

## Overview:

This dynamic application is designed to scrape comments from any YouTube video and analyze their sentiment using advanced natural language processing techniques. Whether you're a content creator looking to gauge audience feedback or a data enthusiast interested in sentiment trends, this tool provides insightful visualizations and summaries to meet your needs.

## Features:

1. Scrape YouTube Comments: Effortlessly comments from any YouTube video.
2. Sentiment Analysis: Utilize VADER, a state-of-the-art sentiment analysis tool, to classify comments as Positive, Neutral, or Negative.
3. Data Preprocessing: Advanced text preprocessing techniques ensure accurate analysis, including handling of slang, emoji conversion, and tokenization.
4. Visualizations: Generate informative visualizations like word clouds, bar charts, and pie charts to visualize sentiment distribution.
5. User-Friendly Interface: Interactive web interface built with Flask, providing an easy way to input video URLs and view results.

## Technologies Used:

1. Python: Core language for scraping, processing, and analyzing data.
2. Flask: Web framework to create a user-friendly interface.
3. NLTK & TextBlob: Libraries for natural language processing and text preprocessing.
4. Matplotlib & Seaborn: Libraries for data visualization.
5. Google API Client: To interact with the YouTube Data API for comment extraction.

## How It Works:

1. Input YouTube URL: Enter the URL of the YouTube video you want to analyze.
2. Scrape Comments: The application extracts the specified number of comments from the video.
3. Preprocess Text: Comments are cleaned and preprocessed to remove noise and prepare for analysis.
4. Analyze Sentiment: Each comment is analyzed using VADER to determine its sentiment score.
5. Visualize Results: View comprehensive visualizations and summaries of the sentiment analysis.
