# import libraries
import pandas as pd
import nltk
import os
import json
import re

# from pyspark.sql import SparkSession
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def first_time_setup():
    # download nltk corpus (first time only)
    # Set NLTK_DATA in your Python script
    nltk.data.path = ["D:\\nltk_data"]
    nltk.download('all',download_dir="D:\\nltk_data")


# create preprocess_text function
def preprocess_text(title, text):
    # Remove unnecessaries from the text
    combined_text =  title + ' ' + text
    combined_text = re.sub(r'http\S+', '', combined_text)
    combined_text = re.sub(r'www\S+', '', combined_text)
    combined_text = re.sub(r'[^A-Za-z0-9]+', ' ', combined_text)
    combined_text = combined_text.lower()

    tokens = word_tokenize(combined_text.lower())

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join the tokens back into a string
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

# create get_sentiment function
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    # sentiment = 1 if scores['pos'] > 0 else 0
    sentiment = scores['compound']
    return sentiment

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def main():
    data = load_data('newsArticle.json')
    for article in data:
        article['processed_text'] = preprocess_text(article['title'], article['text'])


if __name__ == "__main__":
    # first_time_setup()
    # create SentimentIntensityAnalyzer object
    analyzer = SentimentIntensityAnalyzer()
    main()
    print("Done!")
