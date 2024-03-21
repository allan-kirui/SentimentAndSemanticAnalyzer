# import libraries
import os
import nltk
import json
import re

from TextPreprocessor import TextPreprocessor
from SentimentAnalyzer import SentimentAnalyzer
from SemanticAnalyzer import SemanticAnalyzer


def first_time_setup():
    # download nltk corpus (first time only)
    # Set NLTK_DATA in your Python script
    nltk.data.path = ["D:\\nltk_data"]
    nltk.download('all',download_dir="D:\\nltk_data")

def load_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def main():
    data = load_data('newsArticle.json')
    text_preprocessor = TextPreprocessor()
    sentiment_analyzer = SentimentAnalyzer()
    semantic_analyzer = SemanticAnalyzer()

    processed_texts = []  # List to store tokenized documents

    for article in data:
        processed_text = text_preprocessor.preprocess_text(article['title'], article['text'])
        article['sentiment_score'] = sentiment_analyzer.perform_sentiment_analysis(processed_text)
        article['processed_text'] = processed_text
        processed_texts.append(processed_text.split())  # Split processed text into tokens and add to list

    semantic_analyzer.train_lda_model(processed_texts,num_topics=len(processed_texts))
    topics_per_article = semantic_analyzer.get_topics_per_article(processed_texts=processed_texts)
    # Assign topics to articles
    for idx, article in enumerate(data):
        article['topics'] = topics_per_article[idx]

    # Save the modified data back to the JSON file
    with open('newsArticle_with_topics.json', 'w') as file:
        json.dump(data, file)

    semantic_analyzer.visualize_topics()

    


if __name__ == "__main__":
    # first_time_setup()

    main()
    
