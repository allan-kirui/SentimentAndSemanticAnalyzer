from nltk.sentiment.vader import SentimentIntensityAnalyzer

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def perform_sentiment_analysis(self, text):
        sentiment_score = self.analyzer.polarity_scores(text)
        return sentiment_score['compound']
