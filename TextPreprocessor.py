import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.stop_words = stopwords.words('english')
        self.stop_words.extend(['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get', 'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack', 'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])
        self.lemmatizer = WordNetLemmatizer()
    
    def preprocess_text(self, title, text):
        # Remove unnecessaries from the text
        combined_text =  title + ' ' + text
        # combined_text = re.sub('\s+', ' ', combined_text)  # remove newline chars
        combined_text = re.sub(r'http\S+', '', combined_text) # remove URLs
        combined_text = re.sub(r'www\S+', '', combined_text) # remove URLs
        combined_text = re.sub("\'", "", combined_text)  # remove single quotes
        combined_text = re.sub(r'[^A-Za-z0-9]+', ' ', combined_text) # remove special characters
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