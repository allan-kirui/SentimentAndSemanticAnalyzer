from gensim import corpora, models
import pandas as pd
import re

class SemanticAnalyzer:
    def __init__(self):
        self.dictionary = None
        self.lda_model = None
        self.corpus = None

    def train_lda_model(self, processed_texts, num_topics=5):
        # Create a dictionary from the processed texts
        self.dictionary = corpora.Dictionary(processed_texts)
        
        # Convert tokenized documents into bag-of-words representation
        self.corpus = [self.dictionary.doc2bow(text) for text in processed_texts]

        # Train the LDA model
        self.lda_model = models.LdaModel(self.corpus, num_topics=num_topics, id2word=self.dictionary, passes=10)
        
    def extract_topics(self, topic_names):
        topics = []
        for topic_string in topic_names:
            # Extract words using regular expression
            words = re.findall(r'"\w+"', topic_string)
            # Remove double quotes and convert to lowercase
            words = [word.strip('"').lower() for word in words]
            topics.append(words)
        return topics

    def get_topics(self):
        # Get the topics from the LDA model
        return self.lda_model.print_topics()
    
    def get_topics_per_article(self, processed_texts):
        topics_per_article = []
        for doc in processed_texts:
            doc_bow = self.dictionary.doc2bow(doc)
            topics = self.lda_model.get_document_topics(doc_bow)
            # Convert topic IDs to topic names
            topic_names = [self.lda_model.print_topic(topic[0]) for topic in topics]
            topics_per_article.extend(self.extract_topics(topic_names))
        return topics_per_article

    def infer_topics(self, new_text):
        # Preprocess the new text
        processed_text = preprocess_text(new_text)

        # Convert the processed text into bag-of-words representation
        bow_vector = self.dictionary.doc2bow(processed_text)

        # Infer the topic distribution for the new text
        topic_distribution = self.lda_model[bow_vector]

        return topic_distribution

    def visualize_topics(self):
        # Visualize the topics using pyLDAvis
        import pyLDAvis.gensim
        
        # Prepare the visualization
        #pyLDAvis.enable_notebook()
        lda_display = pyLDAvis.gensim.prepare(self.lda_model, self.corpus, self.dictionary)
        
        # save the visualization
        pyLDAvis.save_html(lda_display, 'topics/visualisation.html')

    # Additional methods for inference or topic visualization can be added here
