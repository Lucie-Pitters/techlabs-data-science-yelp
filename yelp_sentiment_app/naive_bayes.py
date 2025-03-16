import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('wordnet')

sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()

@st.cache_data
def train_naive_bayes_model(df):
    """Train a Naive Bayes model for sentiment classification."""
    df['clean_text'] = df['text'].apply(preprocess_text)
    df['sentiment'] = df['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else (2 if x == 4 else 3)))
    
    vectorizer = TfidfVectorizer(
        max_features=400000, sublinear_tf=True, max_df=0.5, min_df=3,
        ngram_range=(1,3), stop_words=["restaurant", "food", "place", "service", "menu", "great", "good", "bad", "nice", "love", "best"]
    )
    X_tfidf = vectorizer.fit_transform(df['clean_text'])
    y = df['sentiment'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)
    
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    
    nb_classifier = ComplementNB()
    nb_classifier.fit(X_train, y_train)
    # Debugging
    print("Class distribution in training data:")
    print(pd.Series(y_train).value_counts())

    sample_texts = ["Horrible experience, never coming back!", "The food was okay, nothing special.", "Absolutely loved it!"]
    sample_vectors = vectorizer.transform([preprocess_text(text) for text in sample_texts])
    predictions = nb_classifier.predict(sample_vectors)
    print("Predictions for sample texts:", predictions)

    sentiment_mapping = {0: "Negative", 1: "Neutral", 2: "Positive", 3: "Very Positive"}

    return vectorizer, nb_classifier


def analyze_sentiment_bayes(text):
    """Classifies sentiment using a trained Naive Bayes model."""
    vectorizer = st.session_state.get("vectorizer")
    nb_classifier = st.session_state.get("nb_classifier")
    
    if vectorizer is None or nb_classifier is None:
        return "Neutral"  # Default if the model hasn't been trained yet
    
    clean_text = preprocess_text(text)
    text_vectorized = vectorizer.transform([clean_text])
    sentiment_class = nb_classifier.predict(text_vectorized)[0]
    
    return ["Negative", "Neutral", "Positive", "Very Positive"][sentiment_class]


def preprocess_text(text):
    """Text preprocessing: lowercasing, removing special chars, and lemmatization."""
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(lemmatized_tokens)
