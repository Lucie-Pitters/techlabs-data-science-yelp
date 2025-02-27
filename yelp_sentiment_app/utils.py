import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

def load_data(uploaded_file):
    """Loads data from CSV or JSON."""
    if uploaded_file.name.endswith(".json"):
        return pd.read_json(uploaded_file, lines=True)
    else:
        return pd.read_csv(uploaded_file)

def analyze_sentiment(text):
    """Classifies sentiment using VADER."""
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"
