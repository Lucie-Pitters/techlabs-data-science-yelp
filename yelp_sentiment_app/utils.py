import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import re
from multiprocessing import Pool, cpu_count
import streamlit as st
import tempfile
import json

# Download necessary NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

@st.cache_data
def load_large_data(uploaded_file, chunksize=10000, data_type="review"):
    """Load large JSON file in chunks and process iteratively."""
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json") as tmp:
        tmp.write(uploaded_file.getvalue())  
        tmp_path = tmp.name  

    data_chunks = []
    with pd.read_json(tmp_path, lines=True, chunksize=chunksize) as reader:
        for chunk in reader:
            # Ensure missing columns are handled
             # For review data, ensure the 'review_id' column is present
            if data_type == "review" and "review_id" not in chunk.columns:
                continue  
            # For business data, you might check for a different column, e.g., 'business_id'
            if data_type == "business" and "business_id" not in chunk.columns:
                continue 
            
            processed_chunk = preprocess_text(chunk)
            data_chunks.append(processed_chunk)

    return pd.concat(data_chunks, ignore_index=True) if data_chunks else pd.DataFrame()

@st.cache_data
def preprocess_and_analyze(df, method, chunksize=5000):
    """Preprocess text and perform sentiment analysis in chunks."""
    df_chunks = []
    
    with Pool(cpu_count()) as pool:
        for chunk in chunkify(df, chunksize):
            texts = chunk['text'].tolist()
            if method == "VADER":
                sentiments = list(pool.imap(analyze_sentiment_vader, texts, chunksize=1000))
            else:
                sentiments = list(pool.imap(analyze_sentiment_bayes, texts, chunksize=1000))
            chunk['Sentiment'] = pd.Categorical(sentiments, categories=["Positive", "Negative", "Neutral"])
            df_chunks.append(chunk)
    
    return pd.concat(df_chunks, ignore_index=True)

def analyze_sentiment_vader(text):
    """Classifies sentiment using VADER."""
    text = clean_text(text)
    score = sia.polarity_scores(text)
    if score['compound'] >= 0.05:
        return "Positive"
    elif score['compound'] <= -0.05:
        return "Negative"
    else:
        return "Neutral"

def analyze_sentiment_bayes(text):
    """Placeholder function for Naive Bayes classifier."""
    return "Positive"

def preprocess_text(df):
    df = handle_duplicates(df)
    # Only process the 'text' column if it exists
    if "text" in df.columns:
        df = handle_missing_values(df)
        df['text'] = df['text'].str.lower().str.replace(r'[^a-zA-Z0-9\s]', '', regex=True)
    return df


def clean_text(text):
    """Text cleaning: lowercase & remove special characters."""
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return text

def handle_duplicates(df):
    """Remove duplicate reviews based on 'review_id' if the column exists."""
    if "review_id" in df.columns:
        return df.drop_duplicates(subset=["review_id"], keep="first")
    if "business_id" in df.columns:
        return df.drop_duplicates(subset=["business_id"], keep="first")
    return df  


def handle_missing_values(df):
    """Remove empty text reviews and drop rows with missing values."""
    df = df[df["text"].str.len() > 0]
    return df.dropna()

def chunkify(df, size=5000):
    """Yield successive chunks of DataFrame of given size."""
    for i in range(0, df.shape[0], size):
        yield df.iloc[i:i + size]
