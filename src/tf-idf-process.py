import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

# Initialize stemmer and lemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

# Function for cleaning, tokenizing, and stemming/lemmatizing text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    
    # Apply stemming and lemmatization
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]
    
    return {
        "tokens": filtered_tokens,
        "stemmed": stemmed_tokens,
        "lemmatized": lemmatized_tokens,
    }

# Load dataset (assuming JSON file with multiple lines)
def load_dataset(file_path):
    df = pd.read_json(file_path, lines=True)
    return df

# Process dataset and apply optimized TF-IDF
def process_dataset(file_path, output_dir):
    df = load_dataset(file_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    df['lemmatized_text'] = df['processed_text'].apply(lambda x: ' '.join(x['lemmatized']))
    
    # Apply optimized TF-IDF Vectorization
    vectorizer = TfidfVectorizer(
        max_features=5000,  # Limit vocabulary size for efficiency
        sublinear_tf=True,  # Scale term frequency logarithmically
        max_df=0.95,  # Ignore very common words
        min_df=5,  # Ignore very rare words
        ngram_range=(1,2),  # Consider unigrams and bigrams
    )
    tfidf_matrix = vectorizer.fit_transform(df['lemmatized_text'])
    
    # Convert TF-IDF matrix to DataFrame efficiently
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed text with tokens, stemmed, and lemmatized versions
    df[['review_id', 'processed_text']].to_json(os.path.join(output_dir, "processed_reviews_tokens.json"), orient='records', lines=True)
    
    # Save only the lemmatized text for TF-IDF processing
    df[['review_id', 'lemmatized_text']].to_json(os.path.join(output_dir, "lemmatized_text.json"), orient='records', lines=True)
    
    # Save TF-IDF scores
    tfidf_df.to_csv(os.path.join(output_dir, "tfidf_scores.csv"), index=False)
    
    return df, tfidf_df

# Example usage
if __name__ == "__main__":
    file_path = "data/raw/reviews_2021-01.json"  # Update this with the actual dataset file path
    output_dir = "data/intermediate"  # Define output directory
    processed_df, tfidf_df = process_dataset(file_path, output_dir)
    print(processed_df[['text', 'processed_text']].head())
    print(tfidf_df.head())
