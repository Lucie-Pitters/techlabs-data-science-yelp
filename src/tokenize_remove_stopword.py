import pandas as pd
import os
import nltk
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Ensure necessary NLTK resources are downloaded
nltk.download('punkt')
nltk.download('tokenizer')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Function for cleaning and tokenizing text
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
    
    return filtered_tokens

# Load dataset (assuming JSON file with multiple lines)
def load_dataset(file_path):
    df = pd.read_json(file_path, lines=True)
    return df

# Process dataset and save output
def process_dataset(file_path, output_path):
    df = load_dataset(file_path)
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_json(output_path, orient='records', lines=True)
    return df

# Example usage
if __name__ == "__main__":
    file_path = "data/raw/reviews_2021-01.json"  # Update this with the actual dataset file path
    output_path = "data/intermediate/tokenized_review_texts.json"  # Define output file path
    processed_df = process_dataset(file_path, output_path)
    print(processed_df[['text', 'processed_text']].head())
