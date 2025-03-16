import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# Ensure NLTK resources are downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')  # This might be missing, manually download

# Load the dataset (assuming JSON format, handle line-by-line if necessary)
file_path = r'C:\Users\moham\OneDrive\Dokumente\GitHub\techlabs-data-science-yelp\data\raw\reviews_2021-01.json'

# If the file contains multiple JSON objects per line, use lines=True
df = pd.read_json(file_path, lines=True)

# Check the actual column names in the dataset
print("Columns in the dataset:", df.columns)

# Check the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Set the correct column name for reviews after inspecting the dataset
review_column = 'text'  # Update this to 'text', as it's the correct column name

# 1. Identify unusually long reviews based on character count or word count
def flag_long_reviews(df, word_threshold=300, char_threshold=3000):
    df['long_review_word_count'] = df[review_column].apply(lambda x: len(word_tokenize(x)))
    df['long_review_char_count'] = df[review_column].apply(lambda x: len(x))
    
    long_reviews = df[(df['long_review_word_count'] > word_threshold) | (df['long_review_char_count'] > char_threshold)]
    return long_reviews

# 2. Detect repetitive patterns or spam-like content
def flag_repetitive_reviews(df, repetition_threshold=0.7):
    stop_words = set(stopwords.words('english'))
    
    def get_repetition_score(review):
        words = [word.lower() for word in word_tokenize(review) if word.isalpha()]
        filtered_words = [word for word in words if word not in stop_words]
        word_count = len(filtered_words)
        
        if word_count == 0:
            return 0
        
        word_freq = Counter(filtered_words)
        most_common_freq = word_freq.most_common(1)[0][1]
        
        # Calculate repetition score: frequency of the most common word divided by total words
        repetition_score = most_common_freq / word_count
        return repetition_score
    
    df['repetition_score'] = df[review_column].apply(get_repetition_score)
    repetitive_reviews = df[df['repetition_score'] > repetition_threshold]
    
    return repetitive_reviews

# 3. Remove or flag reviews that exceed defined criteria
def flag_outliers(df, word_threshold=300, char_threshold=3000, repetition_threshold=0.7):
    long_reviews = flag_long_reviews(df, word_threshold, char_threshold)
    repetitive_reviews = flag_repetitive_reviews(df, repetition_threshold)
    
    # Combine the two flagged datasets
    flagged_reviews = pd.concat([long_reviews, repetitive_reviews]).drop_duplicates()
    
    # Optional: Remove flagged reviews
    df_cleaned = df.drop(flagged_reviews.index)
    
    return flagged_reviews, df_cleaned

# Perform the analysis
flagged_reviews, df_cleaned = flag_outliers(df)

# Save flagged reviews and cleaned data to CSV
flagged_reviews.to_csv('flagged_reviews.csv', index=False)
df_cleaned.to_csv('cleaned_reviews.csv', index=False)

print(f"Flagged {len(flagged_reviews)} reviews as unusual.")
print(f"Cleaned dataset with {len(df_cleaned)} reviews saved as 'cleaned_reviews.csv'.")
