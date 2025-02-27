import json
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the JSON file
with open('data/raw/reviews_2021-01.json', 'r') as file:
    reviews = [json.loads(line) for line in file]

# Extract the text from each review
texts = [review['text'] for review in reviews]

# Initialize the CountVectorizer
vectorizer = CountVectorizer(stop_words='english', lowercase=True, max_features=1000)

# Fit and transform the texts into BoW vectors
bow_matrix = vectorizer.fit_transform(texts)

# Convert the matrix to an array for easier viewing
bow_array = bow_matrix.toarray()

# Get the feature names (words in the vocabulary)
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame for the BoW vectors
bow_df = pd.DataFrame(bow_array, columns=feature_names)

# Add the review text as a column for reference
bow_df.insert(0, 'review_text', texts)

# Save the DataFrame to a CSV file
bow_df.to_csv('data/intermediate/bow_vectors.csv', index=False)

print("Bag-of-Words vectors have been saved to 'bow_vectors.csv'.")