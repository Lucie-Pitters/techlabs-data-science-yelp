import numpy as np
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Load the JSON file
file_path = "data/intermediate/processed_reviews_tokens.json"

# Read JSON file line by line
with open(file_path, "r") as file:
    data = [json.loads(line) for line in file] 

review_lengths = [len(review["processed_text"]["tokens"]) for review in data]

print("Average Review Length:", np.mean(review_lengths))
print("Median Review Length:", np.median(review_lengths))
print("Max Review Length:", np.max(review_lengths))
print("Min Review Length:", np.min(review_lengths))

all_tokens = [word for review in data for word in review["processed_text"]["tokens"]]

total_words = len(all_tokens)
unique_words = len(set(all_tokens))

print(f"Total Words: {total_words}")
print(f"Unique Words: {unique_words}")
print(f"Vocabulary Richness: {unique_words / total_words:.4f}")

# Extract all words from lemmatized tokens
all_lemmatized = [token for review in data for token in review["processed_text"]["lemmatized"]]

# Create a word frequency dictionary
word_freq = Counter(all_lemmatized)

# Generate a word cloud
wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)

# Display the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")  # Hide axes
plt.title("Most Frequent Words in Reviews")
plt.show()
