import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Load TF-IDF scores
tfidf_df = pd.read_csv("data/intermediate/tfidf_scores.csv")

# Sum TF-IDF scores per word
top_words = tfidf_df.sum().sort_values(ascending=False).head(20)

# Plot top words
plt.figure(figsize=(12,6))
top_words.plot(kind='bar', color='skyblue')
plt.title("Top 20 Important Words in Reviews (TF-IDF)")
plt.xlabel("Words")
plt.ylabel("TF-IDF Score")
plt.xticks(rotation=45)
plt.show()

word_distribution = (tfidf_df > 0).sum().sort_values(ascending=False).head(20)


#similarity_matrix = cosine_similarity(tfidf_df)
#print("Example similarity score between first two reviews:", similarity_matrix[0,1])

pos_reviews = tfidf_df[y == 1].sum().sort_values(ascending=False).head(10)
neg_reviews = tfidf_df[y == 0].sum().sort_values(ascending=False).head(10)

fig, axes = plt.subplots(1, 2, figsize=(14,6))

pos_reviews.plot(kind='bar', ax=axes[0], color='green')
axes[0].set_title("Top Words in Positive Reviews")
axes[0].set_xticklabels(pos_reviews.index, rotation=45)

neg_reviews.plot(kind='bar', ax=axes[1], color='red')
axes[1].set_title("Top Words in Negative Reviews")
axes[1].set_xticklabels(neg_reviews.index, rotation=45)

plt.show()
