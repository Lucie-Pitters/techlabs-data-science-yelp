import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the TF-IDF scores
tfidf_path = "data/intermediate/tfidf_scores.csv"
tfidf_df = pd.read_csv(tfidf_path)

# Generate labels heuristically (Example: Sentiment Lexicon or Clustering)
# Placeholder: Random binary labels for now
y = np.random.choice([0, 1], size=len(tfidf_df))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(tfidf_df, y, test_size=0.2, random_state=42)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Predict on test set
y_pred = nb_classifier.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# Cross-validation
cv_scores = cross_val_score(nb_classifier, tfidf_df, y, cv=5)
print(f"Cross-validation accuracy: {cv_scores.mean():.4f}")
