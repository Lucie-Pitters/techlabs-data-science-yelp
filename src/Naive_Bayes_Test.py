# Import Necessary Libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings

# Suppress warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# Initialize Lemmatizer
lemmatizer = WordNetLemmatizer()

# Function for Text Preprocessing (Lemmatization & Cleaning)
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-z\s]', '', text)  # Remove punctuation and special characters
    tokens = word_tokenize(text)  # Tokenization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Apply lemmatization
    return " ".join(lemmatized_tokens)  # Join back into a string

# Load dataset
df = pd.read_json("data/raw/reviews_2010.json", lines=True)

# Apply preprocessing to text column
df['text'] = df['text'].apply(preprocess_text)

# Define Multi-Class Sentiment Labels
df['sentiment'] = df['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else (2 if x == 4 else 3)))

# Enhanced TF-IDF representation with expanded stopwords
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union({"great", "good", "bad", "nice", "product", "service"}))
vectorizer = TfidfVectorizer(
    max_features=150000,  
    sublinear_tf=True,
    max_df=0.7,  
    min_df=3,  
    ngram_range=(1, 2),  
    stop_words=custom_stopwords,  
    norm='l2'
)
X_tfidf = vectorizer.fit_transform(df['text'])

# Feature Selection (Optional - Can Be Disabled)
USE_CHI2_SELECTION = False  # Set to True if you want to enable feature selection

if USE_CHI2_SELECTION:
    chi2_selector = SelectKBest(chi2, k=min(25000, X_tfidf.shape[1]))  
    X_tfidf = chi2_selector.fit_transform(X_tfidf, df['sentiment'])

# Feature Engineering - Optimized Selection
df['review_length'] = df['text'].apply(lambda x: len(x.split()))
df['num_exclamation'] = df['text'].apply(lambda x: x.count('!'))
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)

# Convert additional features to a numpy array
X_additional = np.array(df[['review_length', 'num_exclamation', 'avg_word_length']])

# Combine TF-IDF with additional features
X_final = hstack((X_tfidf, X_additional))  
y = df['sentiment'].values  

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)

# Optimize Naive Bayes Model with Expanded Hyperparameter Tuning
param_grid = {'alpha': np.linspace(0.001, 2.0, 20)}  # Expanded range for better tuning
grid_search = GridSearchCV(ComplementNB(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)  
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']

# Train Optimized Naive Bayes Model
nb_classifier = ComplementNB(alpha=best_alpha)
nb_classifier.fit(X_train, y_train)

# Make Predictions
y_pred = nb_classifier.predict(X_test)

# Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive', 'Very Positive']))

# Perform Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(nb_classifier, X_final, y, cv=skf, n_jobs=-1)
print(f" Stratified Cross-validation accuracy: {cv_scores.mean():.4f}")
