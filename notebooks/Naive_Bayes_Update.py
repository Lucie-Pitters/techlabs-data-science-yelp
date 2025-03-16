# Import Necessary Libraries
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from scipy.sparse import hstack
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from imblearn.over_sampling import SMOTE
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize Lemmatizer & Sentiment Analyzer
lemmatizer = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()

# Function for Text Preprocessing (Lemmatization & Cleaning)
def preprocess_text(text):
    text = text.lower()  
    text = re.sub(r"http\S+", "", text) 
    text = re.sub(r"\d+", "", text) 
    text = re.sub(r"[^a-z\s]", "", text)  
    tokens = word_tokenize(text)  
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in tokens]  
    return " ".join(lemmatized_tokens)  

# Load dataset
df = pd.read_json("data/raw/reviews_2021-01.json", lines=True)

# Apply text preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Define Multi-Class Sentiment Labels
df['sentiment'] = df['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else (2 if x == 4 else 3)))

# Define Custom Stopwords (Convert set to list!)
custom_stopwords = ["restaurant", "food", "place", "service", "menu", "great", "good", "bad", "nice", "love", "best"]

# TF-IDF Vectorization with Optimized Parameters
vectorizer = TfidfVectorizer(
    max_features=400000,  
    sublinear_tf=True,
    max_df=0.5,  
    min_df=3,  
    ngram_range=(1,3),  
    stop_words=custom_stopwords,  
    norm='l2'
)

X_tfidf = vectorizer.fit_transform(df['clean_text'])

# Generate Additional Features
df['review_length'] = df['clean_text'].apply(lambda x: len(x.split()))
df['num_exclamation'] = df['text'].apply(lambda x: x.count('!'))
df['avg_word_length'] = df['clean_text'].apply(lambda x: np.mean([len(word) for word in x.split()]) if x.split() else 0)

# Sentiment Scores (Using VADER)
df['sentiment_pos'] = df['text'].apply(lambda x: sia.polarity_scores(x)['pos'])
df['sentiment_neg'] = df['text'].apply(lambda x: sia.polarity_scores(x)['neg'])
df['sentiment_neu'] = df['text'].apply(lambda x: sia.polarity_scores(x)['neu'])

# Convert additional features to NumPy array
X_additional = np.array(df[['review_length', 'num_exclamation', 'avg_word_length', 'sentiment_pos', 'sentiment_neg', 'sentiment_neu']])

# Combine TF-IDF with additional features
X_final = hstack((X_tfidf, X_additional))  
y = df['sentiment'].values  

# Handle Class Imbalance Using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y)

# Compute Class Weights to Adjust for Remaining Imbalance
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_resampled), y=y_resampled)
class_weight_dict = dict(zip(np.unique(y_resampled), class_weights))

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# Optimize Naive Bayes Model with Grid Search
param_grid = {'alpha': np.linspace(0.001, 2.0, 20)}  
grid_search = GridSearchCV(ComplementNB(), param_grid, cv=10, scoring='accuracy', n_jobs=-1)  
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']

# Train Optimized Naive Bayes Model with Class Weights
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
cv_scores = cross_val_score(nb_classifier, X_resampled, y_resampled, cv=skf, n_jobs=-1)
print(f" Stratified Cross-validation accuracy: {cv_scores.mean():.4f}")
