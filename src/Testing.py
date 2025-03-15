import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import SelectKBest, chi2
from scipy.sparse import hstack
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import warnings
import os

# 🔹 Suppress unnecessary warnings
nltk.data.path.append("C:/Users/Marc/AppData/Roaming/nltk_data")
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        nltk.download('vader_lexicon', quiet=True)  # ✅ Silent download
    except:
        pass

sia = SentimentIntensityAnalyzer()

# 🔹 Load dataset (Updated path for reviews_2010.json)
df = pd.read_json("data/raw/reviews_2021-01.json", lines=True)

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_json(file_path, lines=True)

# 🔹 Define Multi-Class Sentiment Labels (Negative, Neutral, Positive, Very Positive)
df['sentiment'] = df['stars'].apply(lambda x: 0 if x <= 2 else (1 if x == 3 else (2 if x == 4 else 3)))  
# 0 = Negative, 1 = Neutral, 2 = Positive, 3 = Very Positive

# 🔹 Enhanced TF-IDF representation
custom_stopwords = list(set(ENGLISH_STOP_WORDS).union({"great", "good", "bad", "nice", "product", "service"}))  

vectorizer = TfidfVectorizer(
    max_features=25000,  # ✅ Increased vocabulary
    sublinear_tf=True,
    max_df=0.80,  # ✅ More strict filtering
    min_df=3,
    ngram_range=(1,3),  # ✅ Includes trigrams
    analyzer='word',
    stop_words=custom_stopwords,  
    norm='l2'
)

X_tfidf = vectorizer.fit_transform(df['text'])

# 🔹 Feature Selection (Top 20,000 Features using Chi-Square)
chi2_selector = SelectKBest(chi2, k=20000)
X_tfidf = chi2_selector.fit_transform(X_tfidf, df['sentiment'])

# 🔹 Feature Engineering - Additional Features
df['review_length'] = df['text'].apply(lambda x: len(x.split()))
df['num_exclamation'] = df['text'].apply(lambda x: x.count('!'))
df['num_commas'] = df['text'].apply(lambda x: x.count(','))
df['avg_word_length'] = df['text'].apply(lambda x: np.mean([len(word) for word in x.split()]))
df['num_capitals'] = df['text'].apply(lambda x: sum(1 for char in x if char.isupper()))
df['vader_sentiment'] = df['text'].apply(lambda x: (sia.polarity_scores(x)['compound'] + 1) / 2)

X_additional = np.array(df[['review_length', 'num_exclamation', 'num_commas', 'avg_word_length', 'num_capitals', 'vader_sentiment']])

# 🔹 Combine TF-IDF with additional features
X_final = hstack((X_tfidf, X_additional))  
y = df['sentiment'].values  

# 🔹 Split Data
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, stratify=y, random_state=42)

# 🔹 Optimize Naive Bayes Model with Hyperparameter Tuning
param_grid = {'alpha': [0.05, 0.1, 0.3, 0.5, 0.7, 1.0]}
grid_search = GridSearchCV(ComplementNB(), param_grid, cv=10, scoring='accuracy')  # ✅ Increased to 10 folds
grid_search.fit(X_train, y_train)

best_alpha = grid_search.best_params_['alpha']
nb_classifier = ComplementNB(alpha=best_alpha)
nb_classifier.fit(X_train, y_train)

# 🔹 Make Predictions
y_pred = nb_classifier.predict(X_test)

# 🔹 Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred, target_names=['Negative', 'Neutral', 'Positive', 'Very Positive']))

# 🔹 Perform Stratified Cross-Validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)  # ✅ Increased to 10 splits
cv_scores = cross_val_score(nb_classifier, X_final, y, cv=skf)
print(f"📊 Stratified Cross-validation accuracy: {cv_scores.mean():.4f}")
