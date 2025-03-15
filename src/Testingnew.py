# Install required packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier
from imblearn.over_sampling import SMOTE
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')

# 🔹 Load dataset and extract sentiment labels
df = pd.read_json("data/raw/reviews_2021-01.json", lines=True)

# 🔹 Define binary sentiment labels (Positive: 4-5 stars, Negative: 1-2 stars, Drop 3-star reviews)
df = df[df['stars'] != 3]  # Remove neutral reviews
df['sentiment'] = (df['stars'] >= 4).astype(int)  # Positive: 1, Negative: 0

# 🔹 Improve TF-IDF representation
vectorizer = TfidfVectorizer(
    max_features=15000,  # Increased vocabulary size for richer representation
    sublinear_tf=True,  # Logarithmic scaling of term frequency
    max_df=0.85,  # Ignore terms appearing in >85% of docs
    min_df=2,  # Ignore terms appearing in <2 docs
    ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
    analyzer='char_wb',  # Use character-level n-grams
    stop_words='english'  # Remove common words
)
X_tfidf = vectorizer.fit_transform(df['text'])  # Transform review text into numerical features

# 🔹 Feature Engineering: Add Length-based & Sentiment-based Features
df['review_length'] = df['text'].apply(lambda x: len(x.split()))
df['num_exclamation'] = df['text'].apply(lambda x: x.count('!'))

# 🔹 POS tagging features
def pos_features(text):
    words = word_tokenize(text)
    pos_tags = pos_tag(words)
    noun_count = sum(1 for word, tag in pos_tags if tag.startswith('NN'))
    verb_count = sum(1 for word, tag in pos_tags if tag.startswith('VB'))
    adj_count = sum(1 for word, tag in pos_tags if tag.startswith('JJ'))
    return noun_count, verb_count, adj_count

df[['noun_count', 'verb_count', 'adj_count']] = df['text'].apply(lambda x: pd.Series(pos_features(x)))

# 🔹 VADER Sentiment Scores
sia = SentimentIntensityAnalyzer()
df['polarity_score'] = df['text'].apply(lambda x: sia.polarity_scores(x)['compound'])

# 🔹 Combine all features
X_additional = np.array(df[['review_length', 'num_exclamation', 'noun_count', 'verb_count', 'adj_count', 'polarity_score']])
X_final = np.hstack((X_tfidf.toarray(), X_additional))
y = df['sentiment'].values  # Use actual labels

# 🔹 Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_final, y)

# 🔹 Split Data into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

# 🔹 Hyperparameter Tuning for ComplementNB
param_grid = {'alpha': [0.1, 0.3, 0.5, 0.7, 1.0]}
grid_search = GridSearchCV(ComplementNB(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
best_alpha = grid_search.best_params_['alpha']

# 🔹 Optimized Naïve Bayes Classifier
cnb = ComplementNB(alpha=best_alpha)
mnb = MultinomialNB(alpha=0.5)

# 🔹 Ensemble Voting Classifier (Combining Multiple Naïve Bayes Models)
voting_clf = VotingClassifier(
    estimators=[('cnb', cnb), ('mnb', mnb)],
    voting='soft'  # Soft voting for probability averaging
)
voting_clf.fit(X_train, y_train)

# 🔹 Make Predictions
y_pred = voting_clf.predict(X_test)

# 🔹 Evaluate Performance
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Optimized Accuracy: {accuracy:.4f}")
print(classification_report(y_test, y_pred))

# 🔹 Perform Stratified Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(voting_clf, X_resampled, y_resampled, cv=skf)
print(f"📊 Stratified Cross-validation accuracy: {cv_scores.mean():.4f}")
