import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import nltk
from collections import Counter

# Load the processed dataset with sentiment labels
file_path = "data/intermediate/reviews_with_sentiment.json"

# Read JSON file into a DataFrame
df = pd.read_json(file_path, lines=True)

### 1️⃣ Bar Chart of Sentiment Distribution ###
plt.figure(figsize=(8, 5))
sns.countplot(x=df["sentiment"], palette="coolwarm", order=["positive", "neutral", "negative"])
plt.title("Sentiment Distribution")
plt.xlabel("Sentiment Category")
plt.ylabel("Number of Reviews")
plt.show()

### 2️⃣ Compare Sentiment with Star Ratings ###
plt.figure(figsize=(10, 6))
sns.countplot(x=df["stars"], hue=df["sentiment"], palette="coolwarm")
plt.title("Sentiment vs. Star Ratings")
plt.xlabel("Star Rating")
plt.ylabel("Number of Reviews")
plt.legend(title="Sentiment")
plt.show()

### 3️⃣ WordCloud of Most Common Words in Each Sentiment Category ###
nltk.download("stopwords")
from nltk.corpus import stopwords

stop_words = set(stopwords.words("english"))

def generate_wordcloud(sentiment_label):
    text = " ".join(df[df["sentiment"] == sentiment_label]["text"])
    words = [word.lower() for word in text.split() if word.lower() not in stop_words]
    word_freq = Counter(words)
    wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most Common Words in {sentiment_label.capitalize()} Reviews")
    plt.show()

# Generate WordClouds for each sentiment category
for sentiment in ["positive", "neutral", "negative"]:
    generate_wordcloud(sentiment)
