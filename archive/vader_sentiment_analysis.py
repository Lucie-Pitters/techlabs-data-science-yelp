import json
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Download necessary VADER lexicon
nltk.download('vader_lexicon')

# Initialize Sentiment Analyzer
sia = SentimentIntensityAnalyzer()

# Load the Yelp review JSON (Assuming you have multiple reviews in a file)
file_path = "data/raw/reviews_2021-01.json"  # Update this to your actual file

with open(file_path, "r", encoding="utf-8") as f:
    reviews = [json.loads(line) for line in f]

# Function to analyze sentiment
def analyze_sentiment(text):
    scores = sia.polarity_scores(text)
    if scores["compound"] >= 0.05:
        return "positive"
    elif scores["compound"] <= -0.05:
        return "negative"
    else:
        return "neutral"

# Apply sentiment analysis
for review in reviews:
    review["sentiment"] = analyze_sentiment(review["text"])

# Convert to DataFrame for easy analysis
df = pd.DataFrame(reviews)

# Save results
df.to_json("data/intermediate/reviews_with_sentiment.json", orient="records", lines=True)

# Display sample results
print(df[["review_id", "text", "sentiment"]].head())
