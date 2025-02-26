import streamlit as st
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Download necessary NLTK resources
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Streamlit App
st.set_page_config(page_title="Yelp Sentiment Analyzer", layout="wide")  # Set wide layout
st.title("📊 Yelp Sentiment Analyzer")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Overview Sentiment", "Classification", "Extras"])

# File Upload
uploaded_file = st.file_uploader("Upload Yelp Reviews (JSON or CSV)", type=["json", "csv"])

if uploaded_file:
    # Load Data
    if uploaded_file.name.endswith(".json"):
        df = pd.read_json(uploaded_file, lines=True)
    else:
        df = pd.read_csv(uploaded_file)
    
    # Sentiment Analysis Function
    def get_sentiment(text):
        score = sia.polarity_scores(text)
        if score['compound'] >= 0.05:
            return "Positive"
        elif score['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"
    
    # Apply Sentiment Analysis
    df['Sentiment'] = df['text'].apply(get_sentiment)

    # Convert 'date' to datetime for time-based analysis
    df['date'] = pd.to_datetime(df['date'])

    # Summary Statistics
    total_reviews = len(df)
    pos_reviews = (df['Sentiment'] == "Positive").sum()
    neg_reviews = (df['Sentiment'] == "Negative").sum()
    neu_reviews = (df['Sentiment'] == "Neutral").sum()

    pos_percent = round((pos_reviews / total_reviews) * 100, 2)
    neg_percent = round((neg_reviews / total_reviews) * 100, 2)
    neu_percent = round((neu_reviews / total_reviews) * 100, 2)

    # ---- Overview Sentiment Page ----
    if page == "Overview Sentiment":
        # Metrics / Cards
        # Metrics / Cards
        st.write("## Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive Reviews", f"{pos_reviews} ({pos_percent}%)", delta=int(pos_reviews))
        col3.metric("Negative Reviews", f"{neg_reviews} ({neg_percent}%)", delta=int(neg_reviews))
        col4.metric("Neutral Reviews", f"{neu_reviews} ({neu_percent}%)", delta=int(neu_reviews))

        # Sentiment Distribution Over Time
        st.write("### Sentiment Distribution Over Time")
        time_option = st.selectbox("Select Time Interval", ["Day", "Month", "Year"])

        if time_option == "Day":
            df["time_group"] = df["date"].dt.date
        elif time_option == "Month":
            df["time_group"] = df["date"].dt.to_period("M")
        else:
            df["time_group"] = df["date"].dt.to_period("Y")

        sentiment_over_time = df.groupby(["time_group", "Sentiment"]).size().unstack().fillna(0)

        fig, ax = plt.subplots(figsize=(10, 5))
        sentiment_over_time.plot(kind="line", ax=ax, marker="o")
        plt.xlabel("Time")
        plt.ylabel("Review Count")
        plt.title(f"Sentiment Trends Over Time ({time_option})")
        plt.legend(title="Sentiment")
        st.pyplot(fig)

        # Sentiment Distribution Bar Chart
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=df, palette='coolwarm', ax=ax)
        st.pyplot(fig)

    # ---- Classification Page ----
    elif page == "Classification":
        st.write("## Sentiment Classification for Reviews")
        st.write(df[['text', 'Sentiment']].head(10))
        # Download Processed Data
        st.download_button("Download Processed Data", df.to_csv(index=False), "processed_reviews.csv", "text/csv")

    # ---- Extras Page ----
    elif page == "Extras":
        st.write("## Additional Sentiment Insights")

        # WordCloud per Sentiment
        st.write("### Most Common Words in Each Sentiment Category")
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            text = ' '.join(df[df['Sentiment'] == sentiment]['text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.write(f"#### {sentiment} Reviews WordCloud")
            st.image(wordcloud.to_array())

   