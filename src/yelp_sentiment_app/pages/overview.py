import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_data, analyze_sentiment

def overview():
    st.title("📊 Overview Sentiment")

    uploaded_file = st.file_uploader("Upload Yelp Reviews (JSON or CSV)", type=["json", "csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        df['Sentiment'] = df['text'].apply(analyze_sentiment)
        df['date'] = pd.to_datetime(df['date'])

        # Summary Statistics
        total_reviews = len(df)
        pos_reviews = (df['Sentiment'] == "Positive").sum()
        neg_reviews = (df['Sentiment'] == "Negative").sum()
        neu_reviews = (df['Sentiment'] == "Neutral").sum()

        # Display Metrics
        st.write("## Summary Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Reviews", total_reviews)
        col2.metric("Positive Reviews", f"{pos_reviews} ({round((pos_reviews / total_reviews) * 100, 2)}%)")
        col3.metric("Negative Reviews", f"{neg_reviews} ({round((neg_reviews / total_reviews) * 100, 2)}%)")
        col4.metric("Neutral Reviews", f"{neu_reviews} ({round((neu_reviews / total_reviews) * 100, 2)}%)")

        # Sentiment Over Time
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

        # Sentiment Distribution
        st.write("### Sentiment Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x='Sentiment', data=df, palette='coolwarm', ax=ax)
        st.pyplot(fig)
