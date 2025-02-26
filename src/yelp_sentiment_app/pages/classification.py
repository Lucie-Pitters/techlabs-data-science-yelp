import streamlit as st
from utils import load_data, analyze_sentiment

def classification():
    st.title("📝 Sentiment Classification")

    uploaded_file = st.file_uploader("Upload Yelp Reviews (JSON or CSV)", type=["json", "csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        df['Sentiment'] = df['text'].apply(analyze_sentiment)

        st.write(df[['text', 'Sentiment']].head(10))

        # Download Processed Data
        st.download_button("Download Processed Data", df.to_csv(index=False), "processed_reviews.csv", "text/csv")
