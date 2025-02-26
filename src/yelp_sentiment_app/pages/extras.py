import streamlit as st
from wordcloud import WordCloud
from utils import load_data, analyze_sentiment

def extras():
    st.title("📌 Additional Sentiment Insights")

    uploaded_file = st.file_uploader("Upload Yelp Reviews (JSON or CSV)", type=["json", "csv"])

    if uploaded_file:
        df = load_data(uploaded_file)
        df['Sentiment'] = df['text'].apply(analyze_sentiment)

        st.write("### Most Common Words in Each Sentiment Category")
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            text = ' '.join(df[df['Sentiment'] == sentiment]['text'])
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            st.write(f"#### {sentiment} Reviews WordCloud")
            st.image(wordcloud.to_array())
