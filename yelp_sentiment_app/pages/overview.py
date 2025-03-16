import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_large_data, preprocess_and_analyze
from wordcloud import WordCloud
import uuid

st.set_page_config(layout="wide")
st.title("📊 Yelp Reviews Sentiment Analysis")
st.text("This tool allows you to upload Yelp reviews and business data, analyze the sentiment of customer feedback, and explore visual insights such as sentiment distribution over time, word clouds, and business-specific sentiment analysis.")
st.page_link("pages/about.py", label="Learn more about our team and the project", icon="🚀")

uploaded_reviews = st.file_uploader("Upload Yelp Reviews (JSON)", type=["json"])
uploaded_businesses = st.file_uploader("Upload Yelp Business Data (JSON)", type=["json"])

def overview(df):
    if df.empty:
        st.warning("No reviews available for this selection.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])  # Remove invalid dates

     #Time period
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    time_range = st.date_input("Select Time Period", [min_date, max_date], min_value=min_date, max_value=max_date)

    df = df[(df["date"] >= pd.to_datetime(time_range[0])) & (df["date"] <= pd.to_datetime(time_range[1]))]

    if df.empty:
        st.warning("No reviews in the selected time range.")
        return
    
    # Summary Statistics
    total_reviews = len(df)
    unique_users = df['user_id'].nunique()
    pos_reviews = (df['Sentiment'] == "Positive").sum()
    neg_reviews = (df['Sentiment'] == "Negative").sum()
    neu_reviews = (df['Sentiment'] == "Neutral").sum()


    # Display Metrics
    st.subheader("Summary Statistics")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Reviews", total_reviews)
    col2.metric("Unique Users", unique_users)
    col3.metric("Positive Reviews", f"{pos_reviews} ({round((pos_reviews / total_reviews) * 100, 2)}%)")
    col4.metric("Negative Reviews", f"{neg_reviews} ({round((neg_reviews / total_reviews) * 100, 2)}%)")
    col5.metric("Neutral Reviews", f"{neu_reviews} ({round((neu_reviews / total_reviews) * 100, 2)}%)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Sentiment Classification")
        
        st.write(df[['text', 'Sentiment']].head(100))
        st.download_button("Download Processed Data", df.to_csv(index=False), "processed_reviews.csv", "text/csv", key=f"download_{uuid.uuid4()}")

    with col2:
        st.write("### Sentiment Distribution Over Time")
        time_option = st.selectbox("Select Time Interval", ["Day", "Month", "Year"], key=f"download_{uuid.uuid4()}")

        df["time_group"] = df["date"].dt.to_period("M")  # Default to monthly grouping

        if time_option == "Day":
            df["time_group"] = df["date"].dt.date
        elif time_option == "Month":
            df["time_group"] = df["date"].dt.to_period("M")
        else:
            df["time_group"] = df["date"].dt.to_period("Y")

        sentiment_over_time = df.groupby(["time_group", "Sentiment"]).size().unstack().fillna(0)
        sentiment_over_time.index = sentiment_over_time.index.astype(str)  # Convert index to string

        if sentiment_over_time.empty or sentiment_over_time.isnull().values.all():
            st.warning("Not enough data to plot sentiment trends.")
            return

        # Create a transparent background for the plot
        fig, ax = plt.subplots(figsize=(10, 5))
        sentiment_over_time.plot(kind="line", ax=ax, marker="o", markersize=2)
        plt.xlabel("Time")
        plt.ylabel("Review Count")
        plt.title(f"Sentiment Trends Over Time ({time_option})")
        plt.legend(title="Sentiment")
        st.pyplot(fig)

def extras(df):
    st.subheader("📌 Additional Sentiment Insights")
    st.write("### Most Common Words in Each Sentiment Category")
    
    col1, col2 = st.columns(2)
    
    for i, sentiment in enumerate(["Positive", "Negative", "Neutral"]):
        text = ' '.join(df[df['Sentiment'] == sentiment]['text'])
        wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(text)
        if i % 2 == 0:
            with col1:
                st.write(f"#### {sentiment} Reviews WordCloud")
                st.image(wordcloud.to_array())
        else:
            with col2:
                st.write(f"#### {sentiment} Reviews WordCloud")
                st.image(wordcloud.to_array())

def business_search(businesses_df, reviews_df):
    st.subheader("🔍 Search for a Business")
    
    # Filter businesses that have at least one review
    businesses_with_reviews = businesses_df[businesses_df['business_id'].isin(reviews_df['business_id'].unique())]
    
    # Get unique business names from the filtered DataFrame
    business_names = businesses_with_reviews['name'].unique()
    
    # Select a business from the dropdown
    selected_business = st.selectbox("Select a Business", business_names, index=0)
    
    if selected_business:
        selected_business_id = businesses_with_reviews[businesses_with_reviews['name'] == selected_business]['business_id'].values[0]
        business_df = reviews_df[reviews_df['business_id'] == selected_business_id]
        
        business_info = businesses_with_reviews[businesses_with_reviews['name'] == selected_business].iloc[0]
        st.markdown(f"### 📍 {business_info['name']}")
        st.write(f"**Address:** {business_info['address']}, {business_info['city']}, {business_info['state']} {business_info['postal_code']}")
        st.write(f"**Rating:** ⭐ {business_info['stars']} ({business_info['review_count']} reviews)")
        st.write(f"**Categories:** {business_info['categories']}")
        st.write(f"**Status:** {'🟢 Open' if business_info['is_open'] else '🔴 Closed'}")

        if business_df.empty:
            st.info("⚠️ This business has no reviews yet.")
        else:
            overview(business_df)


if uploaded_reviews and uploaded_businesses:
    df = load_large_data(uploaded_reviews)

    # Extract available years from the dataset
    df['date'] = pd.to_datetime(df['date'])  # Ensure date is in datetime format
    available_years = sorted(df['date'].dt.year.unique(), reverse=True)

    # User selects a year to analyze
    selected_year = st.selectbox("Select a Year to Analyze", available_years, index=0)

    # Filter dataset by selected year
    df = df[df['date'].dt.year == selected_year]

    businesses_df = load_large_data(uploaded_businesses, chunksize=10000, data_type="business")
    sentiment_method = st.selectbox("Choose Sentiment Analysis Method", ["VADER", "Naive Bayes"], index=0)
    
    progress_bar = st.progress(0)

    if sentiment_method == "VADER":
        df = preprocess_and_analyze(df, "VADER")
    else:
        df = preprocess_and_analyze(df, "Naive Bayes")

    progress_bar.progress(100)

    # Create tabs for different analyses
    overview_tab, wordclouds_tab, business_tab = st.tabs(["Overview", "Wordclouds", "Business Search"])
    with overview_tab:
        overview(df)
    
    with wordclouds_tab:
        extras(df)
    
    with business_tab:
        business_search(businesses_df, df)
