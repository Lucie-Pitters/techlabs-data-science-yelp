import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils import load_large_data, preprocess_and_analyze
from wordcloud import WordCloud
import uuid

from naive_bayes import train_naive_bayes_model

st.set_page_config(layout="wide")
st.title("📊 Yelp Reviews Sentiment Analysis")
st.text("This tool allows you to upload Yelp reviews and business data, analyze the sentiment of customer feedback, and explore visual insights such as sentiment distribution over time, word clouds, and business-specific sentiment analysis.")
st.page_link("pages/about.py", label="Learn more about our team and the project", icon="🚀")

uploaded_files = st.file_uploader("Upload Yelp Dataset Files (Business, Reviews, Users)", type=["json"], accept_multiple_files=True)

def overview(df, user_df, business_df):
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
        
        # Add user and business names to the DataFrame for display
        df = df.merge(user_df, on="user_id", how="left")
        df = df.merge(business_df, on="business_id", how="left")
        
        # Replace NaN user names with 'Anonymous'
        df['name_x'].fillna('Anonymous', inplace=True)

        # Create a new DataFrame with the necessary columns for display
        display_df = df[['text', 'Sentiment', 'name_x', 'name_y', 'stars_x', 'date']]
        display_df.columns = ['Review Text', 'Sentiment', 'User Name', 'Business Name', 'Stars', 'Date']

        # Add a custom index starting from 1
        display_df.index = display_df.index + 1

        # Function to color sentiments
        def color_sentiment(val):
            color = 'gray'  # default color for Neutral
            if val == "Positive":
                color = 'lightgreen'
            elif val == "Negative":
                color = 'tomato'
            return f'background-color: {color}'

        # Apply color formatting to the Sentiment column
        styled_df = display_df.style.applymap(color_sentiment, subset=['Sentiment'])

        # Display the DataFrame in an interactive, sortable way
        st.dataframe(styled_df, use_container_width=True)

        # Download processed data button
        st.download_button("Download Processed Data", df.to_csv(index=False), "processed_reviews.csv", "text/csv", key=f"download_{uuid.uuid4()}")


    with col2:
        
        # Group reviews by day and sentiment
        df["time_group"] = df["date"].dt.to_period("D")
        sentiment_over_time = df.groupby(["time_group", "Sentiment"]).size().unstack().fillna(0)
        sentiment_over_time.index = sentiment_over_time.index.astype(str)

        # Define the color mapping for each sentiment category
        color_mapping = {"Negative": "tomato", "Neutral": "gray", "Positive": "lightgreen"}

        # Create the plot and iterate only over the sentiments that exist in your data
        fig, ax = plt.subplots(figsize=(10, 5))
        for sentiment, color in color_mapping.items():
            if sentiment in sentiment_over_time.columns:
                ax.plot(
                sentiment_over_time.index,
                sentiment_over_time[sentiment],
                marker="o",
                markersize=2,
                color=color,
                label=sentiment,
                )
        plt.xlabel("Time")
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Review Count")
        plt.title("Sentiment Trends Over Time")
        plt.legend(title="Sentiment")
        st.pyplot(fig)

def extras(df):
    st.subheader("📌 Additional Sentiment Insights")
    st.write("### Most Common Words in Each Sentiment Category")
    
    col1, col2 = st.columns(2)
    
    # Iterate over each sentiment category
    for i, sentiment in enumerate(["Positive", "Negative", "Neutral"]):
        # Gather text for the current sentiment
        sentiment_text = ' '.join(df[df['Sentiment'] == sentiment]['text'])
        
        # Only generate and display a wordcloud if there is any text
        if sentiment_text.strip():  # if not empty after stripping whitespace
            wordcloud = WordCloud(width=800, height=400, background_color=None, mode="RGBA").generate(sentiment_text)
            if i % 2 == 0:
                with col1:
                    st.write(f"#### {sentiment} Reviews WordCloud")
                    st.image(wordcloud.to_array())
            else:
                with col2:
                    st.write(f"#### {sentiment} Reviews WordCloud")
                    st.image(wordcloud.to_array())
        # Otherwise, skip displaying this wordcloud without showing an error.


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
            overview(business_df, user_df, businesses_df)


# Dictionary to store uploaded files
datasets = {}

if uploaded_files:
    for uploaded_file in uploaded_files:
        if "yelp_academic_dataset_business.json" in uploaded_file.name:
            datasets["business"] = uploaded_file
        elif "yelp_academic_dataset_review.json" in uploaded_file.name:
            datasets["review"] = uploaded_file
        elif "yelp_academic_dataset_user.json" in uploaded_file.name:
            datasets["user"] = uploaded_file

if "review" in datasets and "business" in datasets:
    review_df = load_large_data(datasets["review"])
    
    review_df['date'] = pd.to_datetime(review_df['date'])  # Ensure date is in datetime format
    available_years = sorted(review_df['date'].dt.year.unique(), reverse=True)
    selected_year = st.selectbox("Select a Year to Analyze", available_years, index=0)
    review_df = review_df[review_df['date'].dt.year == selected_year]

    business_df = load_large_data(datasets["business"], chunksize=10000, data_type="business")
    user_df = load_large_data(datasets["user"], chunksize=10000, data_type="user")
    sentiment_method = st.selectbox("Choose Sentiment Analysis Method", ["VADER", "Naive Bayes", "Multinominal Regression"], index=0)
    
    if "vectorizer" not in st.session_state or "nb_classifier" not in st.session_state:
        with st.spinner("Training Naive Bayes model..."):
            vectorizer, nb_classifier = train_naive_bayes_model(review_df)
            st.session_state.vectorizer = vectorizer
            st.session_state.nb_classifier = nb_classifier
        st.success("Model trained and stored successfully!")

    progress_bar = st.progress(0)
    if sentiment_method == "VADER":
        review_df = preprocess_and_analyze(review_df, "VADER")
    elif sentiment_method == "Naive Bayes":
        review_df = preprocess_and_analyze(review_df, "Naive Bayes")
    elif sentiment_method == "Multinominal Regression":
        review_df = preprocess_and_analyze(review_df, "Multinominal Regression")
    else:
        st.error("Invalid method selected.")

    progress_bar.progress(100)

    # Create tabs for different analyses
    overview_tab, wordclouds_tab, business_tab = st.tabs(["Overview", "Wordclouds", "Business Search"])
    with overview_tab:
        overview(review_df, user_df, business_df)
    
    with wordclouds_tab:
        extras(review_df)
    
    with business_tab:
        business_search(business_df, review_df)