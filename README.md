## Abstract:  

Our team develops a tool employing different types of models that attempts to analyze text-based patterns in given datasets from Yelp, which concerns various aspects of the Food & Beverage business on social media such as reviews, users, business, tips, check-in… The output is intended to provide categories that are meaningful to our client and help them with making important business decisions. We mainly focus on sentiment analysis in this particular project.

## Overview:
We are provided with 5 datasets that concern these things: reviews from customers, check-in time, information of the business and customers.
We target to build a tool that can help us categorize the sentiment of all the reviews that customers left for the establishment that they visited on Yelp platform. This will help our client have an overview of the reality how customers really feel about the experience they received in different restaurants, thus making informed decisions accordingly. 

## How to use our Sentiment App? 
- First we need these libraries to be installed: streamlit, streamlit-navigation-bar, nltk, pandas, matplotlib, seaborn, wordcloud.
- Next, we need to upload these 3 files to our app: yelp_academic_dataset_business.csv, yelp_academic_dataset_review.csv and yelp_academic_dataset_user.csv
- Note that since the app cannot handle files that are too large, we had to shrink down the review file to only take the data in 01.2021 so that we can upload it.

### Part 1: Notebooks
00_data_preparation.ipynb: this notebook organizes Yelp reviews by month and year, saving the results to the specified output directory.
01_data_cleaning.ipynb: this notebook cleans the review data by putting it to lowercase, removing special characters and numbers, removing duplicates and incomplete reviews. The file is then saved to data/intermediate.

02_data_exploration.ipynb: this notebook converts the json files in data/raw to csv files.
03_preprocessing.ipynb: this notebook manipulates the review texts. It removes the stopwords, tokenizes each review, then stemms and lemmatizes the tokens. Next with nltk package sentiment of each review is determined. Last a tf-idf and Bag-of-Words algorithm is implemented. The resulting files are saved to data/intermediate
04_modeling.ipynb: this notebook contains our machine learning algorithm (Naive Bayes: partially implemented, logistic regression..). It builds on the preprocessing steps (bag-of-words and tf-idf). 
05_results.ipynb: this notebook shows some visualizations for our review data and creates a metric file in data/processed

### Part 2: Sentiment App
In the terminal of the “yelp_sentiment_app” folder of the project, run: streamlit run main.py
The app will automatically opens in the web browser.
Choose the year, month of the review you want to see the analysis. 
Choose the model you want to apply to the reviews for the sentiment analysis.
The app will then load the chosen time of the review and show the sentiment analysis (positive/negative/neutral..) based on the model that you chose.

## Acknowlegment

Special thanks to Techlabs and our Mentor, Nopparat, for your relentless efforts in helping us finish the project. Without your help, this would not have been possible.
