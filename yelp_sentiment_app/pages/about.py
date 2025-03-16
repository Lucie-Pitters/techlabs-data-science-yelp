import streamlit as st


    
st.set_page_config(layout="wide")
st.title("About Us")

st.markdown("""
    This tool was created as part of the **TechLabs “Digital Shaper Program”** in Düsseldorf during the **Winter Term 2024**.
    
    ## Our Tool
    
    Welcome to our Yelp Reviews Sentiment Analysis tool! This tool provides an interactive way to analyze Yelp reviews based on sentiment. You can upload Yelp reviews data (in JSON or CSV format) and explore various metrics like sentiment classification, sentiment trends over time, and word clouds for different sentiment categories. You can also search for specific businesses and get insights based on their reviews.
    
    We used the **Yelp Academic Dataset** for this project, which you can access [here](https://business.yelp.com/data/resources/open-dataset/).
    
    Explore the full functionality and see how sentiment analysis can provide valuable insights into customer feedback!
            
     """)

st.page_link("pages/overview.py", label="Explore our tool", icon="📊")

st.markdown("""
    ## Our Team
    
    This project was developed by a diverse team of students, each bringing their unique expertise to the table:
    
    - **Linh** – Business Informatics Student  
    - **Marc** – Business Student  
    - **Lucie** – Human-Computer Interaction Student  
    - **Taqi** – Informatics Student  
    
    
 """)

