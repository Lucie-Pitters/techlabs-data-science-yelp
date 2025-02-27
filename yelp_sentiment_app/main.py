import streamlit as st
import pages as pg

# Set the page configuration
st.set_page_config(page_title="Yelp Sentiment Analyzer", layout="wide")

# Initialize session state if not already set
if 'page' not in st.session_state:
    st.session_state.page = "Overview"  # Default page


# Handle the page navigation based on session state
if st.session_state.page == "Overview":
    pg.overview()
elif st.session_state.page == "Classification":
    pg.classification()
elif st.session_state.page == "Extras":
    pg.extras()
