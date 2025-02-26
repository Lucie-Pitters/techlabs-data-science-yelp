import streamlit as st
from streamlit_navigation_bar import st_navbar

def navbar():
    return st_navbar(["Overview", "Classification", "About"])
