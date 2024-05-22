import streamlit as st
from multi_page_manager import MultiPage
from pages import home, language_capabilities

st.set_page_config(page_title="Multilingual Sentiment Analysis", layout="wide")

# Create a MultiPage instance
app = MultiPage()

# Add all your applications (pages) here
app.add_page("Home", home.app)
app.add_page("Language Capabilities", language_capabilities.app)

# The main app
app.run()
