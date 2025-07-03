import streamlit as st
from src.data_loader import load_data_github, upload_and_predict_data
from src.visualizations import show_visualizations
from src.classification import show_classification_tab
from src.clustering import show_clustering_tab
from src.association_rules import show_association_tab
from src.regression import show_regression_tab

st.set_page_config(page_title="UAE Health Drink Market Dashboard", layout="wide")

st.title("UAE Health Drink Market Analysis Dashboard")
st.markdown("""
Welcome! This dashboard analyzes synthetic consumer survey data for a new health drink business in the UAE. 
Navigate through tabs for insights, machine learning, clustering, association mining, and regression analysis.
""")

# Load data from GitHub
DATA_URL = "https://raw.githubusercontent.com/sagittarius251201/Anirudh/main/health_drink_survey_uae_synthetic.csv"
df = load_data_github(DATA_URL)

# Tabs
tabs = st.tabs([
    "Data Visualization",
    "Classification",
    "Clustering",
    "Association Rule Mining",
    "Regression"
])

with tabs[0]:
    show_visualizations(df)

with tabs[1]:
    show_classification_tab(df)

with tabs[2]:
    show_clustering_tab(df)

with tabs[3]:
    show_association_tab(df)

with tabs[4]:
    show_regression_tab(df)