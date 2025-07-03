
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"])

DATA_URL = st.sidebar.text_input("Data URL", "https://raw.githubusercontent.com/<username>/<repo>/main/synthetic_health_drink_survey_template.csv")

@st.cache_data
def load_data(url):
    try:
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Error loading data from URL: {e}")
        return pd.DataFrame()

df = load_data(DATA_URL)

if df.empty:
    st.stop()

if tab == "Data Visualization":
    st.header("Data Visualization")
    # Visualizations...
    fig, ax = plt.subplots()
    df['Age'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    st.markdown("**Insight:** Shows proportion of age groups.")
    # [Other visuals unchanged...]

elif tab == "Classification":
    st.header("Classification Models")
    if 'WillingToSwitch' not in df.columns:
        st.error("Target column 'WillingToSwitch' not found.")
    else:
        # Classification code...
        target = 'WillingToSwitch'
        X = df.select_dtypes(include=[np.number])
        y = df[target].map({'Yes':1,'No':0})
        # [Rest unchanged...]

elif tab == "Clustering":
    st.header("K-Means Clustering")
    # [Unchanged...]

elif tab == "Association Rule Mining":
    st.header("Association Rule Mining")
    # [Unchanged...]

elif tab == "Regression":
    st.header("Regression Models")
    # [Unchanged...]
