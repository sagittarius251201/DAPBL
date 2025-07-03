import os
import sys

# Ensure project root and src directory are importable
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, mean_squared_error, r2_score
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# -- Page Config --
st.set_page_config(page_title="Health Drink Dashboard", layout="wide")

# -- Sidebar --
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", [
    "Data Visualization", "Classification", "Clustering",
    "Association Rule Mining", "Regression"
])

# -- Data Loading --
DEFAULT_URL = (
    "https://raw.githubusercontent.com/"
    "sagittarius251201/DAPBL/main/synthetic_health_drink_survey_template.csv"
)
url = st.sidebar.text_input("Data URL", DEFAULT_URL)
upload = st.sidebar.file_uploader("Or upload CSV", type=["csv"])

@st.cache_data
 def load_data(url, upload_file):
    try:
        if upload_file is not None:
            return pd.read_csv(upload_file)
        return pd.read_csv(url)
    except Exception as e:
        st.error(f"Data load error: {e}")
        return pd.DataFrame()

# Load
 df = load_data(url, upload)
if df.empty:
    st.stop()

# Show shape
st.sidebar.markdown(f"**Data shape:** {df.shape}")

# Convert spend column
if 'WillingnessToSpend' in df.columns:
    df['WillingnessToSpend'] = (
        df['WillingnessToSpend'].astype(str)
        .str.replace(r"[^0-9.]", "", regex=True)
        .astype(float)
    )

# -- Data Visualization --
if tab == "Data Visualization":
    st.header("Descriptive Insights")
    # Age
    if 'Age' in df:
        fig, ax = plt.subplots()
        df['Age'].value_counts().sort_index().plot.bar(ax=ax)
        ax.set_title("Age Distribution")
        st.pyplot(fig)
    # Income
    if 'Monthly_Income' in df:
        fig, ax = plt.subplots()
        df['Monthly_Income'].value_counts().plot.bar(ax=ax)
        ax.set_title("Income Brackets")
        st.pyplot(fig)
    # Consumption freq
    if 'HealthDrinkFreq' in df:
        fig, ax = plt.subplots()
        df['HealthDrinkFreq'].value_counts().plot.bar(ax=ax)
        ax.set_title("Consumption Frequency")
        st.pyplot(fig)
    # Likelihood to try
    if 'LikelihoodTryNewBrand' in df:
        fig, ax = plt.subplots()
        df['LikelihoodTryNewBrand'].value_counts().plot.bar(ax=ax)
        ax.set_title("Likelihood to Try New Brand")
        st.pyplot(fig)
    # Flavor prefs
    if 'FlavorPreferences' in df:
        prefs = df['FlavorPreferences'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
        st.table(prefs.to_frame('Count'))

# -- Classification --
elif tab == "Classification":
    st.header("Classification Models")
    target = 'WillingToSwitch'
    if target not in df.columns:
        st.error(f"Missing target: {target}")
        st.stop()
    sub = df[df[target].isin(['Yes','No'])]
    y = sub[target].map({'Yes':1,'No':0})
    X = sub.select_dtypes(include=[np.number]).dropna()
    y = y.loc[X.index]
    if X.empty:
        st.error("No numeric features.")
        st.stop()
    try:
        Xt, Xv, yt, yv = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)
    except Exception as e:
        st.error(f"Train-test split error: {e}")
        st.stop()
    models = {
        'KNN': KNeighborsClassifier(),
        'Tree': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'GB': GradientBoostingClassifier()
    }
    rows = []
    for n,m in models.items():
        m.fit(Xt, yt)
        p = m.predict(Xv)
        rows.append({
            'Model': n,
            'Acc': accuracy_score(yv,p),
            'Prec': precision_score(yv,p),
            'Rec': recall_score(yv,p),
            'F1': f1_score(yv,p)
        })
    st.table(pd.DataFrame(rows))
    sel = st.selectbox("Confusion matrix:", list(models.keys()))
    cm = confusion_matrix(yv, models[sel].predict(Xv))
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    for (i,j),z in np.ndenumerate(cm): ax.text(j,i,z,ha='center')
    st.pyplot(fig)
    # ROC
    fig, ax = plt.subplots()
    for n,m in models.items():
        prob = m.predict_proba(Xv)[:,1]
        fpr,tpr,_ = roc_curve(yv,prob)
        ax.plot(fpr,tpr,label=n)
    ax.plot([0,1],[0,1],'k--'); ax.legend(); st.pyplot(fig)

# -- Clustering --
elif tab == "Clustering":
    st.header("K-Means Clustering")
    nums = df.select_dtypes(include=[np.number]).dropna(axis=1)
    k = st.slider("Clusters k",2,10,3)
    inertia = [KMeans(n_clusters=i,random_state=42).fit(nums).inertia_ for i in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertia, 'o-'); st.pyplot(fig)
    km = KMeans(n_clusters=k, random_state=42).fit(nums)
    df['cluster'] = km.labels_
    st.dataframe(df.groupby('cluster').mean())
    st.download_button("Download clusters", df.to_csv(index=False),"clusters.csv")

# -- Association Rules --
elif tab == "Association Rule Mining":
    st.header("Association Mining")
    obj = st.multiselect("Cols:", df.select_dtypes(include=['object']).columns.tolist())
    s = st.slider("Min sup",0.01,0.5,0.05)
    c = st.slider("Min conf",0.1,1.0,0.5)
    if st.button("Run"):
        df_bin = df[obj].apply(lambda x: x.str.get_dummies(sep=';'))
        freq = apriori(df_bin, min_support=s, use_colnames=True)
        rules = association_rules(freq, metric='confidence', min_threshold=c)
        st.write(rules.head(10))

# -- Regression --
elif tab == "Regression":
    st.header("Regression Models")
    tgt = 'WillingnessToSpend'
    if tgt not in df.columns:
        st.error(f"Missing target: {tgt}")
        st.stop()
    y = df[tgt].dropna()
    X = df.select_dtypes(include=[np.number]).drop(columns=[tgt], errors='ignore').loc[y.index]
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.3, random_state=42)
    regs = {'Lin':LinearRegression(), 'Ridge':Ridge(), 'Lasso':Lasso(), 'Tree':DecisionTreeRegressor()}
    out=[]
    for n,m in regs.items():
        m.fit(Xt, yt)
        pr = m.predict(Xv)
        out.append({'Model':n, 'RMSE':np.sqrt(mean_squared_error(yv,pr)), 'R2':r2_score(yv,pr)})
    st.table(pd.DataFrame(out))
