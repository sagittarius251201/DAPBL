import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import pandas as pd
import plotly.express as px

def preprocess_clustering(df):
    # Select and encode relevant columns for clustering
    features = ["Age","Gender","Monthly_Income_Bracket","Nationality","Health_Drink_Frequency",
                "Willingness_To_Try_New","Max_Price_Willing_To_Pay"]
    subset = df[features].dropna()
    for col in subset.select_dtypes(include=['object']).columns:
        subset[col] = LabelEncoder().fit_transform(subset[col])
    return subset

def show_clustering_tab(df):
    st.header("Customer Clustering")
    st.markdown("""
    Segment customers with K-means clustering. Adjust the number of clusters and explore customer personas.
    """)
    data = preprocess_clustering(df)
    inertia = []
    for k in range(2, 11):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(data)
        inertia.append(km.inertia_)

    st.subheader("Elbow Plot")
    fig = px.line(x=list(range(2,11)), y=inertia, title="Elbow Method for K Selection")
    fig.update_xaxes(title="Number of Clusters")
    fig.update_yaxes(title="Inertia")
    st.plotly_chart(fig)
    st.caption("Elbow plot helps identify the optimal number of clusters for customer segmentation.")

    # Slider for cluster count
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    data['Cluster'] = labels

    st.subheader("Customer Personas")
    persona = data.groupby('Cluster').mean().reset_index()
    st.write(persona)
    st.caption("Each row represents the average feature values for a customer segment.")

    # Download clustered data
    clustered_df = df.copy().loc[data.index]
    clustered_df['Cluster'] = labels
    st.download_button(
        label="Download Data with Cluster Labels",
        data=clustered_df.to_csv(index=False),
        file_name="clustered_health_drink_data.csv",
        mime="text/csv"
    )