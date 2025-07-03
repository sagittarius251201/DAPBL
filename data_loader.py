import streamlit as st
import pandas as pd
import requests

@st.cache_data
def load_data_github(url):
    try:
        df = pd.read_csv(url)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def upload_and_predict_data(predict_func, feature_cols):
    st.subheader("Upload New Data for Prediction")
    uploaded_file = st.file_uploader("Upload CSV file (without target variable)", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        predictions = predict_func(new_df[feature_cols])
        new_df['Predicted_Label'] = predictions
        st.write("Predicted Results:")
        st.dataframe(new_df)
        st.download_button(
            label="Download Predictions",
            data=new_df.to_csv(index=False),
            file_name="predicted_results.csv",
            mime="text/csv"
        )