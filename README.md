# UAE Health Drink Market Analysis Dashboard

A Streamlit dashboard for analyzing synthetic consumer survey data for the UAE health drink market.

## Features

- **Data Visualization**: Interactive charts and tables (10+ insights).
- **Classification**: KNN, Decision Tree, Random Forest, GBRT. Compare metrics, confusion matrix, ROC, predict new data.
- **Clustering**: K-means with elbow plot, slider, personas, download cluster labels.
- **Association Rule Mining**: Apriori, top-10 rules, filter columns/parameters.
- **Regression**: Linear, Ridge, Lasso, Decision Tree, insights, feature importance.

## How to Use

1. **Clone the repo**  
   `git clone <your-repo-url>`
2. **Navigate to the folder**  
   `cd health_drink_dashboard`
3. **Install requirements**  
   `pip install -r requirements.txt`
4. **Run Streamlit**  
   `streamlit run streamlit_app.py`
5. **Deploy on Streamlit Cloud**  
   - Push to GitHub, connect on [Streamlit Cloud](https://streamlit.io/cloud), set main file to `streamlit_app.py`

## Data

The dashboard loads data directly from the [provided GitHub CSV](https://github.com/sagittarius251201/Anirudh/blob/main/health_drink_survey_uae_synthetic.csv).  
You can upload new data for prediction and download results.

## Structure

- `streamlit_app.py`: Main dashboard app
- `src/`: Feature modules (data loading, ML, clustering, ARM, regression)
- `requirements.txt`: Dependencies

## Notes

- Every chart/table includes an explanation.
- All ML models use ready-to-go sklearn pipelines.
- For ARM, select columns and set min_support/confidence as desired.

---

**Enjoy exploring UAE consumer sentiments for health drinks!**