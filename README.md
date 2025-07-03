
# Health Drink Survey Dashboard

Interactive Streamlit dashboard for analyzing the health drink survey dataset.

## Setup

1. Clone this repository.
2. Ensure `synthetic_health_drink_survey_template.csv` is in the repo root.
3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the app:

```
streamlit run app.py
```

## Features

- **Data Visualization**: 10+ descriptive charts & insights.
- **Classification**: KNN, DT, RF, GBRT with metrics, confusion matrix, ROC, and prediction upload/download.
- **Clustering**: K-Means with elbow chart, dynamic cluster slider, persona table, and download.
- **Association Rule Mining**: Apriori with parameter filters and top-10 rules.
- **Regression**: Linear, Ridge, Lasso, and Decision Tree regressors with RMSE & R².

## Deployment on Streamlit Cloud

- Push to GitHub.
- Connect your repo in Streamlit Cloud.
- The default Data URL is already set to your GitHub raw CSV.
