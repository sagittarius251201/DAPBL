
# Health Drink Survey Dashboard

This Streamlit app provides interactive analysis of the health drink survey dataset, including:

- Data Visualization  
- Classification Models (KNN, Decision Tree, Random Forest, Gradient Boosting)  
- Clustering (K-Means)  
- Association Rule Mining  
- Regression (Linear, Ridge, Lasso, Decision Tree)

## Deployment

1. Push this code to a GitHub repository (e.g., https://github.com/<username>/<repo>).
2. Ensure your CSV dataset (`synthetic_health_drink_survey_template.csv`) is in the repo root.
3. On Streamlit Cloud, connect your GitHub repo and deploy the app.
4. Update the `Data URL` in the sidebar to point to the raw CSV URL.

To run locally:

```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```
