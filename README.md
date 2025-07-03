
# Health Drink Survey Dashboard

This Streamlit app provides interactive analysis of the health drink survey dataset.

## Updates
- Replaced deprecated `st.cache` with `st.cache_data`.
- Added error handling in data loading with clear Streamlit error messages.

## Deployment

1. Push this code to your GitHub repository.
2. Ensure your CSV dataset is in the repo root.
3. Connect to Streamlit Cloud and deploy.
4. Update the `Data URL` in the sidebar.

To run locally:
```bash
git clone https://github.com/<username>/<repo>.git
cd <repo>
pip install -r requirements.txt
streamlit run app.py
```
