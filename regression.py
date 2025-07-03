import streamlit as st
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import plotly.express as px

def preprocess_reg(df, target):
    # Simple encoding for demo
    X = df.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
    X = pd.concat([X, df.select_dtypes(include=['number'])], axis=1)
    y = df[target]
    return X, y

def show_regression_tab(df):
    st.header("Regression Analysis")
    st.markdown("""
    Predict consumer spend and explore key numeric relationships.
    Models: Linear, Ridge, Lasso, Decision Tree Regression.
    """)
    # Example: Predict Monthly_Income_AED
    target = "Monthly_Income_AED"
    X, y = preprocess_reg(df, target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Decision Tree": DecisionTreeRegressor()
    }
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        results.append([name, round(score,3)])
    st.subheader("Regression Model R² Scores")
    st.table(pd.DataFrame(results, columns=["Model","R² Score"]))
    st.caption("R² score indicates the proportion of variance in the target explained by the model.")

    # 5-7 quick insights (correlation, scatter, residuals, etc.)
    st.subheader("Insight: Income vs Max Price Willing to Pay")
    fig = px.scatter(df, x="Monthly_Income_AED", y="Max_Price_Willing_To_Pay", trendline="ols")
    st.plotly_chart(fig)
    st.caption("Shows how income relates to max price customers are willing to pay.")

    st.subheader("Insight: Willingness by Income")
    fig2 = px.box(df, x="Monthly_Income_Bracket", y="Willingness_To_Try_New")
    st.plotly_chart(fig2)
    st.caption("Higher income groups may show higher willingness to try new drinks.")

    st.subheader("Insight: Spend by Frequency")
    freq_map = {"Never":0, "Rarely":1, "Occasionally":2, "Frequently":3, "Very Frequently":4}
    df['FreqCode'] = df['Health_Drink_Frequency'].map(freq_map)
    fig3 = px.box(df, x="FreqCode", y="Monthly_Income_AED")
    st.plotly_chart(fig3)
    st.caption("Do more frequent health drink consumers have higher incomes?")

    st.subheader("Insight: Regression Residuals (Linear)")
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    st.line_chart(y_test.values - preds)
    st.caption("Residuals plot for linear regression.")

    st.subheader("Insight: Feature Importance (Decision Tree)")
    dt = DecisionTreeRegressor()
    dt.fit(X_train, y_train)
    importance = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=False)[:10]
    st.bar_chart(importance)
    st.caption("Top 10 features influencing the target variable (income).")