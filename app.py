
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

st.set_page_config(layout='wide', page_title="Health Drink Dashboard")
st.sidebar.title("Navigation")
tab = st.sidebar.radio("Go to", ["Data Visualization", "Classification", "Clustering", "Association Rule Mining", "Regression"])

# Default raw GitHub CSV URL
DATA_URL_DEFAULT = "https://raw.githubusercontent.com/sagittarius251201/DAPBL/main/synthetic_health_drink_survey_template.csv"
DATA_URL = st.sidebar.text_input("Data URL", DATA_URL_DEFAULT)

@st.cache_data
def load_data(url=None, uploaded_file=None):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading uploaded file: {e}")
            return pd.DataFrame()
    try:
        return pd.read_csv(url)
    except Exception as e:
        return pd.DataFrame()

uploaded_file = st.sidebar.file_uploader("Or upload CSV", type="csv")
df = load_data(DATA_URL, uploaded_file)

if df.empty:
    st.error("❗ No data loaded. Please check your URL or upload a CSV.")
    st.stop()

# Debug info
st.sidebar.write("Data loaded:", df.shape)

if tab == "Data Visualization":
    st.header("Data Visualization")
    # 1. Age distribution
    fig, ax = plt.subplots()
    df['Age'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    st.markdown("**Insight:** Distribution of respondents by age group.")

    # 2. Income distribution
    fig, ax = plt.subplots()
    df['Monthly_Income'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Income Bracket Distribution")
    st.pyplot(fig)
    st.markdown("**Insight:** Majority fall in mid-income ranges.")

    # 3. HealthDrinkFreq
    fig, ax = plt.subplots()
    df['HealthDrinkFreq'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Health Drink Consumption Frequency")
    st.pyplot(fig)
    st.markdown("**Insight:** Weekly/Occasional consumption dominates.")

    # 4. LikelihoodTryNewBrand
    fig, ax = plt.subplots()
    df['LikelihoodTryNewBrand'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Likelihood to Try New Brand")
    st.pyplot(fig)
    st.markdown("**Insight:** Many neutral or unlikely to try new brands.")

    # 5. Barriers table
    barriers = df['DietChallenges'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
    st.table(barriers.head(10).to_frame("Count"))
    st.markdown("**Insight:** Top dietary challenges faced by respondents.")

    # 6. Flavor preferences
    flavors = df['FlavorPreferences'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
    st.table(flavors.to_frame("Count"))
    st.markdown("**Insight:** Ranking of preferred flavors.")

    # 7. PurchaseLocation distribution
    fig, ax = plt.subplots()
    df['PurchaseLocation'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Purchase Locations")
    st.pyplot(fig)
    st.markdown("**Insight:** Supermarkets are the primary purchase channel.")

    # 8. Box plot of WillingnessToSpend
    df['WillingnessToSpend'] = df['WillingnessToSpend'].str.replace('[^0-9.]','',regex=True).astype(float)
    fig, ax = plt.subplots()
    df['WillingnessToSpend'].plot(kind='box', ax=ax)
    ax.set_title("Willingness To Spend (AED)")
    st.pyplot(fig)
    st.markdown("**Insight:** Distribution of maximum spend willingness, highlighting outliers.")

    # 9. Experience side effects
    fig, ax = plt.subplots()
    df['ExperiencedSideEffects'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Experienced Side Effects")
    st.pyplot(fig)
    st.markdown("**Insight:** Percentage reporting side effects.")

    # 10. Correlation heatmap
    numeric = df.select_dtypes(include='number')
    corr = numeric.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr)), corr.columns, rotation=90)
    plt.yticks(range(len(corr)), corr.columns)
    st.pyplot(fig)
    st.markdown("**Insight:** Correlation matrix of numeric features.")

elif tab == "Classification":
    st.header("Classification Models")
    if 'WillingToSwitch' not in df.columns:
        st.error("Target column 'WillingToSwitch' not found.")
    else:
        target = 'WillingToSwitch'
        X = df.select_dtypes(include=[np.number])
        y = df[target].map({'Yes':1,'No':0})
        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.3, random_state=42)

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }
        results = []
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred)
            })
        res_df = pd.DataFrame(results)
        st.table(res_df)

        algo = st.selectbox("Select Algorithm for Confusion Matrix", list(models.keys()))
        cm = confusion_matrix(y_test, models[algo].predict(X_test))
        fig, ax = plt.subplots()
        ax.matshow(cm, cmap='Blues')
        for (i, j), z in np.ndenumerate(cm):
            ax.text(j, i, z, ha='center', va='center')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title(f"{algo} Confusion Matrix")
        st.pyplot(fig)

        # ROC curve
        fig, ax = plt.subplots()
        for name, model in models.items():
            y_prob = model.predict_proba(X_test)[:,1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            ax.plot(fpr, tpr, label=name)
        ax.plot([0,1],[0,1],'k--')
        ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.set_title('ROC Curve')
        ax.legend()
        st.pyplot(fig)

        # Upload new data
        uploaded = st.file_uploader("Upload new data for prediction", type=['csv'])
        if uploaded:
            new_df = pd.read_csv(uploaded)
            preds = models[algo].predict(new_df.select_dtypes(include=[np.number]))
            out = new_df.copy()
            out["Prediction"] = preds
            st.dataframe(out)
            st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

elif tab == "Clustering":
    st.header("K-Means Clustering")
    num_df = df.select_dtypes(include=[np.number])
    k = st.slider("Number of clusters", 2, 10, 3)
    # Elbow chart
    inertias = [KMeans(n_clusters=i, random_state=42).fit(num_df).inertia_ for i in range(1,11)]
    fig, ax = plt.subplots()
    ax.plot(range(1,11), inertias, 'o-')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title('Elbow Method')
    st.pyplot(fig)

    km = KMeans(n_clusters=k, random_state=42).fit(num_df)
    df['Cluster'] = km.labels_
    persona = df.groupby('Cluster').mean().select_dtypes(include=[np.number])
    st.dataframe(persona)
    st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv", "text/csv")

elif tab == "Association Rule Mining":
    st.header("Association Rule Mining")
    cols = st.multiselect("Select columns", ['FlavorPreferences','DietChallenges'])
    min_sup = st.slider("Min support", 0.01, 0.5, 0.05)
    min_conf = st.slider("Min confidence", 0.1, 1.0, 0.5)
    if st.button("Run Apriori"):
        basket = df[cols].apply(lambda x: x.str.get_dummies(sep=';'))
        freq = apriori(basket, min_support=min_sup, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
        st.write(rules.sort_values('lift', ascending=False).head(10))

elif tab == "Regression":
    st.header("Regression Models")
    df['WillingnessToSpend'] = df['WillingnessToSpend'].str.replace('[^0-9.]','',regex=True).astype(float)
    X = df.select_dtypes(include=[np.number]).drop(columns=['WillingnessToSpend'], errors='ignore')
    y = df['WillingnessToSpend']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    regs = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "Tree": DecisionTreeRegressor()
    }
    reg_results = []
    for name, model in regs.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        reg_results.append({
            "Model": name,
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "R2": r2_score(y_test, pred)
        })
    reg_df = pd.DataFrame(reg_results)
    st.table(reg_df)
