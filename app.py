
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

# Data loading
DATA_URL = st.sidebar.text_input("Data URL", "https://raw.githubusercontent.com/<username>/<repo>/main/synthetic_health_drink_survey_template.csv")
@st.cache(allow_output_mutation=True)
def load_data(url):
    return pd.read_csv(url)
df = load_data(DATA_URL)

if tab == "Data Visualization":
    st.header("Data Visualization")
    # 1: Age distribution
    fig, ax = plt.subplots()
    df['Age'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Age Distribution")
    st.pyplot(fig)
    st.markdown("**Insight:** Shows proportion of age groups.")
    # 2: Income distribution
    fig, ax = plt.subplots()
    df['Monthly_Income'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Income Bracket Distribution")
    st.pyplot(fig)
    st.markdown("**Insight:** Most respondents fall in mid-income range.")
    # 3: Health awareness
    fig, ax = plt.subplots()
    df['HealthyLifestyleImportance'].value_counts().sort_index().plot(kind='bar', ax=ax)
    ax.set_title("Healthy Lifestyle Importance")
    st.pyplot(fig)
    st.markdown("**Insight:** Majority rate health highly.")
    # 4: Consumption frequency
    fig, ax = plt.subplots()
    df['HealthDrinkFreq'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Health Drink Frequency")
    st.pyplot(fig)
    st.markdown("**Insight:** Weekly/occasional are top.")
    # 5: Purchase likelihood
    fig, ax = plt.subplots()
    if 'LikelihoodTryNewBrand' in df:
        df['LikelihoodTryNewBrand'].value_counts().plot(kind='bar', ax=ax)
        ax.set_title("Likelihood to Try New Brand")
        st.pyplot(fig)
        st.markdown("**Insight:** Many are neutral or unlikely.")
    # 6: Barriers heatmap
    barriers = df['DietChallenges'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
    st.table(barriers.head(10).to_frame("Count"))
    st.markdown("**Insight:** Top dietary challenges.")
    # 7: Flavor preferences
    flavors = df['FlavorPreferences'].str.get_dummies(sep=';').sum().sort_values(ascending=False)
    st.table(flavors.to_frame("Count"))
    st.markdown("**Insight:** Preferred flavors ranking.")
    # 8: Channel distribution
    fig, ax = plt.subplots()
    df['PurchaseLocation'].value_counts().plot(kind='bar', ax=ax)
    ax.set_title("Purchase Location")
    st.pyplot(fig)
    st.markdown("**Insight:** Supermarkets dominate.")
    # 9: Box plot price willingness
    fig, ax = plt.subplots()
    df['WillingnessToSpend'] = df['WillingnessToSpend'].replace('[^0-9.]', '', regex=True).astype(float)
    df['WillingnessToSpend'].plot(kind='box', ax=ax)
    ax.set_title("Willingness to Spend (AED)")
    st.pyplot(fig)
    st.markdown("**Insight:** Provides spend distribution and outliers.")
    # 10: Correlation heatmap
    num_df = df.select_dtypes(include='number')
    corr = num_df.corr()
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)
    st.markdown("**Insight:** Shows correlations among numeric features.")

elif tab == "Classification":
    st.header("Classification Models")
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
        results.append([
            name,
            accuracy_score(y_test, y_pred),
            precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            f1_score(y_test, y_pred)
        ])
    res_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-Score"])
    st.table(res_df)

    algo = st.selectbox("Select Algorithm for Confusion Matrix", list(models.keys()))
    cm = confusion_matrix(y_test, models[algo].predict(X_test))
    fig, ax = plt.subplots()
    ax.matshow(cm, cmap='Blues')
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, z, ha='center', va='center')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f"{algo} Confusion Matrix")
    st.pyplot(fig)

    # ROC curve
    fig, ax = plt.subplots()
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax.plot(fpr, tpr, label=name)
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.set_title('ROC Curve')
    ax.legend()
    st.pyplot(fig)

    # Upload new data
    uploaded = st.file_uploader("Upload new data for prediction", type=['csv'])
    if uploaded:
        new_df = pd.read_csv(uploaded)
        preds = {name: model.predict(new_df.select_dtypes(include=[np.number])) for name, model in models.items()}
        out = new_df.copy()
        out["Prediction"] = preds[algo]
        st.dataframe(out)
        st.download_button("Download Predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

elif tab == "Clustering":
    st.header("K-Means Clustering")
    num_data = df.select_dtypes(include=[np.number])
    slider = st.slider("Number of clusters", 2, 10, 3)
    # Elbow chart
    inertias = []
    for k in range(1, 11):
        km = KMeans(n_clusters=k, random_state=42).fit(num_data)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots()
    ax.plot(range(1, 11), inertias, 'o-')
    ax.set_xlabel('k'); ax.set_ylabel('Inertia'); ax.set_title('Elbow Method')
    st.pyplot(fig)

    km = KMeans(n_clusters=slider, random_state=42).fit(num_data)
    df['Cluster'] = km.labels_
    # Persona table
    persona = df.groupby('Cluster').mean().select_dtypes(include=[np.number])
    st.dataframe(persona)
    st.download_button("Download Clustered Data", df.to_csv(index=False), "clustered_data.csv", "text/csv")

elif tab == "Association Rule Mining":
    st.header("Association Rule Mining")
    items = st.multiselect("Select columns for Apriori", ['FlavorPreferences','DietChallenges'])
    min_support = st.slider("Min support", 0.01, 0.5, 0.05)
    min_confidence = st.slider("Min confidence", 0.1, 1.0, 0.5)
    if st.button("Run Apriori"):
        basket = df[items].apply(lambda x: x.str.get_dummies(sep=';'))
        freq = apriori(basket, min_support=min_support, use_colnames=True)
        rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
        st.write(rules.sort_values('lift', ascending=False).head(10))

elif tab == "Regression":
    st.header("Regression Models")
    # Example: predict WillingnessToSpend
    df['WillingnessToSpend'] = df['WillingnessToSpend'].replace('[^0-9.]','',regex=True).astype(float)
    X = df.select_dtypes(include=[np.number]).drop(columns=['WillingnessToSpend'])
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
        reg_results.append([
            name,
            np.sqrt(mean_squared_error(y_test, pred)),
            r2_score(y_test, pred)
        ])
    reg_df = pd.DataFrame(reg_results, columns=["Model","RMSE","R2"])
    st.table(reg_df)
