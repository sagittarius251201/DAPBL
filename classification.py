import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
import plotly.graph_objects as go

def preprocess(df):
    # Example: Classify Willingness_To_Try_New >=4 as 1 else 0
    df = df.copy()
    df = df.dropna(subset=["Willingness_To_Try_New"])
    y = (df["Willingness_To_Try_New"] >= 4).astype(int)
    # Encode categorical columns
    X = df.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
    X = X.drop(["Comments"], axis=1, errors='ignore')
    X = pd.concat([X, df.select_dtypes(include=['number'])], axis=1)
    X = X.drop(["Willingness_To_Try_New"], axis=1, errors='ignore')
    return X, y

def show_classification_tab(df):
    st.header("Classification Models")
    st.markdown("""
    This section applies K-Nearest Neighbors, Decision Tree, Random Forest, and Gradient Boosted Trees 
    to predict whether a customer is highly willing (4-5) to try a new health drink.
    """)

    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    models = {
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "GBRT": GradientBoostingClassifier(random_state=42)
    }

    results = []
    y_preds = {}
    y_probas = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_preds[name] = y_pred
        try:
            y_probas[name] = model.predict_proba(X_test)[:,1]
        except:
            y_probas[name] = np.zeros_like(y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results.append([
            name,
            round(report['accuracy'],3),
            round(report['1']['precision'],3),
            round(report['1']['recall'],3),
            round(report['1']['f1-score'],3)
        ])

    st.subheader("Model Performance Table")
    st.write(pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1-score"]))

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    algo = st.selectbox("Select Algorithm for Confusion Matrix:", list(models.keys()))
    cm = confusion_matrix(y_test, y_preds[algo])
    st.write(pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["True 0","True 1"]))
    st.caption("Shows correct and incorrect predictions by the selected model.")

    # ROC Curve
    st.subheader("ROC Curve (All Models)")
    fig = go.Figure()
    for name in models:
        fpr, tpr, _ = roc_curve(y_test, y_probas[name])
        roc_auc = auc(fpr, tpr)
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"{name} (AUC={roc_auc:.2f})"))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', name="Random", line=dict(dash='dash')))
    fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    st.plotly_chart(fig)

    # Upload new data and predict
    from src.data_loader import upload_and_predict_data
    st.subheader("Predict on New Data")
    st.caption("Upload new customer data (without target variable) to predict willingness.")
    model_for_pred = models[st.selectbox("Select model for prediction:", list(models.keys()), key="pred_model")]
    def pred_func(X_new):
        X_new_encoded = X_new.select_dtypes(include=['object']).apply(LabelEncoder().fit_transform)
        X_new_encoded = X_new_encoded.reindex(columns=X.columns, fill_value=0)
        return model_for_pred.predict(X_new_encoded)
    upload_and_predict_data(pred_func, X.columns)