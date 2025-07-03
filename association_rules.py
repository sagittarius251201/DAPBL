import streamlit as st
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

def show_association_tab(df):
    st.header("Association Rule Mining")
    st.markdown("""
    Discover patterns in customer features and preferences using association rules.
    """)

    # User selects columns for ARM
    columns = st.multiselect("Select columns for Association Rule Mining (binary/multichoice):", 
                             [c for c in df.columns if df[c].dtype == 'object' or 'Feature' in c or 'Concern' in c], 
                             default=['Preferred_Features', 'Health_Concerns'])
    min_supp = st.slider("Min Support", 0.01, 0.5, 0.1, 0.01)
    min_conf = st.slider("Min Confidence", 0.1, 1.0, 0.5, 0.05)

    # Prepare the data for ARM
    # For multi-choice columns, explode and one-hot encode
    arm_df = pd.DataFrame()
    for col in columns:
        if df[col].str.contains(',').any():
            mlb = df[col].str.get_dummies(sep=",")
            mlb.columns = [f"{col}_{v.strip()}" for v in mlb.columns]
            arm_df = pd.concat([arm_df, mlb], axis=1)
        else:
            arm_df = pd.concat([arm_df, pd.get_dummies(df[col], prefix=col)], axis=1)
    arm_df = arm_df.astype(int)

    # Apply Apriori
    freq_items = apriori(arm_df, min_support=min_supp, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)
    rules = rules.sort_values("lift", ascending=False).head(10)

    st.subheader("Top 10 Association Rules")
    st.write(rules[["antecedents", "consequents", "support", "confidence", "lift"]])
    st.caption("Shows the strongest associations between features/concerns/preferences.")