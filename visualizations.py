import streamlit as st
import plotly.express as px
import pandas as pd

def show_visualizations(df):
    st.header("Data Visualization & Insights")
    st.markdown("Explore key insights about UAE health drink consumers.")

    # Example filters
    col1, col2 = st.columns(2)
    with col1:
        age_filter = st.multiselect("Select Age Group(s)", df['Age'].unique(), default=list(df['Age'].unique()))
    with col2:
        gender_filter = st.multiselect("Select Gender(s)", df['Gender'].unique(), default=list(df['Gender'].unique()))

    filtered_df = df[(df['Age'].isin(age_filter)) & (df['Gender'].isin(gender_filter))]

    # 1. Pie: Health Drink Frequency
    freq_counts = filtered_df['Health_Drink_Frequency'].value_counts()
    fig1 = px.pie(values=freq_counts.values, names=freq_counts.index, title="Health Drink Consumption Frequency")
    st.plotly_chart(fig1)
    st.caption("Shows how frequently different consumer segments consume health drinks.")

    # 2. Bar: Willingness to Try by Age
    st.caption("Willingness to try a new health drink, segmented by age group.")
    fig2 = px.bar(
        filtered_df.groupby("Age")["Willingness_To_Try_New"].mean().reset_index(), 
        x="Age", y="Willingness_To_Try_New", 
        labels={"Willingness_To_Try_New": "Avg Willingness (1-5)"}
    )
    st.plotly_chart(fig2)

    # 3. Boxplot: Max Price Willing to Pay by Income
    st.caption("Distribution of max price willing to pay, by income group.")
    fig3 = px.box(filtered_df, x="Monthly_Income_Bracket", y="Monthly_Income_AED")
    st.plotly_chart(fig3)

    # 4. Bar: Preferred Features
    features = []
    for lst in filtered_df["Preferred_Features"].dropna():
        features.extend(lst.split(','))
    feat_df = pd.Series(features).value_counts().reset_index()
    feat_df.columns = ['Feature', 'Count']
    fig4 = px.bar(feat_df.head(10), x="Feature", y="Count", title="Top Preferred Features")
    st.plotly_chart(fig4)
    st.caption("Top features customers look for in a health drink.")

    # 5. Heatmap: Health Concerns vs. Barriers
    st.caption("Association between health concerns and purchasing barriers.")
    if 'Health_Concerns' in filtered_df.columns and 'Barriers_To_Purchase' in filtered_df.columns:
        import itertools
        concerns = []
        barriers = []
        for i, row in filtered_df.iterrows():
            concerns += row['Health_Concerns'].split(",")
            barriers += row['Barriers_To_Purchase'].split(",")
        crosstab = pd.crosstab(pd.Series(concerns, name="Health Concern"), pd.Series(barriers, name="Barrier"))
        st.dataframe(crosstab)

    # 6. Bar: Top Motivations to Buy
    st.caption("Motivations that drive health drink purchases.")
    motivs = []
    for lst in filtered_df["Motivation_To_Buy"].dropna():
        motivs.extend(lst.split(','))
    motiv_df = pd.Series(motivs).value_counts().reset_index()
    motiv_df.columns = ['Motivation', 'Count']
    st.bar_chart(motiv_df.set_index('Motivation'))

    # 7. Histogram: Monthly Income Distribution
    st.caption("Monthly income distribution of survey respondents.")
    st.histogram(filtered_df['Monthly_Income_AED'], bins=20)

    # 8. Pie: Interest in Tailored Drink
    st.caption("Consumer interest in a drink tailored for the Middle Eastern climate.")
    st.plotly_chart(px.pie(filtered_df, names='Interest_Tailored_Drink'))

    # 9. Table: Most Consumed Brands
    brands = []
    for lst in filtered_df["Brands_Consumed"].dropna():
        brands.extend(lst.split(','))
    brand_df = pd.Series(brands).value_counts().reset_index()
    brand_df.columns = ['Brand', 'Count']
    st.dataframe(brand_df.head(10))
    st.caption("Most popular health drink brands among respondents.")

    # 10. Download filtered data
    st.download_button(
        label="Download Filtered Data",
        data=filtered_df.to_csv(index=False),
        file_name="filtered_health_drink_data.csv",
        mime="text/csv"
    )