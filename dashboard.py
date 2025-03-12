import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

# Load dataset
def load_data():
    file_path = "fraud test2.csv"  # Update path if needed
    df = pd.read_csv(file_path)
    return df

df = load_data()

# Streamlit dashboard
st.set_page_config(layout="wide", page_title="Credit Card Fraud Detection Dashboard")
st.title("📊 Credit Card Fraud Detection Dashboard")

# Sidebar navigation
st.sidebar.header("Navigation")
option = st.sidebar.radio("Select Analysis Type", 
                          ["Dataset Overview", "Fraud Distribution", "Transaction Amount Analysis", "Feature Correlations", "Geospatial Analysis", "Model Performance Metrics"])

# Dataset Overview
if option == "Dataset Overview":
    st.subheader("📌 Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head(10))
    st.write("### Dataset Statistics")
    st.write(df.describe())
    st.write("### Missing Values")
    st.write(df.isnull().sum())

# Fraud Distribution Analysis
elif option == "Fraud Distribution":
    st.subheader("📌 Fraud vs. Legitimate Transactions")
    fraud_counts = df['is_fraud'].value_counts()
    fraud_labels = ['Legitimate', 'Fraud']
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(x=fraud_labels, y=fraud_counts, labels={'x': 'Transaction Type', 'y': 'Count'}, title="Fraud Distribution")
        st.plotly_chart(fig)
    with col2:
        fig_pie = px.pie(names=fraud_labels, values=fraud_counts, title="Fraud vs. Legitimate Transactions Pie Chart")
        st.plotly_chart(fig_pie)

# Transaction Amount Analysis
elif option == "Transaction Amount Analysis":
    st.subheader("📌 Transaction Amount Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df['amt'], bins=50, kde=True, color='blue', ax=ax)
        ax.set_xlabel("Transaction Amount ($)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Transaction Amounts")
        st.pyplot(fig)
    with col2:
        fig_box = px.box(df, y='amt', title="Transaction Amount Box Plot")
        st.plotly_chart(fig_box)

# Feature Correlations
elif option == "Feature Correlations":
    st.subheader("📌 Feature Correlation Heatmap")
    numeric_features = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_features.corr()
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

# Geospatial Analysis
elif option == "Geospatial Analysis":
    st.subheader("📌 Geospatial Distribution of Transactions")
    col1, col2 = st.columns(2)
    with col1:
        fig_map = px.scatter_mapbox(df, lat="lat", lon="long", color="is_fraud", zoom=3, title="Customer Location Map",
                                    mapbox_style="carto-positron")
        st.plotly_chart(fig_map)
    with col2:
        fig_merch_map = px.scatter_mapbox(df, lat="merch_lat", lon="merch_long", color="is_fraud", zoom=3, title="Merchant Location Map",
                                          mapbox_style="carto-positron")
        st.plotly_chart(fig_merch_map)

# Model Performance Metrics (Placeholder for Future Model Integration)
elif option == "Model Performance Metrics":
    st.subheader("📌 Model Performance Metrics")
    st.write("Coming soon: Display model accuracy, precision, recall, and AUC scores once models are trained.")

st.sidebar.write("\n\n**Developed for Credit Card Fraud Detection Analysis**")
