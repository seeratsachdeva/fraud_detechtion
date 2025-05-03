import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Page Configuration
st.set_page_config(page_title="Fraud Detection App", layout="centered")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a page", ["Dynamic Fraud Prediction", "Static Dataset Comparison"])

# ------------------- DYNAMIC PREDICTION PAGE -------------------

if page == "Dynamic Fraud Prediction":
    st.title("🔎 Dynamic Fraud Prediction")

    # Load the model and scaler
    model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Input fields for transaction details
    with st.form("fraud_prediction_form"):
        distance_from_home = st.number_input("Distance from Home", min_value=0.0, step=1.0)
        distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0, step=1.0)
        ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0, step=0.1)
        repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
        used_chip = st.selectbox("Used Chip", [0, 1])
        used_pin_number = st.selectbox("Used PIN Number", [0, 1])
        online_order = st.selectbox("Online Order", [0, 1])

        submit_button = st.form_submit_button(label="Predict")

    if submit_button:
        # Input data
        input_data = np.array([[
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        ]])

        # Scaling input data
        input_scaled = scaler.transform(input_data)

        # Prediction
        prediction = model.predict(input_scaled)[0]
        prediction_proba = model.predict_proba(input_scaled)[0][1]

        # Show the result
        if prediction == 1:
            st.error(f"⚠️ This transaction is **FRAUDULENT** with probability {prediction_proba:.2%}")
        else:
            st.success(f"✅ This transaction is **NOT Fraudulent** with probability {1 - prediction_proba:.2%}")


# ------------------- STATIC DATASET COMPARISON PAGE -------------------

elif page == "Static Dataset Comparison":
    st.title("📊 Static Dataset Comparison")

    # File upload form for user to upload two datasets
    uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"])  # Dataset 1 upload
    uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"])  # Dataset 2 upload

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        # Read the CSV files into DataFrames
        df1 = pd.read_csv(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2)

        # Show Dataset Information and Head
        st.write("### Dataset 1 Information")
        st.write(df1.info())
        st.write(df1.head())

        st.write("### Dataset 2 Information")
        st.write(df2.info())
        st.write(df2.head())

        # Comparison of fraud vs non-fraud distribution
        st.write("### Fraud vs Non-Fraud Distribution Comparison")

        # Create subplots
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        # Dataset 1 Fraud Distribution
        sns.countplot(x='fraud', data=df1, ax=ax[0])
        ax[0].set_title("Dataset 1: Fraud vs Non-Fraud")

        # Dataset 2 Fraud Distribution
        sns.countplot(x='fraud', data=df2, ax=ax[1])
        ax[1].set_title("Dataset 2: Fraud vs Non-Fraud")

        st.pyplot(fig)

