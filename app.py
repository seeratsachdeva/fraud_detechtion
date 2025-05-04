import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set the page configuration once, at the top
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Static Comparison", "Dynamic Prediction"])

# ----------------------------- HOME PAGE -----------------------------
if page == "Home":
    st.title("💳 Financial Fraud Detection App")
    st.markdown("""
        Welcome to the **Financial Fraud Detection App** built using **Machine Learning** and **Streamlit**.
        
        This application helps in detecting fraudulent transactions using trained models and provides:
        - 📊 Static analysis & comparison of datasets
        - 🔎 Dynamic transaction fraud prediction
        - ✅ Visualization insights for better understanding

        **Features of the dataset:**
        - `distance_from_home`: How far from home the transaction occurred
        - `distance_from_last_transaction`: Distance from the previous transaction
        - `ratio_to_median_purchase_price`: Ratio of transaction price to typical price
        - `repeat_retailer`: If the retailer is a repeat one
        - `used_chip`, `used_pin_number`, `online_order`: Transaction methods
        - `fraud`: Label whether fraud occurred or not

        👉 Use the sidebar to explore different features of this app.
    """)

# -------------------------- STATIC COMPARISON ------------------------
elif page == "Static Comparison":
    st.title("📊 Static Dataset Comparison")

    uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"])
    uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"])

    if uploaded_file_1 and uploaded_file_2:
        df1 = pd.read_csv(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2)

        st.subheader("📁 Dataset 1 Overview")
        st.write(df1.head())
        st.write(df1.describe())

        st.subheader("📁 Dataset 2 Overview")
        st.write(df2.head())
        st.write(df2.describe())

        # Fraud vs Non-Fraud Pie Charts
        fig1, ax1 = plt.subplots()
        df1['fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax1)
        ax1.set_title("Dataset 1 - Fraud Distribution")
        st.pyplot(fig1)

        fig2, ax2 = plt.subplots()
        df2['fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax2)
        ax2.set_title("Dataset 2 - Fraud Distribution")
        st.pyplot(fig2)

        # Correlation Heatmaps
        st.subheader("🔍 Correlation Heatmap - Dataset 1")
        fig3, ax3 = plt.subplots()
        sns.heatmap(df1.corr(), annot=True, cmap="coolwarm", ax=ax3)
        st.pyplot(fig3)

        st.subheader("🔍 Correlation Heatmap - Dataset 2")
        fig4, ax4 = plt.subplots()
        sns.heatmap(df2.corr(), annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

# ------------------------- DYNAMIC PREDICTION ------------------------
elif page == "Dynamic Prediction":
    st.title("🔎 Dynamic Fraud Prediction")

    try:
        model = joblib.load("random_forest_model.pkl")
        scaler = joblib.load("scaler.pkl")
    except:
        st.error("Model or scaler not found. Please make sure 'random_forest_model.pkl' and 'scaler.pkl' are in your directory.")
    else:
        with st.form("fraud_form"):
            distance_from_home = st.number_input("Distance from Home", min_value=0.0)
            distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0)
            ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
            repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
            used_chip = st.selectbox("Used Chip", [0, 1])
            used_pin_number = st.selectbox("Used PIN Number", [0, 1])
            online_order = st.selectbox("Online Order", [0, 1])
            submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = np.array([[
                distance_from_home,
                distance_from_last_transaction,
                ratio_to_median_purchase_price,
                repeat_retailer,
                used_chip,
                used_pin_number,
                online_order
            ]])
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ Fraudulent Transaction with probability {probability:.2%}")
            else:
                st.success(f"✅ Legitimate Transaction with probability {1 - probability:.2%}")
