import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import StandardScaler

# Set page config (only once)
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Load the saved model and scaler
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("scaler.pkl")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Static Dataset Comparison", "Dynamic Prediction"])

# Home page
if page == "Home":
    st.title("💳 Financial Fraud Detection Using Big Data Analytics")
    st.markdown("""
        Welcome to the Financial Fraud Detection App.

        **Sections:**
        - 📊 **Static Dataset Comparison**: Upload and compare two datasets side by side with fraud distribution and advanced visualizations.
        - 🤖 **Dynamic Prediction**: Enter transaction details and predict whether it is fraudulent or not using a trained ML model.

        This app is based on the research paper: **"Financial Fraud Detection Using Big Data Analytics"**.
    """)

# Static Dataset Comparison Page
elif page == "Static Dataset Comparison":
    st.title("📊 Static Dataset Comparison")

    uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"], key="file1")
    uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"], key="file2")

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        df1 = pd.read_csv(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2)

        st.subheader("Dataset 1 Preview")
        st.write(df1.head())

        st.subheader("Dataset 2 Preview")
        st.write(df2.head())

        # Show fraud distribution pie charts
        st.subheader("Fraud vs Non-Fraud Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if 'fraud' in df1.columns:
            df1['fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax1)
            ax1.set_title("Dataset 1")
        else:
            ax1.text(0.5, 0.5, 'fraud column not found', ha='center')

        if 'fraud' in df2.columns:
            df2['fraud'].value_counts().plot.pie(autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax2)
            ax2.set_title("Dataset 2")
        else:
            ax2.text(0.5, 0.5, 'fraud column not found', ha='center')

        st.pyplot(fig)

        # Feature selection and chart type
        st.subheader("📈 Feature Comparison")
        selected_feature = st.selectbox("Select Feature to Compare", df1.columns)
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Boxplot", "Violinplot", "KDE Plot"])

        fig, ax = plt.subplots(figsize=(10, 5))
        if chart_type == "Histogram":
            sns.histplot(df1[selected_feature], color='blue', label='Dataset 1', kde=False, ax=ax)
            sns.histplot(df2[selected_feature], color='orange', label='Dataset 2', kde=False, ax=ax)
        elif chart_type == "Boxplot":
            combined = pd.concat([
                pd.DataFrame({selected_feature: df1[selected_feature], 'Dataset': 'Dataset 1'}),
                pd.DataFrame({selected_feature: df2[selected_feature], 'Dataset': 'Dataset 2'})
            ])
            sns.boxplot(x='Dataset', y=selected_feature, data=combined, ax=ax)
        elif chart_type == "Violinplot":
            combined = pd.concat([
                pd.DataFrame({selected_feature: df1[selected_feature], 'Dataset': 'Dataset 1'}),
                pd.DataFrame({selected_feature: df2[selected_feature], 'Dataset': 'Dataset 2'})
            ])
            sns.violinplot(x='Dataset', y=selected_feature, data=combined, ax=ax)
        elif chart_type == "KDE Plot":
            sns.kdeplot(df1[selected_feature], label='Dataset 1', ax=ax, fill=True)
            sns.kdeplot(df2[selected_feature], label='Dataset 2', ax=ax, fill=True)

        ax.set_title(f"{chart_type} of {selected_feature}")
        ax.legend()
        st.pyplot(fig)

# Dynamic Prediction Page
elif page == "Dynamic Prediction":
    st.title("🔍 Dynamic Fraud Detection")

    st.subheader("Enter Transaction Details")
    
    distance_from_home = st.number_input("Distance from Home", min_value=0.0)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
    repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
    used_chip = st.selectbox("Used Chip", [0, 1])
    used_pin_number = st.selectbox("Used PIN Number", [0, 1])
    online_order = st.selectbox("Online Order", [0, 1])

    if st.button("Predict Fraud"):
        user_data = np.array([
            distance_from_home,
            distance_from_last_transaction,
            ratio_to_median_purchase_price,
            repeat_retailer,
            used_chip,
            used_pin_number,
            online_order
        ]).reshape(1, -1)

        user_data_scaled = scaler.transform(user_data)
        prediction = model.predict(user_data_scaled)[0]
        prediction_proba = model.predict_proba(user_data_scaled)[0][1]

        if prediction == 1:
            st.error(f"⚠️ Transaction is Fraudulent with {prediction_proba*100:.2f}% confidence")
        else:
            st.success(f"✅ Transaction is NOT Fraudulent with {(1-prediction_proba)*100:.2f}% confidence")
