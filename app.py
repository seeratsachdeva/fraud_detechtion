import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Set page config (must be at the top)
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Static Dataset Comparison", "Dynamic Fraud Prediction"])

# -------------------------- Home Page --------------------------
if page == "Home":
    st.title("💳 Financial Fraud Detection App")
    st.markdown("""
    Welcome to the **Financial Fraud Detection App**.

    🔍 This application allows you to:

    - 📊 Compare two datasets to understand fraud distribution.
    - 🤖 Dynamically predict if a transaction is fraudulent.

    Upload your datasets or enter transaction details to get started.
    """)

# -------------------------- Static Dataset Comparison --------------------------
elif page == "Static Dataset Comparison":
    st.title("📊 Static Dataset Comparison")

    uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"])
    uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"])

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        df1 = pd.read_csv(uploaded_file_1)
        df2 = pd.read_csv(uploaded_file_2)

        st.write("### Dataset 1 Information")
        st.write(df1.info())
        st.write(df1.head())

        st.write("### Dataset 2 Information")
        st.write(df2.info())
        st.write(df2.head())

        st.write("### Dataset Comparison - Fraud vs Non-Fraud Distribution")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        if 'fraud' in df1.columns:
            df1['fraud'].value_counts().plot.pie(
                autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax1
            )
            ax1.set_title("Dataset 1: Fraud vs Non-Fraud")
        else:
            ax1.text(0.5, 0.5, '"fraud" column not found in Dataset 1', ha='center')
            ax1.set_axis_off()

        if 'fraud' in df2.columns:
            df2['fraud'].value_counts().plot.pie(
                autopct='%1.1f%%', labels=['Not Fraud', 'Fraud'], colors=["#66b3ff", "#ff9999"], ax=ax2
            )
            ax2.set_title("Dataset 2: Fraud vs Non-Fraud")
        else:
            ax2.text(0.5, 0.5, '"fraud" column not found in Dataset 2', ha='center')
            ax2.set_axis_off()

        st.pyplot(fig)

# -------------------------- Dynamic Fraud Prediction --------------------------
elif page == "Dynamic Fraud Prediction":
    st.title("🔮 Dynamic Fraud Prediction")

    # Load model and scaler
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        st.stop()

    st.subheader("Enter Transaction Details:")

    distance_from_home = st.number_input("Distance from Home", min_value=0.0)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)

    repeat_retailer = st.selectbox("Repeat Retailer", ["No", "Yes"])
    used_chip = st.selectbox("Used Chip", ["No", "Yes"])
    used_pin_number = st.selectbox("Used PIN Number", ["No", "Yes"])
    online_order = st.selectbox("Online Order", ["No", "Yes"])

    if st.button("Predict Transaction Fraud"):
        try:
            input_data = pd.DataFrame({
                'distance_from_home': [distance_from_home],
                'distance_from_last_transaction': [distance_from_last_transaction],
                'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
                'repeat_retailer': [1 if repeat_retailer == "Yes" else 0],
                'used_chip': [1 if used_chip == "Yes" else 0],
                'used_pin_number': [1 if used_pin_number == "Yes" else 0],
                'online_order': [1 if online_order == "Yes" else 0],
            })

            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]

            if prediction == 1:
                st.error(f"⚠️ This transaction is predicted to be FRAUDULENT with {probability*100:.2f}% probability.")
            else:
                st.success(f"✅ This transaction is predicted to be LEGITIMATE with {100 - probability*100:.2f}% probability.")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
