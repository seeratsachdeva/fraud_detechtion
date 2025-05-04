import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# This must be at the very top
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Title
st.title("📊 Static Dataset Comparison")

# Upload CSVs
uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"])
uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    try:
        df1 = pd.read_csv(uploaded_file_1)
        st.success("Dataset 1 loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Dataset 1: {e}")

    try:
        df2 = pd.read_csv(uploaded_file_2)
        st.success("Dataset 2 loaded successfully!")
    except Exception as e:
        st.error(f"Error loading Dataset 2: {e}")

    # Show Dataset Overviews
    st.subheader("Dataset 1 Overview")
    st.write(df1.head())
    st.write(df1.describe())

    st.subheader("Dataset 2 Overview")
    st.write(df2.head())
    st.write(df2.describe())

    # Fraud distribution pie charts
    st.subheader("Fraud vs Non-Fraud Distribution")

    col1, col2 = st.columns(2)
    with col1:
        if 'fraud' in df1.columns:
            fig1, ax1 = plt.subplots()
            df1['fraud'].value_counts().plot.pie(
                autopct='%1.1f%%',
                labels=['Not Fraud', 'Fraud'],
                colors=["#66b3ff", "#ff9999"],
                ax=ax1
            )
            ax1.set_title("Dataset 1 Fraud Distribution")
            st.pyplot(fig1)
        else:
            st.warning("⚠️ 'fraud' column not found in Dataset 1.")

    with col2:
        if 'fraud' in df2.columns:
            fig2, ax2 = plt.subplots()
            df2['fraud'].value_counts().plot.pie(
                autopct='%1.1f%%',
                labels=['Not Fraud', 'Fraud'],
                colors=["#66b3ff", "#ff9999"],
                ax=ax2
            )
            ax2.set_title("Dataset 2 Fraud Distribution")
            st.pyplot(fig2)
        else:
            st.warning("⚠️ 'fraud' column not found in Dataset 2.")

    # Dropdown for additional comparisons
    st.subheader("🔽 Visual Comparison by Feature")
    compare_feature = st.selectbox("Choose a feature to compare:", options=[col for col in df1.columns if df1[col].dtype in ['int64', 'float64']])

    if compare_feature:
        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df1[compare_feature], kde=True, ax=ax[0], color='skyblue')
        ax[0].set_title(f"Dataset 1 - {compare_feature}")
        sns.histplot(df2[compare_feature], kde=True, ax=ax[1], color='salmon')
        ax[1].set_title(f"Dataset 2 - {compare_feature}")
        st.pyplot(fig)

else:
    st.warning("Please upload both datasets to proceed.")
