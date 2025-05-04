import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure page (if not already at top of script)
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Title
st.title("📊 Static Dataset Comparison")

# Upload files
uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"])
uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"])

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    df1 = pd.read_csv(uploaded_file_1)
    df2 = pd.read_csv(uploaded_file_2)

    st.subheader("Dataset 1 Overview")
    st.write(df1.head())
    st.write(df1.describe())

    st.subheader("Dataset 2 Overview")
    st.write(df2.head())
    st.write(df2.describe())

    # Pie Charts
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

    # Feature comparison
    st.subheader("🔽 Visual Comparison by Feature and Chart Type")

    numeric_columns = [col for col in df1.columns if df1[col].dtype in ['int64', 'float64']]
    
    selected_feature = st.selectbox("Select feature to compare:", numeric_columns)
    chart_type = st.selectbox("Select chart type:", ["Histogram", "Boxplot", "Violinplot", "KDE Plot"])

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    if chart_type == "Histogram":
        sns.histplot(df1[selected_feature], kde=True, ax=ax[0], color='skyblue')
        sns.histplot(df2[selected_feature], kde=True, ax=ax[1], color='salmon')
    elif chart_type == "Boxplot":
        sns.boxplot(y=df1[selected_feature], ax=ax[0], color='skyblue')
        sns.boxplot(y=df2[selected_feature], ax=ax[1], color='salmon')
    elif chart_type == "Violinplot":
        sns.violinplot(y=df1[selected_feature], ax=ax[0], color='skyblue')
        sns.violinplot(y=df2[selected_feature], ax=ax[1], color='salmon')
    elif chart_type == "KDE Plot":
        sns.kdeplot(df1[selected_feature], ax=ax[0], color='skyblue', fill=True)
        sns.kdeplot(df2[selected_feature], ax=ax[1], color='salmon', fill=True)

    ax[0].set_title(f"Dataset 1 - {chart_type} of {selected_feature}")
    ax[1].set_title(f"Dataset 2 - {chart_type} of {selected_feature}")

    st.pyplot(fig)
