import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib

# Set page config
st.set_page_config(page_title="Financial Fraud Detection App", layout="wide")

# Cache data loading
@st.cache_data
def load_data(uploaded_file):
    return pd.read_csv(uploaded_file)

# Sidebar navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Home", "Static Analysis", "Dynamic Prediction"])

# Apply color theme
st.markdown("""
    <style>
    .stApp {
        background-color: #f0f4f8;
    }
    .stSidebar {
        background-color: #ffffff;
    }
    .title {
        color: #003366;
        font-size: 40px;
        font-weight: bold;
    }
    .description, .feature {
        font-size: 16px;
        color: #1f2f3f;
        line-height: 1.6;
    }
    .feature {
        font-weight: bold;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# HOME PAGE
if app_mode == "Home":
    st.markdown('<div class="title">🏦 Financial Fraud Detection App</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="description">
        Welcome to the Financial Fraud Detection App. This app leverages the power of Machine Learning and Big Data Analytics to detect fraudulent transactions in real-time.
    </div>

    <div class="feature">Key Features of the App:</div>

    <div class="description">
        - 🔍 <b>Dataset Exploration:</b> Upload and compare datasets for fraud patterns.<br>
        - 🤖 <b>Fraud Detection:</b> Predict fraudulent transactions using our machine learning models.
    </div>

    <div class="description">
        Built with cutting-edge techniques, this app provides insightful analyses and predictions. The app is a <b>Final Year Project</b> using advanced ML models such as Random Forest for prediction.
    </div>

    <div class="description">
        Explore the sections on the left to get started!
    </div>
    """, unsafe_allow_html=True)

# STATIC ANALYSIS
elif app_mode == "Static Analysis":
    st.title("📊 Static Dataset Comparison")

    uploaded_file_1 = st.file_uploader("Upload Dataset 1", type=["csv"], key="file1")
    uploaded_file_2 = st.file_uploader("Upload Dataset 2", type=["csv"], key="file2")

    if uploaded_file_1 is not None and uploaded_file_2 is not None:
        df1 = load_data(uploaded_file_1)
        df2 = load_data(uploaded_file_2)

        st.subheader("Dataset 1 Overview")
        st.write(df1.head())

        st.subheader("Dataset 2 Overview")
        st.write(df2.head())

        st.subheader("📌 Compare Fraud vs Non-Fraud Distribution")

        try:
            col1, col2 = st.columns(2)

            with col1:
                fig1, ax1 = plt.subplots(figsize=(4, 4))
                df1['fraud'].value_counts().plot.pie(
                    autopct='%1.1f%%',
                    labels=['Not Fraud', 'Fraud'],
                    colors=["#4da6ff", "#ff9999"],
                    ax=ax1
                )
                ax1.set_title("Dataset 1 Fraud Distribution")
                ax1.axis('equal')
                st.pyplot(fig1)

            with col2:
                fig2, ax2 = plt.subplots(figsize=(4, 4))
                df2['fraud'].value_counts().plot.pie(
                    autopct='%1.1f%%',
                    labels=['Not Fraud', 'Fraud'],
                    colors=["#4da6ff", "#ff9999"],
                    ax=ax2
                )
                ax2.set_title("Dataset 2 Fraud Distribution")
                ax2.axis('equal')
                st.pyplot(fig2)

        except KeyError:
            st.error("One of the datasets is missing the 'fraud' column. Please ensure both datasets have it.")

        # Feature Visualization
        st.subheader("📈 Visualize Features")
        selected_feature = st.selectbox("Select Feature to Visualize", df1.columns)
        chart_type = st.selectbox("Select Chart Type", ["Histogram", "Boxplot", "Barplot"])

        fig, ax = plt.subplots(figsize=(6, 4))

        if chart_type == "Histogram":
            sns.histplot(df1[selected_feature], kde=True, ax=ax, color="#4da6ff")
            ax.set_title(f"Histogram of {selected_feature}")

        elif chart_type == "Boxplot":
            sns.boxplot(x=df1[selected_feature], ax=ax, color="#b3e0ff")
            ax.set_title(f"Boxplot of {selected_feature}")

        elif chart_type == "Barplot":
            counts = df1[selected_feature].value_counts().nlargest(10)
            sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, palette=["#4da6ff"] * len(counts))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_title(f"Top 10 Barplot of {selected_feature}")
            for container in ax.containers:
                ax.bar_label(container, fmt='%d')

        fig.tight_layout()
        st.pyplot(fig)

# DYNAMIC PREDICTION
elif app_mode == "Dynamic Prediction":
    st.title("⚙️ Dynamic Fraud Prediction")

    st.markdown("Enter transaction details to predict if it's fraudulent.")

    distance_from_home = st.number_input("Distance from Home", min_value=0.0)
    distance_from_last_transaction = st.number_input("Distance from Last Transaction", min_value=0.0)
    ratio_to_median_purchase_price = st.number_input("Ratio to Median Purchase Price", min_value=0.0)
    repeat_retailer = st.selectbox("Repeat Retailer", [0, 1])
    used_chip = st.selectbox("Used Chip", [0, 1])
    used_pin_number = st.selectbox("Used PIN Number", [0, 1])
    online_order = st.selectbox("Online Order", [0, 1])

    input_data = pd.DataFrame({
        'distance_from_home': [distance_from_home],
        'distance_from_last_transaction': [distance_from_last_transaction],
        'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
        'repeat_retailer': [repeat_retailer],
        'used_chip': [used_chip],
        'used_pin_number': [used_pin_number],
        'online_order': [online_order]
    })

    try:
        scaler = joblib.load("scaler.pkl")
        model = joblib.load("random_forest_model.pkl")
        scaled_input = scaler.transform(input_data)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0][1]

        if st.button("Predict"):
            if prediction == 1:
                st.error(f"🚨 Alert: This transaction is likely FRAUDULENT! (Confidence: {prediction_proba:.2f})")
            else:
                st.success(f"✅ This transaction appears to be LEGITIMATE. (Confidence: {1 - prediction_proba:.2f})")
    except FileNotFoundError:
        st.error("Required model files not found. Please ensure 'scaler.pkl' and 'random_forest_model.pkl' exist.")
