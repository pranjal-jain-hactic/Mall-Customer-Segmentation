import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Mall Customer Segmentation",
    page_icon="🛍️",
    layout="wide"
)
# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/Mall_Customers.csv")
# Predict cluster for every customer


# ---------------- LOAD MODEL ----------------
model = joblib.load("model/kmeans_v2.pkl")
scaler = joblib.load("model/scaler.pkl")

# Predict cluster for every customer
X_app = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
X_scaled = scaler.transform(X_app)
df["Cluster"] = model.predict(X_scaled)
# ---------------- SIDEBAR ----------------
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home",
        "📊 Data Analysis",
        "🤖 Customer Prediction",
        "📈 Cluster Visualization",
        "📋 Cluster Statistics",
        "ℹ️ About"
    ]
)
if page == "🏠 Home":

    st.title("🛍️ Mall Customer Segmentation")

    st.write("""
    This project uses **K-Means Clustering** to segment mall customers
    based on:

    - Age
    - Annual Income
    - Spending Score

    Businesses can use these customer segments for targeted marketing,
    personalized offers, and improving customer engagement.
    """)

    st.subheader("Dataset")

    st.dataframe(df.head())
    # ---------------- DATA ANALYSIS ----------------
elif page == "📊 Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write("Explore the customer dataset through interactive visualizations.")

    # Display Dataset
    if st.checkbox("Show Dataset"):
        st.dataframe(df)

    # Age Distribution
    st.subheader("Age Distribution")

    fig_age = px.histogram(
        df,
        x="Age",
        nbins=15,
        title="Distribution of Customer Age"
    )

    st.plotly_chart(fig_age, use_container_width=True)

    # Gender Distribution
    st.subheader("Gender Distribution")

    fig_gender = px.histogram(
        df,
        x="Gender",
        color="Gender",
        title="Gender Distribution"
    )

    st.plotly_chart(fig_gender, use_container_width=True)

    # Annual Income Distribution
    st.subheader("Annual Income Distribution")

    fig_income = px.histogram(
        df,
        x="Annual Income (k$)",
        nbins=15,
        title="Annual Income Distribution"
    )

    st.plotly_chart(fig_income, use_container_width=True)

    # Spending Score Distribution
    st.subheader("Spending Score Distribution")

    fig_score = px.histogram(
        df,
        x="Spending Score (1-100)",
        nbins=15,
        title="Spending Score Distribution"
    )

    st.plotly_chart(fig_score, use_container_width=True)
# ---------------- CUSTOMER PREDICTION ----------------
elif page == "🤖 Customer Prediction":

    st.title("🤖 Customer Segmentation Predictor")

    st.write(
        "Enter the customer's details to predict the customer segment."
    )

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider(
            "Age",
            min_value=18,
            max_value=70,
            value=30
        )

        income = st.slider(
            "Annual Income (k$)",
            min_value=15,
            max_value=137,
            value=60
        )

    with col2:
        spending = st.slider(
            "Spending Score (1-100)",
            min_value=1,
            max_value=99,
            value=50
        )

    if st.button("Predict Customer Segment"):

        # Create input dataframe
        input_data = pd.DataFrame(
            [[age, income, spending]],
            columns=[
                "Age",
                "Annual Income (k$)",
                "Spending Score (1-100)"
            ]
        )

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict cluster
        cluster = model.predict(input_scaled)[0]

        # Segment Information
        segment_info = {
            0: {
                "name": "Average Customers",
                "insight": "Moderate income with average spending behavior.",
                "recommendation": "Provide loyalty rewards and seasonal offers."
            },
            1: {
                "name": "Premium Customers",
                "insight": "High income and high spending customers.",
                "recommendation": "Offer VIP memberships and premium products."
            },
            2: {
                "name": "Young Active Customers",
                "insight": "Young customers with good spending habits.",
                "recommendation": "Target with fashion trends and personalized promotions."
            },
            3: {
                "name": "Potential Customers",
                "insight": "High income but low spending.",
                "recommendation": "Encourage purchases using personalized discounts."
            }
        }

        result = segment_info[cluster]

        st.success("Prediction Completed Successfully!")

        st.markdown("---")

        st.subheader("Customer Profile")

        st.metric("Predicted Segment", result["name"])

        st.write("### Business Insight")
        st.info(result["insight"])

        st.write("### Recommendation")
        st.success(result["recommendation"])
# ---------------- CLUSTER VISUALIZATION ----------------
elif page == "📈 Cluster Visualization":

    st.title("📈 Customer Segments")

    fig = px.scatter(
        df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        color=df["Cluster"].astype(str),
        hover_data=["Age"],
        title="Customer Segments"
    )

    st.plotly_chart(fig, use_container_width=True)
# ---------------- CLUSTER STATISTICS ----------------
elif page == "📋 Cluster Statistics":

    st.title("📋 Cluster Statistics")

    summary = (
        df.groupby("Cluster")
        [["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
        .mean()
        .round(2)
    )

    count = (
        df["Cluster"]
        .value_counts()
        .sort_index()
        .rename("Customers")
    )

    st.subheader("Average Values")

    st.dataframe(summary)

    st.subheader("Customers in Each Cluster")

    st.bar_chart(count)
# ---------------- ABOUT ----------------
elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.markdown("""
## Mall Customer Segmentation

This project applies **K-Means Clustering** to segment mall customers using:

- Age
- Annual Income
- Spending Score

### Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

### Machine Learning Algorithm

K-Means Clustering

### Objective

Help businesses understand customer behavior and create targeted marketing strategies.
""")