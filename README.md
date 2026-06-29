# 🛍️ Mall Customer Segmentation using K-Means Clustering

## 📌 Overview

This project uses **Machine Learning (K-Means Clustering)** to segment mall customers into meaningful groups based on their shopping behavior. The application helps businesses identify different customer categories and create targeted marketing strategies.

An interactive **Streamlit Dashboard** allows users to explore customer data, visualize clusters, and predict the customer segment based on Age, Annual Income, and Spending Score.

---

## 🚀 Features

- Customer Segmentation using K-Means Clustering
- Feature Scaling using StandardScaler
- Elbow Method for selecting the optimal number of clusters
- Interactive Streamlit Dashboard
- Customer Segment Prediction
- Business Insights & Marketing Recommendations
- Interactive Data Visualizations
- Cluster Statistics Dashboard

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Mall-Customer-Segmentation/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── Mall_Customers.csv
│
├── model/
│   ├── kmeans_v2.pkl
│   └── scaler.pkl
│
├── images/
│
└── mall_customer_segmentation.ipynb
```

---

## 📊 Dataset

The dataset contains information about **200 mall customers**.

### Features

- Customer ID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

---

## 🤖 Machine Learning Workflow

1. Data Collection
2. Data Exploration
3. Data Visualization
4. Feature Selection
5. Feature Scaling
6. Elbow Method
7. K-Means Clustering
8. Customer Segmentation
9. Streamlit Dashboard Deployment

---

## 📈 Customer Segments

The model groups customers into **4 meaningful segments**:

- ⭐ Premium Customers
- 👨‍👩‍👧 Average Customers
- 🛍️ Young Active Customers
- 💼 Potential Customers

Each segment includes personalized business insights and marketing recommendations.

---

## 🎯 Streamlit Dashboard

The dashboard includes:

- 🏠 Home Page
- 📊 Exploratory Data Analysis
- 🤖 Customer Segment Prediction
- 📈 Cluster Visualization
- 📋 Cluster Statistics
- ℹ️ About Page

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Mall-Customer-Segmentation.git
```

Move into the project folder

```bash
cd Mall-Customer-Segmentation
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Screenshots

Add screenshots of:

- Home Page
- Customer Prediction
- Cluster Visualization
- Dashboard

inside the **images** folder and update this section.

Example:

```
images/
home.png
prediction.png
clusters.png
dashboard.png
```

---

## 🔮 Future Improvements

- Customer Recommendation System
- PDF Report Generation
- Model Comparison with DBSCAN and Hierarchical Clustering
- Real-time Customer Data Upload
- Cloud Deployment

---

## 👨‍💻 Author

**Pranjal Jain**

B.Tech (CSIT)

Machine Learning | Data Science | Web Development | DSA



---

## ⭐ If you found this project useful, don't forget to star the repository.