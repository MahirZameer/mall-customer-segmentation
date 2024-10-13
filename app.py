import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Custom CSS for fonts
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@700&family=Roboto:wght@400&display=swap');

    html, body, [class*="st-"]  {
        font-family: 'Roboto', sans-serif;
    }
    h1, h2, h3 {
        font-family: 'Montserrat', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Add the mall photo as a banner
st.image("https://images.pexels.com/photos/2767756/pexels-photo-2767756.jpeg?auto=compress&cs=tinysrgb&dpr=2&h=750&w=1260", 
         use_column_width=True)

# Title and Introduction
st.title('🌟 Mall Customer Segmentation using K-Means 🌟')
st.write("""
This app uses the **k-means clustering algorithm** to segment customers based on demographic and shopping behavior data.
""")

# Load the dataset
dataset = pd.read_csv('Mall.csv')

# Data Preprocessing
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fixed number of clusters
n_clusters = 5

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
dataset['Cluster'] = clusters

# Suggest marketing strategy based on cluster
def suggest_marketing_strategy(cluster_label):
    strategies = {
        0: "Older, Low Income, Low Spending: Suggest discounts and value products to attract.",
        1: "Middle-aged, High Income, Low Spending: Focus on premium loyalty programs and luxury offers.",
        2: "Young, Moderate Income, High Spending: Target with trendy products and high-end promotions.",
        3: "Middle-aged, High Income, Moderate Spending: Offer incentives like mid-tier discounts and loyalty points.",
        4: "Young, Low Income, Moderate Spending: Offer affordable bundles and special youth-targeted deals.",
        5: "Older, High Income, High Spending: Promote exclusive luxury items and personalized services."
    }
    return strategies.get(cluster_label, "No strategy available")

# Form for customer input
st.subheader('🔍 Find Your Cluster')
with st.form(key='customer_form'):
    name = st.text_input('Enter your name')
    email = st.text_input('Enter your email')
    age = st.slider('Age', 18, 100, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    income = st.slider('Annual Salary (in k$)', 10, 150, 50)
    spending_score = st.slider('Spending Score (1-100)', 1, 100, 50)
    submit_button = st.form_submit_button(label='Submit')

    if submit_button:
        # Preprocess input for prediction
        gender_value = 0 if gender == 'Male' else 1
        customer_data = scaler.transform([[gender_value, age, income, spending_score]])
        customer_cluster = kmeans.predict(customer_data)[0]
        st.success(f"Thank you, {name}. Your details have been submitted.")
        st.info(f"You have been placed in Cluster {customer_cluster} ({suggest_marketing_strategy(customer_cluster)}).")

# Admin view for mall owner to check customer clusters
st.subheader("Mall Owner Dashboard")
owner_password = st.text_input("Enter mall owner password", type="password")
if owner_password == 'mallowner2024':  # A fixed password for simplicity
    st.subheader('Mall Customer Segmentation App')
    cluster_distribution = pd.DataFrame(dataset['Cluster'].value_counts().sort_index(), columns=['Count'])
    st.bar_chart(cluster_distribution)
    st.write("Customer Breakdown per Cluster")
    st.write(pd.DataFrame({
        'Cluster': [0, 1, 2, 3, 4],
        'Age': [56.47, 39.5, 28.69, 37.89, 27.31],
        'Annual Income (k$)': [46.09, 85.15, 60.90, 82.12, 38.84],
        'Spending Score (1-100)': [39.31, 14.05, 70.23, 54.44, 56.21]
    }))
