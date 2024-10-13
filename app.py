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

# Title and Introduction
st.title('🌟 Mall Customer Segmentation 🌟')

# Collecting customer information
st.header("Customer Information")
name = st.text_input("Name")
email = st.text_input("Email")
age = st.number_input("Age", min_value=1, max_value=100)
gender = st.selectbox("Gender", ["Male", "Female"])
income = st.number_input("Annual Salary (in k$)", min_value=0)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

# Gender mapping for KMeans
gender = 0 if gender == "Male" else 1

# Preprocessing customer input
input_data = pd.DataFrame([[gender, age, income, spending_score]],
                          columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
scaler = StandardScaler()
input_scaled = scaler.fit_transform(input_data)

# K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(input_scaled)
cluster = kmeans.predict(input_scaled)[0]

# Cluster naming
cluster_names = {
    0: "Older, Low Income, Low Spending",
    1: "Middle-aged, High Income, Low Spending",
    2: "Young, Moderate Income, High Spending",
    3: "Middle-aged, High Income, Moderate Spending",
    4: "Young, Low Income, Moderate Spending"
}

# Display result to the customer
if st.button("Submit"):
    st.success(f"Thank you, {name}. Your details have been submitted.")
    st.write(f"You have been placed in Cluster {cluster} ({cluster_names[cluster]}).")

# Mall owner dashboard
st.sidebar.header("Mall Owner Dashboard")
view_mode = st.sidebar.selectbox("Select View", ["Customer Form", "Mall Owner Dashboard"])

if view_mode == "Mall Owner Dashboard":
    password = st.sidebar.text_input("Enter mall owner password", type="password")
    if password == "mallowner123":  # Placeholder password for simplicity
        st.subheader("Mall Owner Dashboard")
        st.write("Customer Breakdown per Cluster")
        # Simulated data to show customer breakdown in each cluster
        cluster_summary = pd.DataFrame({
            'Cluster': [0, 1, 2, 3, 4],
            'Age': [56, 39, 28, 37, 27],
            'Annual Income (k$)': [46, 85, 60, 82, 38],
            'Spending Score (1-100)': [39, 14, 70, 54, 56]
        })
        st.write(cluster_summary)
        
        # Cluster visualization bar chart
        st.subheader("Cluster Distribution")
        fig, ax = plt.subplots()
        ax.bar(cluster_summary['Cluster'], cluster_summary['Spending Score (1-100)'])
        ax.set_xlabel("Cluster")
        ax.set_ylabel("Spending Score")
        st.pyplot(fig)
    else:
        st.error("Incorrect password.")
