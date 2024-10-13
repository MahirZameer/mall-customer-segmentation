import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# Set K to a fixed value of 5
K = 5

# Title and introduction
st.title('Mall Customer Segmentation App')

# Customer input form
st.subheader("Customer Input Form")
name = st.text_input("Name")
email = st.text_input("Email")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ['Male', 'Female'])
salary = st.number_input("Annual Salary (in k$)", min_value=0, max_value=200, step=1)
spending_score = st.number_input("Spending Score (1-100)", min_value=0, max_value=100, step=1)

# Convert Gender to numeric
gender_value = 0 if gender == 'Male' else 1

# Load Mall.csv dataset for training the KMeans model
dataset = pd.read_csv('Mall.csv')

# Preprocess the dataset (same as before)
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the KMeans model with K=5
kmeans = KMeans(n_clusters=K, random_state=42)
kmeans.fit(X_scaled)

# Predict the cluster for the customer input
customer_data = np.array([[gender_value, age, salary, spending_score]])
customer_data_scaled = scaler.transform(customer_data)
customer_cluster = kmeans.predict(customer_data_scaled)[0]

# Customer gets a confirmation message
if st.button('Submit'):
    st.success(f'Thank you, {name}. Your details have been submitted.')

# Only show the cluster to the mall owner (password-protected or hidden from customer)
st.sidebar.subheader("Mall Owner View")
password = st.sidebar.text_input("Enter mall owner password", type="password")
if password == "mallowner123":  # Example password
    st.sidebar.write(f"Customer {name} has been assigned to Cluster {customer_cluster}.")
else:
    st.sidebar.write("Access restricted to mall owners.")

