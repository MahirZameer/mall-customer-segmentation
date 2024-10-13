import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load the pre-trained model and scaler
kmeans = joblib.load('kmeans_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit app interface
st.title('Mall Customer Segmentation App')

# Customer input form
name = st.text_input("Customer Name")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
income = st.number_input("Annual Income (in k$)", min_value=1, max_value=200)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100)

# Predict cluster on button click
if st.button("Predict Cluster"):
    # Convert Gender to numerical
    gender_num = 0 if gender == "Male" else 1
    
    # Prepare input data
    new_customer = [[gender_num, age, income, spending_score]]
    
    # Scale the input
    new_customer_scaled = scaler.transform(new_customer)
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(new_customer_scaled)[0]
    
    # Display the predicted cluster
    st.write(f"Customer {name} belongs to Cluster {predicted_cluster}")
    
    # Provide suggestions based on the cluster
    if predicted_cluster == 0:
        st.write("Suggested Action: Offer budget-friendly products.")
    elif predicted_cluster == 1:
        st.write("Suggested Action: Provide exclusive high-end products.")
    elif predicted_cluster == 2:
        st.write("Suggested Action: Promote mid-range family-oriented deals.")
    elif predicted_cluster == 3:
        st.write("Suggested Action: Target tech-savvy young customers.")
