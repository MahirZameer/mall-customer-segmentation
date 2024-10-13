import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib

# Load and preprocess the data (previously done in a separate file)
@st.cache
def load_and_clean_data():
    # Load the dataset
    dataset = pd.read_csv('Mall.csv')
    
    # Clean the data (e.g., handle missing values, encode categorical data)
    dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})  # Convert Gender to numerical
    
    # Select relevant features for clustering
    X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    
    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, scaler

# Train the model
@st.cache
def train_model(X_scaled):
    # Train KMeans with k=5
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    
    # Save the model and scaler
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    return kmeans, scaler

# Load or train the model
X_scaled, scaler = load_and_clean_data()
kmeans = train_model(X_scaled)

# Streamlit app interface
st.title('Mall Customer Segmentation App')

# Form for customer input
name = st.text_input("Name")
age = st.number_input("Age", min_value=18, max_value=100, step=1)
gender = st.selectbox("Gender", ("Male", "Female"))
income = st.number_input("Annual Income (in k$)", min_value=1, max_value=200)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100)

if st.button("Predict Cluster"):
    # Convert Gender to numerical
    gender_num = 0 if gender == "Male" else 1
    
    # Prepare input for the model
    new_customer = [[gender_num, age, income, spending_score]]
    
    # Scale the input
    new_customer_scaled = scaler.transform(new_customer)
    
    # Predict the cluster
    predicted_cluster = kmeans.predict(new_customer_scaled)
    
    # Display the cluster and suggested action
    st.write(f"Customer {name} belongs to Cluster {predicted_cluster[0]}")
    
    # Display cluster-specific action (example)
    if predicted_cluster[0] == 0:
        st.write("Suggested Action: Offer budget-friendly products.")
    elif predicted_cluster[0] == 1:
        st.write("Suggested Action: Provide exclusive high-end products.")
    # Add more actions for other clusters here
