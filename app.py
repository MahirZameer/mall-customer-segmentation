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
The data is sourced from the Mall Customer Segmentation Dataset.
""")

# Load the default dataset (Mall.csv)
dataset = pd.read_csv('Mall.csv')

# Dataset overview
st.subheader('📊 Dataset Overview')
st.write(dataset.head())

# Data Preprocessing
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar for customization
st.sidebar.header("Customization Options 🎨")

# Number of clusters
n_clusters = 5

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
dataset['Cluster'] = clusters

# Cluster Centers and Reverse Transforming for interpretation
cluster_centers = kmeans.cluster_centers_
cluster_centers_original = scaler.inverse_transform(cluster_centers)

# Display the cluster centers in their original form (unscaled)
st.subheader("📊 Cluster Analysis")
cluster_info = pd.DataFrame(cluster_centers_original, columns=['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)'])
st.write(cluster_info)

# Naming clusters based on the center characteristics
cluster_names = {
    0: "Senior, Low Income, Moderate Spending",
    1: "Middle-aged, High Income, Low Spending",
    2: "Young, Moderate Income, High Spending",
    3: "Middle-aged, High Income, Moderate Spending",
    4: "Young, Low Income, High Spending"
}

# Display clusters
st.subheader('🧩 Clustered Dataset')
dataset['Cluster Label'] = dataset['Cluster'].map(cluster_names)
st.write(dataset[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster Label']])

# Visualize the clusters
st.subheader('🎨 Cluster Visualization')
x_axis = st.sidebar.selectbox('X-Axis', ['Annual Income (k$)', 'Age', 'Spending Score (1-100)'])
y_axis = st.sidebar.selectbox('Y-Axis', ['Annual Income (k$)', 'Age', 'Spending Score (1-100)'])

# Get the index for X and Y axis
axis_map = {
    'Annual Income (k$)': 2,
    'Age': 1,
    'Spending Score (1-100)': 3
}

fig, ax = plt.subplots()
ax.scatter(X_scaled[:, axis_map[x_axis]], X_scaled[:, axis_map[y_axis]], c=clusters, s=100, cmap='coolwarm')
ax.set_xlabel(f'{x_axis} (scaled)')
ax.set_ylabel(f'{y_axis} (scaled)')
st.pyplot(fig)

# Cluster information in the sidebar
st.sidebar.subheader("Cluster Information")
st.sidebar.write("Number of Customers per Cluster")
st.sidebar.write(pd.DataFrame(dataset['Cluster'].value_counts()).rename(columns={'Cluster': 'Count'}))

# Collect new customer information
st.subheader("📥 Enter New Customer Information")
name = st.text_input("Name")
email = st.text_input("Email")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 18, 100, 25)
salary = st.slider("Annual Income (k$)", 15, 150, 40)
spending_score = st.slider("Spending Score (1-100)", 1, 100, 50)

if st.button("Submit"):
    # Preprocess the customer input for clustering
    gender_value = 0 if gender == "Male" else 1
    customer_data = [[gender_value, age, salary, spending_score]]

    # Standardize the new customer data using the same scaler used for the dataset
    customer_scaled = scaler.transform(customer_data)

    # Predict the cluster
    customer_cluster = kmeans.predict(customer_scaled)[0]
    customer_cluster_label = cluster_names[customer_cluster]

    st.write(f"Thank you, {name}. Your details have been submitted.")
    st.write(f"Cluster Assignment (Visible to Mall Owner): {customer_cluster_label}")
