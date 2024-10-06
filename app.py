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
n_clusters = st.sidebar.slider('Select Number of Clusters', 2, 10, 5)

# Apply K-Means Clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
dataset['Cluster'] = clusters

# Display clusters
st.subheader('🧩 Clustered Dataset')
st.write(dataset[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

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
