import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Title and Introduction
st.title('Mall Customer Segmentation using K-Means')
st.write("""
This app uses the **k-means clustering algorithm** to segment customers based on demographic and shopping behavior data.
The data is sourced from the Mall Customer Segmentation Dataset.
""")

# Step 1: Load the default dataset (Mall.csv)
dataset = pd.read_csv('Mall.csv')

# Step 2: Display the dataset
st.write("### Dataset Overview")
st.write(dataset.head())

# Step 3: Data preprocessing
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar Section for input parameters
st.sidebar.header("Input Parameters")

# Sidebar input for number of clusters
n_clusters = st.sidebar.slider('Select Number of Clusters', 2, 10, 5)

# Step 4: Apply K-Means Clustering with custom number of clusters
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)

# Step 5: Add the cluster assignment to the dataset
dataset['Cluster'] = clusters

# Step 6: Display the cluster assignment
st.write("### Clustered Dataset")
st.write(dataset[['CustomerID', 'Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Cluster']])

# Step 7: Visualize the clusters
st.subheader('Cluster Visualization')
x_axis = st.sidebar.selectbox('X-Axis', ['Annual Income (k$)', 'Age', 'Spending Score (1-100)'])
y_axis = st.sidebar.selectbox('Y-Axis', ['Annual Income (k$)', 'Age', 'Spending Score (1-100)'])

# Get the index for X and Y axis
axis_map = {
    'Annual Income (k$)': 2,
    'Age': 1,
    'Spending Score (1-100)': 3
}

fig, ax = plt.subplots()
ax.scatter(X_scaled[:, axis_map[x_axis]], X_scaled[:, axis_map[y_axis]], c=clusters, s=100, cmap='viridis')
ax.set_xlabel(f'{x_axis} (scaled)')
ax.set_ylabel(f'{y_axis} (scaled)')
st.pyplot(fig)

# Sidebar for cluster information
st.sidebar.subheader("Cluster Information")
st.sidebar.write("Number of Customers per Cluster")
st.sidebar.write(pd.DataFrame(dataset['Cluster'].value_counts()).rename(columns={'Cluster': 'Count'}))
