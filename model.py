import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset
dataset = pd.read_csv('Mall.csv')

# Preprocessing (convert Gender to numerical and select relevant features)
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use k=4 for the KMeans model
kmeans = KMeans(n_clusters=4, init='k-means++', random_state=42)
kmeans.fit(X_scaled)

# Predict clusters
clusters = kmeans.predict(X_scaled)

# Evaluate the model
inertia = kmeans.inertia_  # Inertia (WCSS)
silhouette_avg = silhouette_score(X_scaled, clusters)  # Silhouette score

# Print evaluation results
print(f"Inertia (WCSS): {inertia}")
print(f"Silhouette Score: {silhouette_avg}")

# Save the model if needed
import joblib
joblib.dump(kmeans, 'kmeans_model_k4.pkl')
joblib.dump(scaler, 'scaler_k4.pkl')
