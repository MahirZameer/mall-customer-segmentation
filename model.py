from sklearn.cluster import KMeans
import joblib
from data_preprocessing import load_and_clean_data

def train_and_save_model():
    # Load and preprocess the data
    X_scaled, scaler = load_and_clean_data()
    
    # Train KMeans with k=5
    kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    
    # Save the model and scaler for later use
    joblib.dump(kmeans, 'kmeans_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

    return kmeans, scaler

# Run the training function when called
if __name__ == "__main__":
    train_and_save_model()
