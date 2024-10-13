import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Title and introduction
st.title('🌟 Mall Customer Segmentation using K-Means 🌟')

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

# Load and preprocess dataset
@st.cache
def load_data():
    dataset = pd.read_csv('Mall.csv')
    dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
    X = dataset[['Gender', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return dataset, X_scaled

dataset, X_scaled = load_data()

# Apply K-Means Clustering with K=5
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(X_scaled)
clusters = kmeans.predict(X_scaled)
dataset['Cluster'] = clusters

# Cluster names based on the data characteristics
cluster_names = {
    0: "Older, Low Income, Low Spending",
    1: "Middle-aged, High Income, Low Spending",
    2: "Young, High Income, High Spending",
    3: "Middle-aged, Moderate Income, High Spending",
    4: "Young, Low Income, High Spending"
}

# Customer Input Form
st.header("Customer Input Form 📝")
with st.form(key='customer_form'):
    name = st.text_input("Name")
    email = st.text_input("Email")
    age = st.number_input("Age", min_value=0, max_value=100, value=25)
    gender = st.selectbox("Gender", ["Male", "Female"])
    salary = st.number_input("Annual Income (k$)", min_value=0)
    spending_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100)
    
    submit_button = st.form_submit_button("Submit")

if submit_button:
    # Preprocess the customer input for clustering
    gender = 0 if gender == "Male" else 1
    customer_data = [[gender, age, salary, spending_score]]
    customer_scaled = StandardScaler().fit_transform(customer_data)
    
    # Predict the cluster
    customer_cluster = kmeans.predict(customer_scaled)[0]
    
    # Thank the customer (This is the public-facing message)
    st.success("Thank you for submitting your information! We will be in touch shortly.")
    
    # Store customer information (this part could be enhanced to save in a database)
    with open('customer_data.csv', 'a') as f:
        f.write(f"{name},{email},{age},{gender},{salary},{spending_score},{customer_cluster}\n")

# Hidden section for the Mall Owner to view customer clusters
st.header("Mall Owner Dashboard (Only Visible to You) 🔒")

view_customers = st.checkbox("View Submitted Customer Data")
if view_customers:
    # Load the saved customer data
    try:
        customer_data_df = pd.read_csv('customer_data.csv', header=None)
        customer_data_df.columns = ["Name", "Email", "Age", "Gender", "Annual Income (k$)", "Spending Score", "Cluster"]
        customer_data_df['Cluster'] = customer_data_df['Cluster'].map(cluster_names)
        st.write(customer_data_df)

        # Suggested marketing strategy based on the cluster
        st.subheader("Suggested Marketing Strategies")
        for cluster, name in cluster_names.items():
            st.write(f"**{name}**: {suggest_marketing_strategy(cluster)}")

    except FileNotFoundError:
        st.write("No customer data available yet.")

# Function to suggest marketing strategies based on clusters
def suggest_marketing_strategy(cluster):
    strategies = {
        0: "Offer discounts on essential products to encourage more spending.",
        1: "Promote premium products or loyalty rewards to increase spending.",
        2: "Send exclusive offers on luxury goods and memberships.",
        3: "Provide discounts on bundled offers and special services.",
        4: "Offer trendy and budget-friendly promotions to attract spending."
    }
    return strategies[cluster]

