import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import base64
def load_image(image_file):
    with open(image_file, "rb") as f:
        img_bytes = f.read()
    return base64.b64encode(img_bytes).decode()
def add_background(image_file):
    img_base64 = load_image(image_file)
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/png;base64,{img_base64}) no-repeat center center fixed;
            background-size: cover;
            height: 100vh;
        }}
        .sidebar .sidebar-content {{
            background: rgba(255, 255, 255, 0.8);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
st.title("Customer Segmentation")
add_background("image23.jpg")
st.header("Upload Customer Data")
st.write("A Customer Segmentation tool.")
st.write("Upload the customer data and set preprocessing steps and Download your Visualization data")
st.markdown("Developed by Raghu Dharsan")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(data)

    st.header("Preprocessing Settings")

    # Handle missing values
    if st.checkbox("Fill missing values"):
        fill_value = st.number_input("Fill value", value=0)
        data = data.fillna(fill_value)
        st.write("Data after filling missing values:")
        st.write(data)
    
    # Encode categorical variables
    if st.checkbox("Encode categorical variables"):
        data = pd.get_dummies(data, drop_first=True)
        st.write("Data after encoding categorical variables:")
        st.write(data)

    # Normalize/Standardize the data
    if st.checkbox("Standardize data"):
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        data = pd.DataFrame(data_scaled, columns=data.columns)
        st.write("Data after standardization:")
        st.write(data)

    # Clustering
    st.header("Clustering Settings")
    n_clusters = st.slider("Number of Clusters", 2, 10, 3)
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data)

    # Assign clusters to data
    data['Cluster'] = kmeans.labels_

    st.write("Clustered Data:")
    st.write(data)

    # Plotting the clusters (assuming data has 2 features for simplicity)
    st.header("Cluster Plot")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=data.columns[0], y=data.columns[1], hue='Cluster', palette='viridis')
    plt.title('Customer Segments')
    st.pyplot(plt)
else:
    st.write("Please upload a CSV file to proceed.")
