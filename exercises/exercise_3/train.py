import streamlit as st
import mlflow
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
import numpy as np

def train_mnist_3(X_selected, y_selected, X_test, y_test):
    st.header("Training Clustering Models")
    
    # Chia tập train/valid
    X_train, X_valid = train_test_split(X_selected, test_size=0.2, random_state=42)
    
    col1, col2 = st.columns(2)
    
    with mlflow.start_run():
        # K-Means
        with col1:
            st.subheader("K-Means Training")
            n_clusters = st.slider("Số cụm", 5, 15, 10, key="train_kmeans")
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            kmeans.fit(X_train)
            
            mlflow.log_param("n_clusters", n_clusters)
            mlflow.sklearn.log_model(kmeans, "kmeans_model")
            st.write("Đã hoàn thành huấn luyện K-Means")
        
        # DBSCAN
        with col2:
            st.subheader("DBSCAN Training")
            eps = st.slider("Epsilon", 0.1, 5.0, 1.0, key="train_dbscan_eps")
            min_samples = st.slider("Min Samples", 2, 20, 5, key="train_dbscan_min")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            dbscan.fit(X_train)
            
            mlflow.log_param("eps", eps)
            mlflow.log_param("min_samples", min_samples)
            mlflow.sklearn.log_model(dbscan, "dbscan_model")
            st.write("Đã hoàn thành huấn luyện DBSCAN")
    
    return kmeans, dbscan, X_train, X_valid, X_test, y_test

if __name__ == "__main__":
    # Để test file này standalone, cần có data
    from preprocess import preprocess_mnist
    X_selected, y_selected, X_test, y_test = preprocess_mnist()
    train_mnist(X_selected, y_selected, X_test, y_test)