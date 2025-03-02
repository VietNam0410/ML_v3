import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.datasets import make_blobs

def introduce_clustering():
    st.header("Introduction to Clustering Algorithms")
    
    st.write("""
    ### K-Means Clustering
    - Là thuật toán phân cụm không giám sát
    - Chia dữ liệu thành k cụm dựa trên khoảng cách Euclidean
    - Yêu cầu xác định trước số cụm (k)
    
    ### DBSCAN Clustering
    - Density-Based Spatial Clustering of Applications with Noise
    - Phân cụm dựa trên mật độ điểm
    - Không cần xác định số cụm trước
    - Có thể tìm ra noise/outliers
    """)
    
    # Tạo dữ liệu mẫu
    n_samples = st.slider("Số lượng mẫu", 100, 1000, 300)
    random_state = 42
    
    X, _ = make_blobs(n_samples=n_samples, centers=4, cluster_std=1, random_state=random_state)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("K-Means")
        k = st.slider("Số cụm (k)", 2, 10, 4, key="kmeans_k")
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans_labels = kmeans.fit_predict(X)
        
        fig1, ax1 = plt.subplots()
        scatter = ax1.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='viridis')
        plt.colorbar(scatter)
        st.pyplot(fig1)
    
    with col2:
        st.subheader("DBSCAN")
        eps = st.slider("Epsilon", 0.1, 2.0, 0.5, key="dbscan_eps")
        min_samples = st.slider("Min Samples", 2, 20, 5, key="dbscan_min")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X)
        
        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(X[:, 0], X[:, 1], c=dbscan_labels, cmap='viridis')
        plt.colorbar(scatter)
        st.pyplot(fig2)

if __name__ == "__main__":
    introduce_clustering()