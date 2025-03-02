import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def show_mnist_demo_3(kmeans, dbscan, X_train, X_valid, X_test, y_test):
    st.header("Clustering Results Demo")
    
    # Dự đoán
    kmeans_pred = kmeans.predict(X_test)
    dbscan_pred = dbscan.predict(X_test)
    
    # Hiển thị kết quả
    n_samples = st.slider("Số lượng mẫu để hiển thị", 5, 20, 10)
    idx = np.random.choice(len(X_test), n_samples, replace=False)
    
    fig, axes = plt.subplots(3, n_samples, figsize=(n_samples*2, 6))
    
    for i, sample_idx in enumerate(idx):
        # Hình ảnh gốc
        axes[0, i].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
        axes[0, i].set_title(f"True: {y_test[sample_idx]}")
        axes[0, i].axis('off')
        
        # K-Means prediction
        axes[1, i].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
        axes[1, i].set_title(f"K-Means: {kmeans_pred[sample_idx]}")
        axes[1, i].axis('off')
        
        # DBSCAN prediction
        axes[2, i].imshow(X_test[sample_idx].reshape(28, 28), cmap='gray')
        axes[2, i].set_title(f"DBSCAN: {dbscan_pred[sample_idx]}")
        axes[2, i].axis('off')
    
    st.pyplot(fig)
    
    # Download button
    if st.button("Tải xuống kết quả"):
        st.write("Tính năng tải xuống sẽ được thêm sau")

if __name__ == "__main__":
    # Để test standalone, cần tạo dữ liệu mẫu
    from sklearn.cluster import KMeans, DBSCAN
    import numpy as np
    kmeans = KMeans(n_clusters=10)
    dbscan = DBSCAN(eps=1.0, min_samples=5)
    X_dummy = np.random.rand(100, 784)
    y_dummy = np.random.randint(0, 10, 100)
    show_mnist_demo(kmeans, dbscan, X_dummy, X_dummy, X_dummy, y_dummy)