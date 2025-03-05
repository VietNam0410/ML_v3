import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.datasets import mnist
import openml
import plotly.express as px
import mlflow
import os
import dagshub
import datetime
from sklearn.preprocessing import StandardScaler

# Thiết lập MLflow và DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

# Cache dữ liệu MNIST để tối ưu hóa
@st.cache_data(ttl=86400)  # Làm mới sau 24 giờ
def load_mnist(max_samples=10000):  # Giảm mặc định xuống 10,000 để tăng tốc
    with st.spinner("Đang tải dữ liệu MNIST..."):
        try:
            dataset = openml.datasets.get_dataset(554)
            X, y, _, _ = dataset.get_data(target='class')
            X = X.values.reshape(-1, 28 * 28) / 255.0  # Làm phẳng và chuẩn hóa
            y = y.astype(np.int32)
        except Exception as e:
            st.error(f"Không thể tải từ OpenML: {str(e)}. Sử dụng TensorFlow.")
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X = np.concatenate([X_train, X_test], axis=0) / 255.0
            y = np.concatenate([y_train, y_test], axis=0)
            X = X.reshape(-1, 28 * 28)
        if max_samples < len(X):
            indices = np.random.choice(len(X), max_samples, replace=False)
            X, y = X[indices], y[indices]
        return X, y

# Cache scaler để tái sử dụng
@st.cache_resource
def get_scaler():
    return StandardScaler()

# Hàm huấn luyện và giảm chiều (tối ưu hóa tốc độ)
def train_dimensionality_reduction(X, y, method, n_components):
    scaler = get_scaler()
    X_scaled = scaler.fit_transform(X)
    
    start_time = datetime.datetime.now()
    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "t-SNE":
        model = TSNE(n_components=n_components, perplexity=15, n_iter=500, random_state=42, n_jobs=-1)  # Tăng tốc t-SNE
    
    with st.spinner(f"Đang giảm chiều bằng {method}..."):
        X_reduced = model.fit_transform(X_scaled)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    return X_reduced, model, duration

# Hàm trực quan hóa kết quả (tương tác 2D/3D với Plotly, bỏ độ tin cậy)
def visualize_reduction(X_reduced, y, method, n_components):
    if n_components == 2:
        fig = px.scatter(
            X_reduced, x=0, y=1, color=y, labels={'color': 'Nhãn (0-9)'},
            title=f"Trực quan hóa {method} (2D)", color_continuous_scale='Viridis'
        )
    else:  # n_components == 3
        fig = px.scatter_3d(
            X_reduced, x=0, y=1, z=2, color=y, labels={'color': 'Nhãn (0-9)'},
            title=f"Trực quan hóa {method} (3D)", color_continuous_scale='Viridis'
        )
    
    st.plotly_chart(fig, use_container_width=True)

# Hàm log kết quả vào MLflow (chỉ giữ các thông tin cần thiết)
def log_results(method, n_components, duration, X, y, X_reduced, model):
    experiment_name = "MNIST_Dimensionality_Reduction"
    mlflow.set_experiment(experiment_name)
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_name = f"{method}_n={n_components}_{timestamp.replace(' ', '_').replace(':', '-')}"
    
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_param("method", method)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("max_samples", len(X))
        mlflow.log_param("timestamp", timestamp)
        mlflow.log_metric("duration_seconds", duration)
        
        if method == "PCA":
            mlflow.log_metric("explained_variance_ratio", np.sum(model.explained_variance_ratio_))
        
        # Chỉ log model, không log scaler để tránh lỗi
        mlflow.sklearn.log_model(model, "model", input_example=X[:1])
        
        run_id = run.info.run_id
        mlflow_uri = DAGSHUB_MLFLOW_URI
        st.success(f"Đã log kết quả vào MLflow! (Run: {run_name}, ID: {run_id}, Thời gian: {timestamp})")
        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow]({mlflow_uri})")

# Giao diện Streamlit
def dimensionality_reduction_app():
    st.title("🌐 Giảm Chiều Dữ Liệu MNIST với PCA và t-SNE")

    # Tải dữ liệu với thanh trạng thái
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.write("Bắt đầu tải dữ liệu MNIST...")
    X, y = load_mnist()
    progress_bar.progress(100)
    status_text.write("Dữ liệu MNIST đã sẵn sàng! ✅")

    # Chọn số mẫu (giảm mặc định để tăng tốc)
    max_samples = st.slider("Chọn số lượng mẫu (0 = toàn bộ, tối đa 70.000)", 0, 70000, 10000, step=1000)
    if max_samples == 0 or max_samples > len(X):
        st.warning(f"Số mẫu {max_samples} vượt quá {len(X)}. Sử dụng toàn bộ {len(X)} mẫu.")
        max_samples = len(X)
    elif max_samples < len(X):
        indices = np.random.choice(len(X), max_samples, replace=False)
        X, y = X[indices], y[indices]

    # Chọn phương pháp giảm chiều
    method = st.selectbox("Chọn phương pháp giảm chiều", ["PCA", "t-SNE"])

    # Chọn số chiều (chỉ 2 hoặc 3)
    n_components = st.slider("Chọn số chiều sau khi giảm (2 hoặc 3)", 2, 3, 2, step=1)

    # Huấn luyện và trực quan hóa
    if st.button("Giảm chiều và trực quan hóa", key="reduce_and_visualize"):
        with st.spinner("Đang thực hiện giảm chiều..."):
            X_reduced, model, duration = train_dimensionality_reduction(X, y, method, n_components)
            visualize_reduction(X_reduced, y, method, n_components)
        
        # Log kết quả
        log_results(method, n_components, duration, X, y, X_reduced, model)

    # Thông tin về phương pháp
    st.subheader("📚 Thông tin về các phương pháp")
    if method == "PCA":
        st.write("""
            **PCA (Principal Component Analysis)**:
            - Là kỹ thuật tuyến tính giảm chiều, giữ lại các thành phần chính (principal components) giải thích phần lớn phương sai trong dữ liệu.
            - Ưu điểm: Nhanh, dễ hiểu, hiệu quả với dữ liệu tuyến tính.
            - Nhược điểm: Không hoạt động tốt với dữ liệu phi tuyến.
        """)
    else:  # t-SNE
        st.write("""
            **t-SNE (t-Distributed Stochastic Neighbor Embedding)**:
            - Là kỹ thuật phi tuyến giảm chiều, tối ưu hóa để bảo toàn cấu trúc cục bộ (local structure) của dữ liệu.
            - Ưu điểm: Tốt cho trực quan hóa dữ liệu phức tạp, phi tuyến.
            - Nhược điểm: Chậm, nhạy với tham số (perplexity, n_iter).
        """)

    # Thêm nút làm mới cache với key duy nhất
    if st.button("🔄 Làm mới dữ liệu", key=f"refresh_data_{datetime.datetime.now().microsecond}"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    dimensionality_reduction_app()