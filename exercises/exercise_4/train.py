import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import datetime
import time

# Thiết lập MLflow và DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

# Cache scaler để tái sử dụng
@st.cache_resource
def get_scaler():
    return StandardScaler()

# Hàm train với tiến trình đồng bộ, chỉ hiển thị thanh tiến trình
@st.cache_data
def train_dimensionality_reduction(X, y, method, n_components):
    scaler = get_scaler()
    
    # Khởi tạo thanh tiến trình
    progress = st.progress(0)
    status_text = st.empty()

    # Bước 1: Chuẩn hóa dữ liệu (20% tiến trình)
    progress.progress(0.2)
    time.sleep(0.1)  # Giả lập thời gian xử lý
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    
    start_time = datetime.datetime.now()
    if method == "PCA":
        # Bước 2: Giảm chiều với PCA (80% tiến trình)
        progress.progress(0.8)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        model = PCA(n_components=n_components, svd_solver='randomized')
        X_reduced = model.fit_transform(X_scaled)
    elif method == "t-SNE":
        # Bước 2: Giảm chiều trung gian với PCA (50% tiến trình)
        progress.progress(0.5)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        intermediate_dims = min(50, X_scaled.shape[1])
        pca = PCA(n_components=intermediate_dims, svd_solver='randomized')
        X_intermediate = pca.fit_transform(X_scaled)
        
        # Bước 3: Giảm chiều cuối cùng với t-SNE (100% tiến trình)
        progress.progress(1.0)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        perplexity = min(30, max(5, len(X) // 200))
        model = TSNE(n_components=n_components, perplexity=perplexity, max_iter=250, 
                     random_state=42, n_jobs=-1)
        X_reduced = model.fit_transform(X_intermediate)
    
    duration = (datetime.datetime.now() - start_time).total_seconds()
    status_text.text("Hoàn tất giảm chiều!")
    return X_reduced, model, duration

def visualize_reduction(X_reduced, y, method, n_components):
    if n_components == 2:
        df = pd.DataFrame({
            'x': X_reduced[:, 0],
            'y': X_reduced[:, 1],
            'label': y.astype(str)
        })
        fig = go.Figure()
        for label in sorted(df['label'].unique()):
            df_label = df[df['label'] == label]
            fig.add_trace(go.Scatter(
                x=df_label['x'], y=df_label['y'], mode='markers', marker=dict(size=5),
                name=f'Nhãn {label}', hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>Nhãn: %{customdata}<extra></extra>',
                customdata=df_label['label']
            ))
        fig.update_layout(title=f"Trực quan hóa {method} (2D)", xaxis_title="Thành phần 1", yaxis_title="Thành phần 2", showlegend=True)
    else:
        df = pd.DataFrame({
            'x': X_reduced[:, 0], 'y': X_reduced[:, 1], 'z': X_reduced[:, 2], 'label': y.astype(str)
        })
        fig = go.Figure()
        for label in sorted(df['label'].unique()):
            df_label = df[df['label'] == label]
            fig.add_trace(go.Scatter3d(
                x=df_label['x'], y=df_label['y'], z=df_label['z'], mode='markers', marker=dict(size=3),
                name=f'Nhãn {label}', hovertemplate='x: %{x:.2f}<br>y: %{y:.2f}<br>z: %{z:.2f}<br>Nhãn: %{customdata}<extra></extra>',
                customdata=df_label['label']
            ))
        fig.update_layout(title=f"Trực quan hóa {method} (3D)", scene=dict(xaxis_title="Thành phần 1", yaxis_title="Thành phần 2", zaxis_title="Thành phần 3"), showlegend=True)
    
    st.plotly_chart(fig, use_container_width=True)

def log_results(method, n_components, duration, X, y, X_reduced, model):
    mlflow.set_experiment("MNIST_Dimensionality_Reduction")
    log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    run_name = f"{method}_n={n_components}_{log_time}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("method", method)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("max_samples", len(X))
        mlflow.log_param("log_time", log_time)
        mlflow.log_metric("duration_seconds", duration)
        if method == "PCA":
            mlflow.log_metric("explained_variance_ratio", np.sum(model.explained_variance_ratio_))
        mlflow.sklearn.log_model(model, "model", input_example=X.reshape(X.shape[0], -1)[:1], signature=None)

def display_logs():
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name("MNIST_Dimensionality_Reduction")
    if not experiment:
        st.warning("Chưa có experiment 'MNIST_Dimensionality_Reduction'.")
        return
    
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) == 0:
        st.warning("Không có log nào trong experiment.")
        return
    
    data = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        method = run.data.params.get("method", "N/A")
        n_components = run.data.params.get("n_components", "N/A")
        max_samples = run.data.params.get("max_samples", "N/A")
        log_time = run.data.params.get("log_time", "N/A")
        duration = run.data.metrics.get("duration_seconds", np.nan)
        explained_variance = run.data.metrics.get("explained_variance_ratio", np.nan)
        data.append({
            "Run ID": run.info.run_id,
            "Tên Run": run_name,
            "Phương pháp": method,
            "Số chiều": n_components,
            "Số mẫu": max_samples,
            "Thời gian Log": log_time,
            "Thời gian chạy (giây)": duration,
            "Phương sai giải thích": explained_variance
        })
    
    df = pd.DataFrame(data, dtype='object')
    st.subheader("Log Các Lần Giảm Chiều")
    st.dataframe(df, hide_index=True, width=1200)

def dimensionality_reduction_app(X, y):
    st.title("🌐 Giảm Chiều Dữ liệu MNIST")
    total_samples = len(X)
    max_samples = st.slider("Chọn số lượng mẫu", 1000, 70000, 5000, step=1000, key='max_samples_ex4')
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X, y = X[indices], y[indices]
    if max_samples > 30000:
        st.warning('Số mẫu lớn (>30,000) có thể làm chậm t-SNE. Đề nghị giảm nếu cần.')

    method = st.selectbox("Chọn phương pháp", ["PCA", "t-SNE"], key='method_ex4')
    n_components = st.slider("Chọn số chiều", 2, 3, 2, key='n_components_ex4')

    if st.button("Giảm chiều", key='reduce_button_ex4'):
        X_reduced, model, duration = train_dimensionality_reduction(X, y, method, n_components)
        visualize_reduction(X_reduced, y, method, n_components)
        log_results(method, n_components, duration, X, y, X_reduced, model)
        
        st.success(f"Hoàn tất trong {duration:.2f} giây!")
        display_logs()  # Hiển thị log sau khi train