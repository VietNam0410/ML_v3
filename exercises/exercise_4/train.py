import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import mlflow
import mlflow.sklearn
import os
import datetime

DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

@st.cache_resource
def get_scaler():
    return StandardScaler()

def train_dimensionality_reduction(X, y, method, n_components):
    scaler = get_scaler()
    X_scaled = scaler.fit_transform(X.reshape(X.shape[0], -1))
    
    start_time = datetime.datetime.now()
    if method == "PCA":
        model = PCA(n_components=n_components)
    elif method == "t-SNE":
        model = TSNE(n_components=n_components, perplexity=15, n_iter=500, random_state=42, n_jobs=-1)
    
    X_reduced = model.fit_transform(X_scaled)
    duration = (datetime.datetime.now() - start_time).total_seconds()
    
    return X_reduced, model, duration

def visualize_reduction(X_reduced, y, method, n_components):
    if n_components == 2:
        fig = px.scatter(X_reduced, x=0, y=1, color=y, labels={'color': 'Nhãn'},
                         title=f"Trực quan hóa {method} (2D)", color_continuous_scale='Viridis')
    else:
        fig = px.scatter_3d(X_reduced, x=0, y=1, z=2, color=y, labels={'color': 'Nhãn'},
                            title=f"Trực quan hóa {method} (3D)", color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)

def log_results(method, n_components, duration, X, y, X_reduced, model):
    mlflow.set_experiment("MNIST_Dimensionality_Reduction")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{method}_n={n_components}_{timestamp}"
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("method", method)
        mlflow.log_param("n_components", n_components)
        mlflow.log_param("max_samples", len(X))
        mlflow.log_metric("duration_seconds", duration)
        if method == "PCA":
            mlflow.log_metric("explained_variance_ratio", np.sum(model.explained_variance_ratio_))
        mlflow.sklearn.log_model(model, "model", input_example=X.reshape(X.shape[0], -1)[:1])

def dimensionality_reduction_app(X, y):
    st.title("🌐 Giảm Chiều Dữ Liệu MNIST")
    total_samples = len(X)
    max_samples = st.slider("Chọn số lượng mẫu", 0, total_samples, min(10000, total_samples), step=1000)
    if max_samples == 0:
        max_samples = total_samples
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X, y = X[indices], y[indices]

    method = st.selectbox("Chọn phương pháp", ["PCA", "t-SNE"])
    n_components = st.slider("Chọn số chiều", 2, 3, 2)

    if st.button("Giảm chiều"):
        X_reduced, model, duration = train_dimensionality_reduction(X, y, method, n_components)
        visualize_reduction(X_reduced, y, method, n_components)
        log_results(method, n_components, duration, X, y, X_reduced, model)
        st.success(f"Hoàn tất trong {duration:.2f} giây!")