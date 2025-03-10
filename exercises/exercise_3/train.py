import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
import os
import datetime

@st.cache_resource
def get_scaler():
    return StandardScaler()

@st.cache_resource
def get_model(model_choice, **params):
    if model_choice == 'K-means':
        return KMeans(**params)
    else:
        return DBSCAN(**params)

def mlflow_input():
    DAGSHUB_MLFLOW_URI = 'https://dagshub.com/VietNam0410/ML_v3.mlflow'
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'VietNam0410'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b'
    return 'ML_v3'

def train_clustering(X_full, y_full):
    st.header('Huấn luyện Clustering trên MNIST 🧮')
    if mlflow.active_run():
        mlflow.end_run()

    DAGSHUB_REPO = mlflow_input()
    if DAGSHUB_REPO is None:
        st.error('Lỗi kết nối MLflow.')
        return

    mlflow.set_experiment('MNIST_Train_Clustering')

    total_samples = len(X_full)
    max_samples = st.slider('Số mẫu tối đa', 0, total_samples, min(10000, total_samples), step=100)
    if max_samples == 0:
        max_samples = total_samples
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X_full = X_full[indices]
        y_full = y_full[indices]

    test_size = st.slider('Tỷ lệ kiểm tra (%)', 10, 50, 20, step=5) / 100
    train_size_relative = st.slider('Tỷ lệ huấn luyện (%)', 50, 90, 70, step=5) / 100
    val_size_relative = 1 - train_size_relative

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_remaining, y_remaining, train_size=train_size_relative, random_state=42, stratify=y_remaining
    )

    st.write(f'Tổng số mẫu: {max_samples}')
    st.write(f'Tập huấn luyện: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu')

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)

    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_valid_scaled = scaler.transform(X_valid_flat)

    model_choice = st.selectbox('Chọn thuật toán', ['K-means', 'DBSCAN'])
    if model_choice == 'K-means':
        n_clusters = st.slider('Số cụm (K)', 2, 20, 10)
        model_params = {'n_clusters': n_clusters, 'random_state': 42}
    else:
        eps = st.slider('Khoảng cách tối đa (eps)', 0.1, 2.0, 0.5, step=0.1)
        min_samples = st.slider('Số điểm tối thiểu', 2, 20, 5)
        model_params = {'eps': eps, 'min_samples': min_samples}

    run_name = st.text_input('Nhập tên Run ID', value='')
    if not run_name.strip():
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'MNIST_{model_choice}_{timestamp}'

    if st.button('Huấn luyện'):
        with st.spinner(f'Đang huấn luyện {model_choice}...'):
            model = get_model(model_choice, **model_params)
            model.fit(X_train_scaled)
            labels_train = model.predict(X_train_scaled) if model_choice == 'K-means' else model.fit_predict(X_train_scaled)
            labels_valid = model.predict(X_valid_scaled) if model_choice == 'K-means' else model.fit_predict(X_valid_scaled)
            n_clusters_found = len(np.unique(labels_valid)) - (1 if -1 in labels_valid else 0)

            silhouette_train = silhouette_score(X_train_scaled, labels_train) if n_clusters_found > 1 else -1
            silhouette_valid = silhouette_score(X_valid_scaled, labels_valid) if n_clusters_found > 1 else -1

            with mlflow.start_run(run_name=run_name):
                mlflow.log_param('model_type', model_choice)
                mlflow.log_params(model_params)
                mlflow.log_metric('silhouette_train', silhouette_train)
                mlflow.log_metric('silhouette_valid', silhouette_valid)
                mlflow.sklearn.log_model(model, 'model', input_example=X_train_scaled[:1])
                run_id = mlflow.active_run().info.run_id

            st.write(f'**Thuật toán**: {model_choice}')
            st.write(f'Silhouette Score: Train: {silhouette_train:.4f}, Valid: {silhouette_valid:.4f}')
            st.success(f'Huấn luyện hoàn tất! Run ID: {run_id}')