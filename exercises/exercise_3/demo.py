import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
import logging
import numpy as np
import datetime

# Thiết lập logging để debug nếu cần
logging.getLogger("mlflow").setLevel(logging.INFO)

# Thiết lập MLflow và DagsHub
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

# Hàm hiển thị và xóa log từ experiment, xử lý lỗi kiểu dữ liệu
@st.cache_data
def display_logs(_client, experiment_name):
    experiment = _client.get_experiment_by_name(experiment_name)
    if not experiment:
        st.warning(f"Chưa có experiment '{experiment_name}'. Sẽ tạo khi có log đầu tiên.")
        return None, None
    
    runs = _client.search_runs(experiment_ids=[experiment.experiment_id])
    if not runs:
        st.warning(f"Không có log nào trong experiment '{experiment_name}'.")
        return None, None
    
    data = []
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", run.info.run_id)
        model_type = run.data.params.get("model_type", "N/A")
        log_time = run.data.params.get("log_time", datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'))

        # Lấy các tham số cụ thể dựa trên model_type
        if model_type == "K-means":
            n_clusters = run.data.params.get("n_clusters", "N/A")
            eps = "N/A"
            min_samples = "N/A"
        elif model_type == "DBSCAN":
            n_clusters = run.data.metrics.get("n_clusters_found", "N/A")
            eps = run.data.params.get("eps", "N/A")
            min_samples = run.data.params.get("min_samples", "N/A")
        else:
            n_clusters = "N/A"
            eps = "N/A"
            min_samples = "N/A"

        # Lấy các metrics
        silhouette_train = run.data.metrics.get("silhouette_train", np.nan)
        silhouette_valid = run.data.metrics.get("silhouette_valid", np.nan)
        training_duration = run.data.metrics.get("training_duration", np.nan)

        # Chuyển đổi kiểu dữ liệu nếu cần
        silhouette_train = float(silhouette_train) if not pd.isna(silhouette_train) else np.nan
        silhouette_valid = float(silhouette_valid) if not pd.isna(silhouette_valid) else np.nan
        training_duration = float(training_duration) if not pd.isna(training_duration) else np.nan

        data.append({
            "Tên Run": run_name,
            "Loại Mô hình": model_type,
            "Thời gian Log": log_time,
            "Số cụm": n_clusters,
            "eps": eps,
            "min_samples": min_samples,
            "Silhouette Score (Train)": silhouette_train,
            "Silhouette Score (Valid)": silhouette_valid,
            "Thời gian huấn luyện (giây)": training_duration,
            "Run ID": run.info.run_id,
            "Experiment": experiment_name
        })
    
    df = pd.DataFrame(data, dtype='object')
    st.dataframe(df, hide_index=True, width=1200)
    return df, runs

# Hàm xóa log theo lựa chọn
def clear_selected_logs(client, selected_runs):
    if not selected_runs:
        st.warning("Vui lòng chọn ít nhất một run để xóa.")
        return
    
    with st.spinner("Đang xóa các run đã chọn..."):
        for run_id in selected_runs:
            client.delete_run(run_id)
        st.success(f"Đã xóa {len(selected_runs)} run thành công!")
    st.rerun()

# Giao diện Streamlit (chỉ hiển thị log huấn luyện)
def view_clustering_logs():
    st.title("📜 Xem và Quản lý Log Huấn luyện Clustering MNIST")

    # Tạo client MLflow
    client = MlflowClient()

    # Chỉ hiển thị log từ MNIST_Train_Clustering (huấn luyện)
    st.subheader("Log từ Experiment Huấn luyện (MNIST_Train_Clustering)")
    with st.spinner("Đang tải log huấn luyện..."):
        train_df, train_runs = display_logs(client, "MNIST_Train_Clustering")

    if train_df is not None and not train_df.empty:
        # Lấy danh sách Run ID từ dataframe huấn luyện
        train_run_ids = train_df["Run ID"].tolist()
        # Cho phép người dùng chọn run để xóa
        selected_train_runs = st.multiselect("Chọn các run huấn luyện để xóa", train_run_ids)
        if st.button("Xóa các run huấn luyện đã chọn", key="delete_train_runs"):
            clear_selected_logs(client, selected_train_runs)
