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

# Hàm hiển thị và xóa log từ experiment
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
        if experiment_name == "MNIST_Training":
            model_type = run.data.params.get("model_type", "N/A")
            log_time = run.data.params.get("log_time", datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S'))
            train_acc = run.data.metrics.get("train_accuracy", np.nan)
            valid_acc = run.data.metrics.get("valid_accuracy", np.nan)
            test_acc = run.data.metrics.get("test_accuracy", np.nan)
            training_duration = run.data.metrics.get("training_duration", np.nan)
            
            # Chuyển đổi kiểu dữ liệu
            train_acc = float(train_acc) if not pd.isna(train_acc) else np.nan
            valid_acc = float(valid_acc) if not pd.isna(valid_acc) else np.nan
            test_acc = float(test_acc) if not pd.isna(test_acc) else np.nan
            training_duration = float(training_duration) if not pd.isna(training_duration) else np.nan
            
            data.append({
                "Tên Run": run_name,
                "Loại Mô hình": model_type,
                "Thời gian Log": log_time,
                "Độ chính xác (Train)": train_acc,
                "Độ chính xác (Valid)": valid_acc,
                "Độ chính xác (Test)": test_acc,
                "Thời gian huấn luyện (giây)": training_duration,
                "Run ID": run.info.run_id,
                "Experiment": experiment_name
            })
        elif experiment_name == "MNIST_Demo":
            predicted_digit = run.data.params.get("predicted_digit", "N/A")
            confidence = run.data.params.get("confidence", np.nan)
            input_method = run.data.params.get("input_method", "N/A")
            model_run_id = run.data.params.get("model_run_id", "N/A")
            
            # Chuyển đổi kiểu dữ liệu
            confidence = float(confidence) if not pd.isna(confidence) else np.nan
            predicted_digit = str(predicted_digit)  # Đảm bảo kiểu dữ liệu đồng nhất
            
            data.append({
                "Tên Run": run_name,
                "Chữ số dự đoán": predicted_digit,
                "Độ tin cậy": confidence,
                "Phương thức nhập": input_method,
                "Model Run ID": model_run_id,
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

# Giao diện Streamlit
def view_logs():
    st.title("📜 Xem và Quản lý Logs MNIST")

    # Tạo client MLflow
    client = MlflowClient()

    # Hiển thị log từ MNIST_Training (huấn luyện)
    st.subheader("Log từ Experiment Huấn luyện (MNIST_Training)")
    with st.spinner("Đang tải log huấn luyện..."):
        train_df, train_runs = display_logs(client, "MNIST_Training")

    if train_df is not None and not train_df.empty:
        train_run_ids = train_df["Run ID"].tolist()
        selected_train_runs = st.multiselect("Chọn các run huấn luyện để xóa", train_run_ids)
        if st.button("Xóa các run huấn luyện đã chọn", key="delete_train_runs"):
            clear_selected_logs(client, selected_train_runs)

    # Hiển thị log từ MNIST_Demo (dự đoán)
    st.subheader("Log từ Experiment Dự đoán (MNIST_Demo)")
    with st.spinner("Đang tải log dự đoán..."):
        demo_df, demo_runs = display_logs(client, "MNIST_Demo")

    if demo_df is not None and not demo_df.empty:
        demo_run_ids = demo_df["Run ID"].tolist()
        selected_demo_runs = st.multiselect("Chọn các run dự đoán để xóa", demo_run_ids)
        if st.button("Xóa các run dự đoán đã chọn", key="delete_demo_runs"):
            clear_selected_logs(client, selected_demo_runs)
