import streamlit as st
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import os
import logging
import numpy as np  # Thêm import numpy
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
        method = run.data.params.get("method", "N/A")
        n_components = run.data.params.get("n_components", "N/A")
        source = run.data.params.get("source", "N/A")
        timestamp = run.data.params.get("timestamp", "N/A")
        confidence = run.data.metrics.get("confidence", np.nan)
        explained_variance = run.data.metrics.get("explained_variance_ratio", np.nan)
        position = [run.data.metrics.get(f"position_{i}", np.nan) for i in range(3)]
        data.append({
            "Tên Run": run_name,
            "Phương pháp": method,
            "Số chiều": n_components,
            "Nguồn": source,
            "Thời gian": timestamp,
            "Vị trí": str(position[:int(n_components) if n_components != "N/A" else 2]),  # Chỉ lấy số chiều tương ứng
            "Độ tin cậy": confidence,  # Sử dụng np.nan thay vì "N/A"
            "Phương sai": explained_variance,  # Sử dụng np.nan thay vì "N/A"
            "Run ID": run.info.run_id
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
def view_logs_app():
    st.title("📜 Xem và Quản lý Log Huấn luyện Giảm Chiều MNIST")

    # Tạo client MLflow
    client = MlflowClient()

    # Chỉ hiển thị log từ MNIST_Dimensionality_Reduction (huấn luyện)
    st.subheader("Log từ Experiment Huấn luyện (MNIST_Dimensionality_Reduction)")
    with st.spinner("Đang tải log huấn luyện..."):
        train_df, train_runs = display_logs(client, "MNIST_Dimensionality_Reduction")

    if train_df is not None and not train_df.empty:
        # Lấy danh sách Run ID từ dataframe huấn luyện
        train_run_ids = train_df["Run ID"].tolist()
        # Cho phép người dùng chọn run để xóa
        selected_train_runs = st.multiselect("Chọn các run huấn luyện để xóa", train_run_ids)
        if st.button("Xóa các run huấn luyện đã chọn", key="delete_train_runs"):
            clear_selected_logs(client, selected_train_runs)

    # Thêm nút làm mới cache với key duy nhất
    if st.button("🔄 Làm mới dữ liệu", key=f"refresh_data_{datetime.datetime.now().microsecond}"):
        st.cache_data.clear()
        st.rerun()

# Hàm chính
if __name__ == "__main__":
    view_logs_app()