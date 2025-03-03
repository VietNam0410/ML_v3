import streamlit as st
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
import mlflow
import os
import dagshub
import datetime

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/vn0410.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "22fd02345f8ff45482a20960058627630acaf190"  # Thay bằng token cá nhân của bạn
    DAGSHUB_REPO = "vn0410"
    return DAGSHUB_REPO

# Cache dữ liệu MNIST
@st.cache_data
def load_mnist_data():
    with st.spinner("Đang tải dữ liệu MNIST từ OpenML..."):
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        X = X.values.reshape(-1, 28 * 28) / 255.0  # Chuẩn hóa về [0, 1]
        y = y.astype(np.int32)
    return X, y

def preprocess():
    st.header("Tiền xử lý Dữ liệu MNIST 🧮")

    # Gọi hàm mlflow_input để thiết lập MLflow tại DAGSHUB_MLFLOW_URI
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment", value="MNIST_Preprocess")
    with st.spinner("Đang thiết lập Experiment trên DagsHub MLflow..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                st.warning(f"Experiment '{experiment_name}' đã bị xóa trước đó. Vui lòng chọn tên khác hoặc khôi phục experiment qua DagsHub UI.")
                new_experiment_name = st.text_input("Nhập tên Experiment mới", value=f"{experiment_name}_Restored_{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                if new_experiment_name:
                    mlflow.set_experiment(new_experiment_name)
                    experiment_name = new_experiment_name
                else:
                    st.error("Vui lòng nhập tên experiment mới để tiếp tục.")
                    return
            else:
                mlflow.set_experiment(experiment_name)
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
            return

    # Tải dữ liệu MNIST
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_data()
        st.success(f"Đã tải dữ liệu MNIST: {st.session_state['X_full'].shape[0]} mẫu, {st.session_state['X_full'].shape[1]} đặc trưng")

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']

    # Chọn số mẫu
    max_samples = st.slider("Chọn số mẫu để xử lý (ít hơn để nhanh hơn)", 100, X_full.shape[0], 1000, key="max_samples_slider")
    if max_samples < X_full.shape[0]:
        indices = np.random.choice(X_full.shape[0], max_samples, replace=False)
        X_subset = X_full[indices]
        y_subset = y_full[indices]
    else:
        X_subset = X_full
        y_subset = y_full

    # Chia tập dữ liệu: test trước, sau đó train/validation
    st.subheader("Chia tập dữ liệu")
    test_size = st.slider("Tỷ lệ tập kiểm tra (%)", 10, 50, 20, key="test_size_slider") / 100
    remaining_size = 1 - test_size
    train_size_relative = st.slider("Tỷ lệ tập huấn luyện (trong phần còn lại) (%)", 10, 90, 70, key="train_size_slider") / 100

    # Tính toán tỷ lệ thực tế
    train_size = remaining_size * train_size_relative
    valid_size = remaining_size * (1 - train_size_relative)

    # Hiển thị tỷ lệ thực tế
    st.write(f"Tỷ lệ thực tế: Huấn luyện {train_size*100:.1f}%, Validation {valid_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%")
    st.write(f"Kiểm tra tổng tỷ lệ: {train_size*100 + valid_size*100 + test_size*100:.1f}% (phải luôn bằng 100%)")

    # Cho phép người dùng đặt tên run ID cho việc chia dữ liệu
    run_name = st.text_input("Nhập tên Run ID cho việc chia dữ liệu (để trống để tự động tạo)", value="", max_chars=20, key="data_split_run_name_input")
    if run_name.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_DataSplit_{timestamp.replace(' ', '_').replace(':', '-')}"  # Định dạng tên run hợp lệ cho MLflow

    if st.button("Chia dữ liệu"):
        with st.spinner("Đang chia dữ liệu..."):
            # Chia tập test trước
            X_temp, X_test, y_temp, y_test = train_test_split(X_subset, y_subset, test_size=test_size, random_state=42)
            # Chia tập train và validation từ phần còn lại
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, train_size=train_size_relative, random_state=42)

            st.success(f"Đã chia dữ liệu: Train {len(X_train)}, Validation {len(X_valid)}, Test {len(X_test)}")

            # Lưu dữ liệu ban đầu vào session_state
            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

            # Logging việc chia dữ liệu vào MLflow tại DAGSHUB_MLFLOW_URI
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("run_id", run.info.run_id)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("valid_size", valid_size)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("valid_samples", len(X_valid))
                mlflow.log_metric("test_samples", len(X_test))

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']
                st.success(f"Dữ liệu đã được chia và log vào DagsHub MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {timestamp})")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

if __name__ == "__main__":
    preprocess()