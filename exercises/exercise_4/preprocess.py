import streamlit as st
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
import mlflow
import os
import dagshub

# Thiết lập thông tin DagsHub
DAGSHUB_USERNAME = "VietNam0410"
DAGSHUB_REPO = "vn0410"

try:
    with st.spinner("Đang kết nối với DagsHub..."):
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    st.success("Đã kết nối với DagsHub thành công!")
except Exception as e:
    st.error(f"Không thể kết nối với DagsHub: {str(e)}. Sử dụng MLflow cục bộ.")
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

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
    st.header("Tiền xử lý dữ liệu MNIST: PCA và t-SNE 🧮")

    # Giới thiệu PCA
    st.subheader("1. PCA (Principal Component Analysis)")
    st.write("""
    PCA là một phương pháp giảm chiều tuyến tính, biến đổi dữ liệu thành các thành phần chính (principal components) 
    sao cho giữ được tối đa phương sai của dữ liệu. Các thông số điều chỉnh:
    - **Số thành phần (n_components):** Số chiều dữ liệu giảm xuống (mặc định 2 để trực quan hóa).
    """)

    # Giới thiệu t-SNE
    st.subheader("2. t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    st.write("""
    t-SNE là một phương pháp phi tuyến tính, tập trung vào việc bảo tồn cấu trúc cục bộ của dữ liệu trong không gian 
    chiều thấp. Các thông số điều chỉnh:
    - **Perplexity:** Độ cân bằng giữa cấu trúc cục bộ và toàn cục (thường từ 5-50).
    - **Số lần lặp (n_iter):** Số lần tối ưu hóa (thường từ 250-1000).
    """)

    # Thiết lập experiment
    experiment_name = st.text_input("Nhập tên Experiment", value="MNIST_DimReduction")
    if experiment_name:
        with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
            mlflow.set_experiment(experiment_name)

    # Tải dữ liệu MNIST
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_data()
        st.success(f"Đã tải dữ liệu MNIST: {st.session_state['X_full'].shape[0]} mẫu, {st.session_state['X_full'].shape[1]} đặc trưng")

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']

    # Chọn số mẫu
    max_samples = st.slider("Chọn số mẫu để xử lý (ít hơn để nhanh hơn)", 100, X_full.shape[0], 1000)
    if max_samples < X_full.shape[0]:
        indices = np.random.choice(X_full.shape[0], max_samples, replace=False)
        X_subset = X_full[indices]
        y_subset = y_full[indices]
    else:
        X_subset = X_full
        y_subset = y_full

    # Chia tập dữ liệu: test trước, sau đó train/validation
    st.subheader("Chia tập dữ liệu")
    test_size = st.slider("Tỷ lệ tập kiểm tra (%)", 10, 50, 20) / 100
    valid_size = st.slider("Tỷ lệ tập validation (trong phần còn lại) (%)", 10, 50, 20) / 100

    if st.button("Chia dữ liệu"):
        with st.spinner("Đang chia dữ liệu..."):
            # Chia tập test trước
            X_temp, X_test, y_temp, y_test = train_test_split(X_subset, y_subset, test_size=test_size, random_state=42)
            # Chia tập train và validation từ phần còn lại
            valid_size_adjusted = valid_size / (1 - test_size)  # Điều chỉnh tỷ lệ validation
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size_adjusted, random_state=42)

            # Lưu vào session_state
            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

            st.success(f"Đã chia dữ liệu: Train {len(X_train)}, Validation {len(X_valid)}, Test {len(X_test)}")

            # Logging với MLflow
            with mlflow.start_run(run_name=f"Data_Split_{max_samples}"):
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("valid_size", valid_size)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("valid_samples", len(X_valid))
                mlflow.log_metric("test_samples", len(X_test))

if __name__ == "__main__":
    preprocess()