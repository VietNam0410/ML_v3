import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
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
def load_mnist_from_openml():
    with st.spinner("Đang tải dữ liệu MNIST..."):
        try:
            dataset = openml.datasets.get_dataset(554)
            X, y, _, _ = dataset.get_data(target='class')
            X = X.values.reshape(-1, 28, 28, 1) / 255.0
            y = y.astype(np.int32)
            return X, y
        except Exception as e:
            st.error(f"Không thể tải dữ liệu từ OpenML. Sử dụng dữ liệu từ TensorFlow: {str(e)}")
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X = np.concatenate([X_train, X_test], axis=0) / 255.0
            y = np.concatenate([y_train, y_test], axis=0)
            return X.reshape(-1, 28, 28, 1), y

def preprocess_mnist_clustering():
    st.header("Tiền xử lý Dữ liệu MNIST cho Clustering 🖌️")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow tại DAGSHUB_MLFLOW_URI
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Clustering_Preprocessing")
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
    if 'X_full_clustering' not in st.session_state or 'y_full_clustering' not in st.session_state:
        st.session_state['X_full_clustering'], st.session_state['y_full_clustering'] = load_mnist_from_openml()
        st.success("Dữ liệu MNIST đã được tải và chuẩn hóa thành công! ✅")

    X_full = st.session_state['X_full_clustering']
    y_full = st.session_state['y_full_clustering']
    total_samples = len(X_full)

    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ 🔍")
    st.write(f"Tổng số lượng mẫu: {total_samples}")

    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    max_samples = st.slider("Chọn số lượng mẫu tối đa (0 để dùng toàn bộ)", 0, total_samples, total_samples, step=100)
    
    if max_samples == 0:
        max_samples = total_samples
    elif max_samples > total_samples:
        st.error(f"Số lượng mẫu ({max_samples}) vượt quá tổng số mẫu có sẵn ({total_samples}). Đặt lại về {total_samples}.")
        max_samples = total_samples

    test_size = st.slider("Chọn tỷ lệ tập kiểm tra (%)", min_value=10, max_value=50, value=20, step=5) / 100
    remaining_size = 1 - test_size
    train_size_relative = st.slider(
        "Chọn tỷ lệ tập huấn luyện (% trên phần còn lại sau khi trừ tập test)",
        min_value=10, max_value=90, value=70, step=5
    ) / 100

    # Tính toán tỷ lệ tập train và validation dựa trên phần còn lại (remaining_size)
    train_size = remaining_size * train_size_relative
    val_size = remaining_size * (1 - train_size_relative)

    # Hiển thị tỷ lệ thực tế dựa trên toàn bộ dữ liệu
    st.write(f"Tỷ lệ thực tế: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%")
    st.write(f"Kiểm tra tổng tỷ lệ: {train_size*100 + val_size*100 + test_size*100:.1f}% (phải luôn bằng 100%)")

    # Cho phép người dùng đặt tên run ID cho việc chia dữ liệu
    run_name = st.text_input("Nhập tên Run ID cho việc chia dữ liệu (để trống để tự động tạo)", value="", max_chars=20, key="data_split_run_name_input")
    if run_name.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_Clustering_DataSplit_{timestamp.replace(' ', '_').replace(':', '-')}"  # Định dạng tên run hợp lệ cho MLflow

    if st.button("Chia dữ liệu"):
        with st.spinner("Đang chia dữ liệu..."):
            if max_samples < total_samples:
                indices = np.random.choice(total_samples, max_samples, replace=False)
                X_subset = X_full[indices]
                y_subset = y_full[indices]
            else:
                X_subset = X_full
                y_subset = y_full

            # Chia tập test trước
            X_remaining, X_test, y_remaining, y_test = train_test_split(
                X_subset, y_subset, test_size=test_size, random_state=42
            )
            # Chia tập train và validation từ phần còn lại (remaining_size)
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_remaining, y_remaining, train_size=train_size_relative, random_state=42
            )

            st.success(f"Đã chia dữ liệu với số lượng mẫu: {max_samples}. Kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%! ✅")
            st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
            st.write(f"Tập validation: {len(X_valid)} mẫu")
            st.write(f"Tập kiểm tra: {len(X_test)} mẫu")

            # Đảm bảo thư mục tồn tại trước khi lưu file
            processed_dir = "exercises/exercise_mnist/data/processed"
            os.makedirs(processed_dir, exist_ok=True)
            processed_file = os.path.join(processed_dir, "mnist_clustering_processed.npz")

            # Lưu dữ liệu cục bộ
            with st.spinner("Đang lưu dữ liệu đã chia..."):
                np.savez(processed_file, 
                         X_train=X_train, y_train=y_train,
                         X_valid=X_valid, y_valid=y_valid,
                         X_test=X_test, y_test=y_test)
                st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

            # Logging vào MLflow tại DAGSHUB_MLFLOW_URI
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("run_id", run.info.run_id)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("test_size", test_size)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("valid_samples", len(X_valid))
                mlflow.log_metric("test_samples", len(X_test))

                # Log file dữ liệu đã chia làm artifact
                mlflow.log_artifact(processed_file, artifact_path="processed_data")
                os.remove(processed_file)  # Xóa file cục bộ sau khi log

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']
                st.success(f"Dữ liệu đã được chia, lưu cục bộ, và log vào DagsHub MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {timestamp})")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

            # Lưu dữ liệu vào session_state để sử dụng sau
            st.session_state['mnist_clustering_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

if __name__ == "__main__":
    preprocess_mnist_clustering()