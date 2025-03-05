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
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

# Hàm tải dữ liệu MNIST với cache
@st.cache_data
def load_mnist_from_openml():
    """Tải dữ liệu MNIST từ OpenML hoặc TensorFlow và lưu vào bộ nhớ đệm."""
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

def preprocess_mnist():
    st.header("Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Khởi tạo DagsHub/MLflow chỉ một lần
    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_REPO = mlflow_input()
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_REPO
    else:
        DAGSHUB_REPO = st.session_state['mlflow_url']

    # Thiết lập experiment "MNIST_Preprocessing" cố định
    experiment_name = "MNIST_Preprocessing"
    with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if not experiment:
                client.create_experiment(experiment_name)
                st.success(f"Đã tạo Experiment mới '{experiment_name}' thành công!")
            elif experiment and experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                st.success(f"Đã khôi phục Experiment '{experiment_name}' thành công!")
            mlflow.set_experiment(experiment_name)  # Đặt experiment cố định
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
            return

    # Tải dữ liệu từ bộ nhớ đệm nếu chưa có trong session_state
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_from_openml()
        st.success("Dữ liệu MNIST đã được tải và chuẩn hóa thành công! ✅")

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']
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
            processed_file = os.path.join(processed_dir, "mnist_processed.npz")

            # Lưu dữ liệu cục bộ và log vào MLflow
            with st.spinner("Đang lưu và log dữ liệu đã chia..."):
                np.savez(processed_file, 
                         X_train=X_train, y_train=y_train,
                         X_valid=X_valid, y_valid=y_valid,
                         X_test=X_test, y_test=y_test)
                st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

            # Logging vào MLflow/DagsHub trong experiment "MNIST_Preprocessing"
            with mlflow.start_run(run_name=f"MNIST_Data_Split_{max_samples}_Samples") as run:
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("test_size", test_size)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("valid_samples", len(X_valid))
                mlflow.log_metric("test_samples", len(X_test))

                mlflow.log_artifact(processed_file, artifact_path="processed_data")
                # Xóa file tạm sau khi log (tuỳ chọn)
                os.remove(processed_file)

                run_id = run.info.run_id
                mlflow_uri = DAGSHUB_REPO
                st.success("Dữ liệu đã được chia và log vào MLflow trong 'MNIST_Preprocessing' ✅.")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

if __name__ == "__main__":
    preprocess_mnist()