import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Backup nếu OpenML không tải được
import pandas as pd
import mlflow
import os

# Thiết lập MLflow Tracking URI cục bộ
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

# Sử dụng st.cache_data để cache dữ liệu, tăng tốc độ load
@st.cache_data
def load_mnist_from_openml():
    try:
        # Tải dữ liệu MNIST từ OpenML (ID dataset MNIST là 554)
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        
        # Chuyển đổi X (DataFrame) và y (Series) thành mảng numpy và chuẩn hóa (28x28x1)
        X = X.values.reshape(-1, 28, 28, 1) / 255.0  # Sử dụng .values để lấy mảng numpy từ DataFrame
        y = y.values.astype(np.int32)  # Chuyển y từ Series thành numpy array
        
        return X, y
    except Exception as e:
        st.error(f"Không thể tải dữ liệu từ OpenML. Sử dụng dữ liệu từ TensorFlow: {str(e)}")
        # Fallback: Tải từ TensorFlow nếu OpenML thất bại
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0) / 255.0
        y = np.concatenate([y_train, y_test], axis=0)
        return X.reshape(-1, 28, 28, 1), y

def preprocess_mnist():
    st.header("Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Preprocessing")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Khởi tạo session_state để lưu dữ liệu tạm thời
    if 'mnist_data' not in st.session_state:
        st.session_state['mnist_data'] = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = {}

    # Tải dữ liệu MNIST từ OpenML (sử dụng cache)
    if st.button("Tải dữ liệu MNIST từ OpenML"):
        # Sửa đổi bởi Grok 3: Tải trực tiếp từ OpenML, sử dụng st.cache_data, đảm bảo trả về numpy array
        X_full, y_full = load_mnist_from_openml()
        
        # Kiểm tra và kết thúc run hiện tại nếu có
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        # Bắt đầu một run MLflow để log tiền xử lý
        with mlflow.start_run(run_name=f"MNIST_Preprocessing_{experiment_name}"):
            # Lưu dữ liệu đầy đủ vào session và log vào MLflow
            st.session_state['mnist_data'] = {
                'X_full': X_full,
                'y_full': y_full
            }
            st.session_state['preprocessing_steps'] = {"loaded": True}

            # Log metadata
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("total_samples", len(X_full))
            mlflow.log_param("data_shape", X_full.shape)
            mlflow.log_text(X_full[0].tobytes(), "sample_image_0.npy")  # Lưu mẫu hình ảnh đầu tiên
            mlflow.log_param("sample_label_0", y_full[0])

            st.success("Dữ liệu MNIST đã được tải từ OpenML, chuẩn hóa và log vào MLflow thành công! ✅")

    # Kiểm tra và hiển thị trạng thái session
    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.warning("Vui lòng nhấn nút 'Tải dữ liệu MNIST từ OpenML' để tải dữ liệu trước khi tiếp tục. ⚠️")
        return

    # Kiểm tra key 'X_full' và 'y_full' trước khi truy cập
    mnist_data = st.session_state['mnist_data']
    if 'X_full' not in mnist_data or 'y_full' not in mnist_data:
        st.error("Dữ liệu 'X_full' hoặc 'y_full' không tồn tại trong session. Vui lòng tải lại dữ liệu MNIST bằng cách nhấn nút 'Tải dữ liệu MNIST từ OpenML'.")
        return

    # Hiển thị thông tin dữ liệu đầy đủ
    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ 🔍")
    st.write(f"Tổng số lượng mẫu: {len(mnist_data['X_full'])}")
    st.write("Hình ảnh mẫu (đầu tiên):")
    st.image(mnist_data['X_full'][0].reshape(28, 28), caption=f"Chữ số: {mnist_data['y_full'][0]}", width=100)

    # Chia tách dữ liệu theo lựa chọn của người dùng
    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    train_size = st.slider("Chọn kích thước tập huấn luyện (%)", min_value=10, max_value=90, value=70, step=5) / 100
    val_size = st.slider("Chọn kích thước tập validation (%)", min_value=0, max_value=30, value=15, step=5) / 100
    test_size = 1 - train_size - val_size  # Tính kích thước tập kiểm tra

    if test_size < 0:
        st.error("Tổng kích thước tập huấn luyện và validation không được vượt quá 100%. Vui lòng điều chỉnh lại.")
    else:
        if st.button("Chia dữ liệu"):
            X_full = mnist_data['X_full']
            y_full = mnist_data['y_full']

            # Kiểm tra và kết thúc run hiện tại nếu có
            # Sửa đổi bởi Grok 3: Thêm log chia dữ liệu vào MLflow
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()

            with mlflow.start_run(run_name=f"MNIST_Split_{experiment_name}", nested=True):
                # Chia dữ liệu thành tập huấn luyện, validation, và kiểm tra
                X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=42)
                X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=val_size/(train_size+val_size), random_state=42)

                # Lưu vào session_state và log vào MLflow
                st.session_state['mnist_data'] = {
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_valid': X_valid,
                    'y_valid': y_valid,
                    'X_test': X_test,
                    'y_test': y_test
                }
                st.session_state['preprocessing_steps']['split'] = {
                    "train_size": train_size,
                    "val_size": val_size,
                    "test_size": test_size
                }

                # Log các tham số chia dữ liệu
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_valid))
                mlflow.log_param("test_samples", len(X_test))
                mlflow.log_text(X_train[0].tobytes(), "train_sample_0.npy")  # Lưu mẫu từ tập huấn luyện

                st.success(f"Đã chia dữ liệu với kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}% và log vào MLflow! ✅")

                st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
                st.write(f"Tập validation: {len(X_valid)} mẫu")
                st.write(f"Tập kiểm tra: {len(X_test)} mẫu")
                st.write("Hình ảnh mẫu từ tập huấn luyện:", X_train[0].reshape(28, 28))
                st.write(f"Chữ số thực tế: {y_train[0]}")  # Sử dụng numpy array, không cần thay đổi

    # Lưu dữ liệu đã tiền xử lý vào MLflow (không lưu file cục bộ)
    st.write("### Lưu dữ liệu đã tiền xử lý vào MLflow 💾")
    if st.button("Lưu dữ liệu đã xử lý vào MLflow 📋"):
        # Kiểm tra và kết thúc run hiện tại nếu có
        # Sửa đổi bởi Grok 3: Thêm log dữ liệu đã xử lý vào MLflow
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"MNIST_Processed_{experiment_name}", nested=True):
            processed_data = st.session_state['mnist_data']
            mlflow.log_param("final_data_shape", processed_data.get('X_train', processed_data['X_full']).shape)
            mlflow.log_param("final_samples", len(processed_data.get('X_train', processed_data['X_full'])))
            mlflow.log_text(processed_data.get('X_train', processed_data['X_full'])[0].tobytes(), "final_sample_0.npy")

            # Log các bước tiền xử lý
            mlflow.log_params(st.session_state['preprocessing_steps'])
            st.success("Dữ liệu đã được tiền xử lý và log vào MLflow thành công! ✅")

            st.subheader("Xem trước dữ liệu đã xử lý trong MLflow 🔚")
            st.write(st.session_state['mnist_data'])

if __name__ == "__main__":
    preprocess_mnist()