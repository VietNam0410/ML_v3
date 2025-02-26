import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Backup nếu OpenML không tải được
import pandas as pd
import mlflow
import os
import tempfile

# Thiết lập MLflow Tracking URI cục bộ
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

# Sử dụng st.cache_data để cache dữ liệu, tăng tốc độ load
@st.cache_data
def load_mnist_from_openml(max_samples=None):
    try:
        # Tải dữ liệu MNIST từ OpenML (ID dataset MNIST là 554)
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        
        # Giới hạn số lượng mẫu nếu max_samples được chỉ định
        if max_samples and max_samples < len(X):
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X.iloc[indices]
            y = y.iloc[indices]

        # Chuyển đổi X (DataFrame) và y (Series) thành mảng numpy và chuẩn hóa (28x28x1)
        X = X.values.reshape(-1, 28, 28, 1) / 255.0  # Sử dụng .values để lấy mảng numpy từ DataFrame
        y = y.values.astype(np.int32)  # Chuyển y từ Series thành numpy array
        
        return X, y
    except Exception as e:
        st.error(f"Không thể tải dữ liệu từ OpenML. Sử dụng dữ liệu từ TensorFlow: {str(e)}")
        # Fallback: Tải từ TensorFlow nếu OpenML thất bại, giới hạn kích thước nếu cần
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0) / 255.0
        y = np.concatenate([y_train, y_test], axis=0)
        
        if max_samples and max_samples < len(X):
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
            y = y[indices]
        
        return X.reshape(-1, 28, 28, 1), y

def preprocess_mnist():
    st.header("Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Preprocessing")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Cho phép người dùng chọn kích thước dữ liệu tối đa để giảm bộ nhớ
    # Sửa đổi bởi Grok 3: Tăng max_samples lên 40,000
    max_samples = st.slider("Chọn số lượng mẫu tối đa để tải (0 để tải tất cả)", 0, 40000, 20000, step=1000)
    max_samples = max_samples if max_samples > 0 else None  # 0 nghĩa là tải toàn bộ

    # Tải dữ liệu MNIST từ OpenML (sử dụng cache)
    if st.button("Tải dữ liệu MNIST từ OpenML"):
        # Sửa đổi bởi Grok 3: Tải trực tiếp từ OpenML, giới hạn kích thước, sử dụng st.cache_data
        X_full, y_full = load_mnist_from_openml(max_samples)
        
        # Kiểm tra và kết thúc run hiện tại nếu có
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        # Bắt đầu một run MLflow để log tiền xử lý
        with mlflow.start_run(run_name=f"MNIST_Preprocessing_{experiment_name}"):
            # Log dữ liệu đầy đủ vào MLflow (không dùng session_state)
            mlflow.log_param("experiment_name", experiment_name)
            mlflow.log_param("total_samples", len(X_full))
            mlflow.log_param("data_shape", X_full.shape)
            mlflow.log_param("max_samples_used", max_samples if max_samples else "all")
            
            # Log 5 nhãn đầu tiên
            for i in range(5):
                mlflow.log_param(f"sample_label_{i}", y_full[i])

            # Lưu dữ liệu đầy đủ vào MLflow dưới dạng file .npy
            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_x:
                np.save(tmp_x.name, X_full)
                mlflow.log_artifact(tmp_x.name, "data_full.npy")
            os.unlink(tmp_x.name)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_y:
                np.save(tmp_y.name, y_full)
                mlflow.log_artifact(tmp_y.name, "labels_full.npy")
            os.unlink(tmp_y.name)

            # Lưu 5 mẫu hình ảnh đầu tiên dưới dạng file .npy và log bằng mlflow.log_artifact
            # Sửa đổi bởi Grok 3: Hiển thị 5 ảnh đầu tiên
            for i in range(5):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
                    np.save(tmp.name, X_full[i])
                    mlflow.log_artifact(tmp.name, f"sample_image_{i}.npy")
                os.unlink(tmp.name)

            st.success(f"Dữ liệu MNIST đã được tải từ OpenML, chuẩn hóa (giới hạn {len(X_full)} mẫu) và log vào MLflow thành công! ✅")

    # Kiểm tra dữ liệu từ MLflow (thay vì session_state)
    # Sửa đổi bởi Grok 3: Kiểm tra dữ liệu từ MLflow thay vì session
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    if runs.empty:
        st.warning("Chưa có dữ liệu tiền xử lý nào được log vào MLflow. Vui lòng tải và xử lý dữ liệu trước.")
        return

    latest_run_id = runs['run_id'].iloc[0]
    run = mlflow.get_run(latest_run_id)
    X_full_shape = run.data.params.get("data_shape", "Unknown")
    total_samples = run.data.params.get("total_samples", "Unknown")

    # Hiển thị thông tin dữ liệu từ MLflow
    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ từ MLflow 🔍")
    st.write(f"Tổng số lượng mẫu: {total_samples}")
    st.write("5 Hình ảnh mẫu đầu tiên đã log trong MLflow:")
    
    # Sửa đổi bởi Grok 3: Tải và hiển thị 5 ảnh đầu tiên từ MLflow, sửa lỗi IsADirectoryError
    artifacts_dir = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="")
    for i in range(5):
        sample_file_path = os.path.join(artifacts_dir, f"sample_image_{i}.npy")
        if os.path.isfile(sample_file_path):  # Đảm bảo là file, không phải thư mục
            sample_image = np.load(sample_file_path).reshape(28, 28)
            label = run.data.params.get(f"sample_label_{i}", "Unknown")
            st.image(sample_image, caption=f"Chữ số: {label}", width=100)
        else:
            st.error(f"Không tìm thấy file mẫu {sample_file_path}. Vui lòng kiểm tra MLflow.")

    # Chia tách dữ liệu theo lựa chọn của người dùng
    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    train_size = st.slider("Chọn kích thước tập huấn luyện (%)", min_value=10, max_value=90, value=70, step=5) / 100
    val_size = st.slider("Chọn kích thước tập validation (%)", min_value=0, max_value=30, value=15, step=5) / 100
    test_size = 1 - train_size - val_size  # Tính kích thước tập kiểm tra

    if test_size < 0:
        st.error("Tổng kích thước tập huấn luyện và validation không được vượt quá 100%. Vui lòng điều chỉnh lại.")
    else:
        if st.button("Chia dữ liệu"):
            # Tải dữ liệu đầy đủ từ MLflow
            # Sửa đổi bởi Grok 3: Load X_full và y_full từ MLflow hoặc cache
            artifacts_dir = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="")
            X_full_path = os.path.join(artifacts_dir, "data_full.npy")
            y_full_path = os.path.join(artifacts_dir, "labels_full.npy")
            
            X_full = np.load(X_full_path)
            y_full = np.load(y_full_path)

            # Kiểm tra và kết thúc run hiện tại nếu có
            active_run = mlflow.active_run()
            if active_run:
                mlflow.end_run()

            with mlflow.start_run(run_name=f"MNIST_Split_{experiment_name}", nested=True):
                # Chia dữ liệu thành tập huấn luyện, validation, và kiểm tra
                X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=42)
                X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=val_size/(train_size+val_size), random_state=42)

                # Log dữ liệu chia tách vào MLflow (không dùng session_state)
                mlflow.log_param("train_size", train_size)
                mlflow.log_param("val_size", val_size)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("train_samples", len(X_train))
                mlflow.log_param("val_samples", len(X_valid))
                mlflow.log_param("test_samples", len(X_test))

                # Lưu 5 mẫu từ tập huấn luyện và kiểm tra dưới dạng file .npy và log bằng mlflow.log_artifact
                for i in range(5):  # Lưu 5 mẫu đầu tiên từ mỗi tập
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_train:
                        np.save(tmp_train.name, X_train[i])
                        mlflow.log_artifact(tmp_train.name, f"train_sample_{i}.npy")
                    os.unlink(tmp_train.name)

                    with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_test:
                        np.save(tmp_test.name, X_test[i])
                        mlflow.log_artifact(tmp_test.name, f"test_sample_{i}.npy")
                    os.unlink(tmp_test.name)

                    # Log nhãn tương ứng
                    mlflow.log_param(f"train_label_{i}", y_train[i])
                    mlflow.log_param(f"test_label_{i}", y_test[i])

                st.success(f"Đã chia dữ liệu với kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}% và log vào MLflow! ✅")

                st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
                st.write(f"Tập validation: {len(X_valid)} mẫu")
                st.write(f"Tập kiểm tra: {len(X_test)} mẫu")
                st.write("5 Hình ảnh mẫu từ tập huấn luyện:", [X_train[i].reshape(28, 28) for i in range(5)])
                for i in range(5):
                    st.write(f"Chữ số thực tế mẫu {i}: {y_train[i]}")

                # Lưu dữ liệu chia tách vào MLflow để dùng cho train.py
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_x_train:
                    np.save(tmp_x_train.name, X_train)
                    mlflow.log_artifact(tmp_x_train.name, "X_train.npy")
                os.unlink(tmp_x_train.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_y_train:
                    np.save(tmp_y_train.name, y_train)
                    mlflow.log_artifact(tmp_y_train.name, "y_train.npy")
                os.unlink(tmp_y_train.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_x_valid:
                    np.save(tmp_x_valid.name, X_valid)
                    mlflow.log_artifact(tmp_x_valid.name, "X_valid.npy")
                os.unlink(tmp_x_valid.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_y_valid:
                    np.save(tmp_y_valid.name, y_valid)
                    mlflow.log_artifact(tmp_y_valid.name, "y_valid.npy")
                os.unlink(tmp_y_valid.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_x_test:
                    np.save(tmp_x_test.name, X_test)
                    mlflow.log_artifact(tmp_x_test.name, "X_test.npy")
                os.unlink(tmp_x_test.name)

                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp_y_test:
                    np.save(tmp_y_test.name, y_test)
                    mlflow.log_artifact(tmp_y_test.name, "y_test.npy")
                os.unlink(tmp_y_test.name)

    # Lưu dữ liệu đã tiền xử lý vào MLflow (không lưu file cục bộ hoặc session)
    st.write("### Lưu dữ liệu đã tiền xử lý vào MLflow 💾")
    if st.button("Lưu dữ liệu đã xử lý vào MLflow 📋"):
        # Kiểm tra và kết thúc run hiện tại nếu có
        # Sửa đổi bởi Grok 3: Thêm log dữ liệu đã xử lý vào MLflow
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"MNIST_Processed_{experiment_name}", nested=True):
            X_full = st.session_state.get('mnist_data', {}).get('X_full', None)
            if X_full is None:
                artifacts_dir = mlflow.artifacts.download_artifacts(run_id=latest_run_id, artifact_path="")
                X_full_path = os.path.join(artifacts_dir, "data_full.npy")
                y_full_path = os.path.join(artifacts_dir, "labels_full.npy")
                X_full = np.load(X_full_path)
                y_full = np.load(y_full_path)
            else:
                y_full = st.session_state['mnist_data']['y_full']

            mlflow.log_param("final_data_shape", X_full.shape)
            mlflow.log_param("final_samples", len(X_full))
            
            # Lưu 5 mẫu từ dữ liệu đầy đủ dưới dạng file .npy và log bằng mlflow.log_artifact
            # Sửa đổi bởi Grok 3: Hiển thị 5 ảnh đầu tiên
            for i in range(5):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
                    np.save(tmp.name, X_full[i])
                    mlflow.log_artifact(tmp.name, f"final_sample_{i}.npy")
                os.unlink(tmp.name)

                # Log nhãn tương ứng
                mlflow.log_param(f"final_label_{i}", y_full[i])

            # Log các bước tiền xử lý
            mlflow.log_params(st.session_state.get('preprocessing_steps', {}))
            st.success("Dữ liệu đã được tiền xử lý và log vào MLflow thành công! ✅")

            st.subheader("Xem trước 5 ảnh đầu tiên đã xử lý trong MLflow 🔚")
            for i in range(5):
                final_sample_path = os.path.join(artifacts_dir, f"final_sample_{i}.npy")
                if os.path.isfile(final_sample_path):
                    final_image = np.load(final_sample_path).reshape(28, 28)
                    st.image(final_image, caption=f"Chữ số: {y_full[i]}", width=100)
                else:
                    st.error(f"Không tìm thấy file mẫu {final_sample_path}. Vui lòng kiểm tra MLflow.")

if __name__ == "__main__":
    preprocess_mnist()