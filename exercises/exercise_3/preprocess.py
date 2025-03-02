import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
import mlflow
import os
import dagshub

# Thiết lập thông tin DagsHub
DAGSHUB_USERNAME = "VietNam0410"
DAGSHUB_REPO = "vn0410"

try:
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    st.success("Đã kết nối với DagsHub thành công!")
except Exception as e:
    st.error(f"Không thể kết nối với DagsHub: {str(e)}. Sử dụng MLflow cục bộ.")
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")


def load_mnist_from_openml():
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

    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Clustering_Preprocessing")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

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

    test_size = st.slider("Chọn kích thước tập kiểm tra (%)", min_value=10, max_value=50, value=20, step=5) / 100
    remaining_size = 1 - test_size
    train_size_relative = st.slider(
        "Chọn kích thước tập huấn luyện (% phần còn lại sau test)",
        min_value=10, max_value=90, value=70, step=5
    ) / 100
    train_size = remaining_size * train_size_relative
    val_size = remaining_size * (1 - train_size_relative)

    st.write(f"Tỷ lệ dự kiến: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%")

    if st.button("Chia dữ liệu"):
        if max_samples < total_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            X_subset = X_full[indices]
            y_subset = y_full[indices]
        else:
            X_subset = X_full
            y_subset = y_full

        X_remaining, X_test, y_remaining, y_test = train_test_split(
            X_subset, y_subset, test_size=test_size, random_state=42
        )
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_remaining, y_remaining, test_size=val_size / remaining_size, random_state=42
        )

        st.success(f"Đã chia dữ liệu với số lượng mẫu: {max_samples}. Kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%! ✅")
        st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
        st.write(f"Tập validation: {len(X_valid)} mẫu")
        st.write(f"Tập kiểm tra: {len(X_test)} mẫu")

        with mlflow.start_run(run_name=f"MNIST_Clustering_Data_Split_{max_samples}_Samples") as run:
            mlflow.log_param("max_samples", max_samples)
            mlflow.log_param("train_size", train_size)
            mlflow.log_param("val_size", val_size)
            mlflow.log_param("test_size", test_size)
            mlflow.log_metric("train_samples", len(X_train))
            mlflow.log_metric("valid_samples", len(X_valid))
            mlflow.log_metric("test_samples", len(X_test))

            processed_file = "mnist_clustering_processed.npz"
            np.savez(processed_file, 
                     X_train=X_train, y_train=y_train,
                     X_valid=X_valid, y_valid=y_valid,
                     X_test=X_test, y_test=y_test)
            mlflow.log_artifact(processed_file, artifact_path="processed_data")
            os.remove(processed_file)

            run_id = run.info.run_id
            dagshub_link = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments/#/experiment/{experiment_name}/{run_id}"
            st.success("Dữ liệu đã được chia và log vào MLflow ✅.")
            st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")

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