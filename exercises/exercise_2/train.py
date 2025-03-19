import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
import os
import datetime
import time
import logging

# # Tắt log không cần thiết từ TensorFlow và MLflow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log TensorFlow
# logging.getLogger("mlflow").setLevel(logging.WARNING)  # Giảm log MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = 'https://dagshub.com/VietNam0410/ML_v3.mlflow'
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'VietNam0410'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b'
    os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Chạy trên CPU để tránh lỗi CUDA
    return 'ML_v3'

@st.cache_resource
def get_scaler():
    return StandardScaler()

@st.cache_resource
def get_model(model_choice, **params):
    if model_choice == 'SVM':
        return SVC(**params)
    else:
        return DecisionTreeClassifier(**params)


def train_mnist(X_full, y_full):
    st.header('Huấn luyện Mô hình Nhận diện trên MNIST 🧮')
    if mlflow.active_run():
        mlflow.end_run()

    DAGSHUB_REPO = mlflow_input()
    if DAGSHUB_REPO is None:
        st.error('Lỗi kết nối MLflow.')
        return

    mlflow.set_experiment('MNIST_Training')

    total_samples = len(X_full)
    st.subheader('Chia tập dữ liệu MNIST 🔀')
    max_samples = st.slider(
        'Số mẫu tối đa',
        1000, 70000, min(10000, total_samples),
        step=1000, key='max_samples_ex2',
        help='Số lượng mẫu dữ liệu sẽ được sử dụng để huấn luyện. Giá trị lớn hơn sẽ cải thiện độ chính xác nhưng làm chậm quá trình huấn luyện.'
    )
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X_full = X_full[indices]
        y_full = y_full[indices]
    if max_samples > 50000:
        st.warning('Số mẫu lớn (>50,000) có thể làm chậm huấn luyện.')

    test_size = st.slider(
        'Tỷ lệ tập kiểm tra (%)',
        10, 50, 20, step=5, key='test_size_ex2',
        help='Phần trăm dữ liệu được sử dụng cho tập kiểm tra (test set). Phần còn lại sẽ được chia thành tập huấn luyện và tập validation.'
    ) / 100
    train_size_relative = st.slider(
        'Tỷ lệ tập huấn luyện (%)',
        50, 90, 70, step=5, key='train_size_ex2',
        help='Phần trăm dữ liệu (trong phần không thuộc tập kiểm tra) được dùng để huấn luyện. Phần còn lại sẽ là tập validation.'
    ) / 100
    val_size_relative = 1 - train_size_relative

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_remaining, y_remaining, train_size=train_size_relative, random_state=42, stratify=y_remaining
    )

    st.write(f'Tổng số mẫu: {max_samples}')
    st.write(f'Tỷ lệ: Huấn luyện {train_size_relative*100:.1f}%, Validation {val_size_relative*100:.1f}%, Kiểm tra {test_size*100:.1f}%')
    st.write(f'Tập huấn luyện: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu, Kiểm tra: {len(X_test)} mẫu')

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_valid_scaled = scaler.transform(X_valid_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    st.subheader('Huấn luyện Mô hình 🎯')
    model_choice = st.selectbox(
        'Chọn thuật toán',
        ['SVM', 'Decision Tree'],
        key='model_choice_ex2',
        help='Chọn thuật toán phân loại: SVM (phân loại dựa trên ranh giới tối ưu) hoặc Decision Tree (phân loại dựa trên cây quyết định).'
    )
    model_params = {}
    if model_choice == 'SVM':
        kernel = st.selectbox(
            'Kernel',
            ['linear', 'rbf', 'poly'],
            index=0, key='svm_kernel_ex2',
            help='Loại kernel cho SVM: "linear" (tuyến tính, nhanh nhất), "rbf" (phi tuyến tính, chính xác hơn nhưng chậm hơn), "poly" (đa thức, chậm và phức tạp).'
        )
        model_params = {'kernel': kernel, 'probability': True, 'random_state': 42}
    else:
        max_depth = st.slider(
            'Độ sâu tối đa',
            3, 20, 10, step=1, key='dt_max_depth_ex2',
            help='Số lớp tối đa của cây quyết định. Giá trị lớn hơn làm tăng độ chính xác nhưng có thể dẫn đến overfitting và chậm hơn.'
        )
        model_params = {
            'max_depth': max_depth,
            'random_state': 42
        }

    run_name = st.text_input(
        'Nhập tên Run ID',
        value='', key='run_name_ex2',
        help='Tên để nhận diện lần huấn luyện này trên MLflow. Nếu để trống, hệ thống sẽ tự tạo tên dựa trên thời gian.'
    )
    if not run_name.strip():
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'MNIST_{model_choice}_{timestamp}'

    if st.button('Huấn luyện', key='train_button_ex2'):
        # Khởi tạo thanh tiến trình
        progress = st.progress(0)
        status_text = st.empty()

        # Bắt đầu đo thời gian
        start_time = datetime.datetime.now()

        # Bước 1: Chuẩn hóa dữ liệu (30% tiến trình)
        status_text.text("Chuẩn hóa dữ liệu... 30%")
        progress.progress(0.3)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        scaler = get_scaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_valid_scaled = scaler.transform(X_valid_flat)
        X_test_scaled = scaler.transform(X_test_flat)

        # Bước 2: Huấn luyện mô hình (80% tiến trình)
        status_text.text("Huấn luyện mô hình... 80%")
        progress.progress(0.8)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        model = get_model(model_choice, **model_params)
        model.fit(X_train_scaled, y_train)

        # Bước 3: Đánh giá và hoàn tất (100% tiến trình)
        status_text.text("Đánh giá mô hình... 100%")
        progress.progress(1.0)
        time.sleep(0.1)  # Giả lập thời gian xử lý
        train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
        valid_acc = accuracy_score(y_valid, model.predict(X_valid_scaled))
        test_acc = accuracy_score(y_test, model.predict(X_test_scaled))

        # Tính thời gian huấn luyện
        training_duration = (datetime.datetime.now() - start_time).total_seconds()
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Log kết quả vào MLflow
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('model_type', model_choice)
            mlflow.log_params(model_params)
            mlflow.log_param('log_time', log_time)
            mlflow.log_metric('train_accuracy', train_acc)
            mlflow.log_metric('valid_accuracy', valid_acc)
            mlflow.log_metric('test_accuracy', test_acc)
            mlflow.log_metric('training_duration', training_duration)
            mlflow.sklearn.log_model(model, 'model', input_example=X_train_scaled[:1])  # Đảm bảo input_example hợp lệ
            run_id = mlflow.active_run().info.run_id

        # Hiển thị kết quả
        status_text.text("Huấn luyện hoàn tất!")
        st.write(f'**Mô hình**: {model_choice}')
        st.write(f'**Độ chính xác**: Train: {train_acc:.4f}, Valid: {valid_acc:.4f}, Test: {test_acc:.4f}')
        st.write(f'Thời gian huấn luyện: {training_duration:.2f} giây')
        st.write(f'Thời gian log: {log_time}')
        st.success(f'Huấn luyện hoàn tất! Run ID: {run_id}')

if __name__ == "__main__":
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    train_mnist(X, y)