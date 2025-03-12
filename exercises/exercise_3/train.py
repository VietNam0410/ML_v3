import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import mlflow
import mlflow.sklearn
import os
import datetime
import time

@st.cache_resource
def get_scaler():
    return StandardScaler()

@st.cache_resource
def get_model(model_choice, **params):
    if model_choice == 'K-means':
        return KMeans(**params)
    else:
        return DBSCAN(**params)

def mlflow_input():
    DAGSHUB_MLFLOW_URI = 'https://dagshub.com/VietNam0410/ML_v3.mlflow'
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'VietNam0410'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b'
    return 'ML_v3'

def train_clustering(X_full, y_full):
    st.header('Huấn luyện Clustering trên MNIST 🧮')
    if mlflow.active_run():
        mlflow.end_run()

    DAGSHUB_REPO = mlflow_input()
    if DAGSHUB_REPO is None:
        st.error('Lỗi kết nối MLflow.')
        return

    mlflow.set_experiment('MNIST_Train_Clustering')

    max_samples = st.slider(
        'Số mẫu tối đa',
        1000, 70000, min(10000, len(X_full)),
        step=1000, key='max_samples_ex3',
        help='Số lượng mẫu dữ liệu sẽ được sử dụng để huấn luyện. Giá trị lớn hơn sẽ cải thiện độ chính xác nhưng làm chậm quá trình huấn luyện.'
    )
    total_samples = len(X_full)
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X_full = X_full[indices]
        y_full = y_full[indices]
    if max_samples > 50000:
        st.warning('Số mẫu lớn (>50,000) có thể làm chậm huấn luyện.')

    test_size = st.slider(
        'Tỷ lệ kiểm tra (%)',
        10, 50, 20,
        step=5, key='test_size_ex3',
        help='Phần trăm dữ liệu được sử dụng cho tập kiểm tra (test set). Phần còn lại sẽ được chia thành tập huấn luyện và tập validation.'
    ) / 100
    train_size_relative = st.slider(
        'Tỷ lệ huấn luyện (%)',
        50, 90, 70,
        step=5, key='train_size_ex3',
        help='Phần trăm dữ liệu (trong phần không thuộc tập kiểm tra) được dùng để huấn luyện. Phần còn lại sẽ là tập validation.'
    ) / 100
    val_size_relative = 1 - train_size_relative

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_remaining, y_remaining, train_size=train_size_relative, random_state=42, stratify=y_remaining
    )

    st.subheader("Kích thước các tập dữ liệu sau khi chia:")
    st.write(f"- **Tổng số mẫu**: {max_samples}")
    st.write(f"- **Tập kiểm tra (Test set)**: {len(X_test)} mẫu ({test_size * 100:.1f}%)")
    st.write(f"- **Tập huấn luyện (Train set)**: {len(X_train)} mẫu ({(1 - test_size) * train_size_relative * 100:.1f}%)")
    st.write(f"- **Tập validation (Valid set)**: {len(X_valid)} mẫu ({(1 - test_size) * val_size_relative * 100:.1f}%)")

    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)

    model_choice = st.selectbox(
        'Chọn thuật toán',
        ['K-means', 'DBSCAN'],
        key='model_choice_ex3',
        help='Chọn thuật toán phân cụm: K-means (phân cụm cố định số cụm) hoặc DBSCAN (phân cụm dựa trên mật độ).'
    )
    if model_choice == 'K-means':
        n_clusters = st.slider(
            'Số cụm (K)',
            2, 20, 10,
            key='n_clusters_ex3',
            help='Số lượng cụm mà K-means sẽ tạo ra. Giá trị lớn hơn có thể dẫn đến kết quả chi tiết hơn nhưng phức tạp hơn.'
        )
        model_params = {'n_clusters': n_clusters, 'random_state': 42}
    else:
        eps = st.slider(
            'Khoảng cách tối đa (eps)',
            0.1, 2.0, 0.5,
            step=0.1, key='eps_ex3',
            help='Khoảng cách tối đa giữa hai điểm để chúng được coi là trong cùng một cụm. Giá trị nhỏ hơn sẽ tạo nhiều cụm nhỏ hơn.'
        )
        min_samples = st.slider(
            'Số điểm tối thiểu',
            2, 20, 5,
            key='min_samples_ex3',
            help='Số lượng điểm tối thiểu cần thiết để tạo thành một cụm. Giá trị lớn hơn sẽ làm giảm số lượng cụm và tăng số điểm bị coi là nhiễu.'
        )
        model_params = {'eps': eps, 'min_samples': min_samples}

    run_name = st.text_input(
        'Nhập tên Run ID',
        value='', key='run_name_ex3',
        help='Tên để nhận diện lần huấn luyện này trên MLflow. Nếu để trống, hệ thống sẽ tự tạo tên dựa trên thời gian.'
    )
    if not run_name.strip():
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_name = f'MNIST_{model_choice}_{timestamp}'

    if st.button('Huấn luyện', key='train_button_ex3'):
        progress = st.progress(0)
        status_text = st.empty()

        start_time = datetime.datetime.now()

        status_text.text("Chuẩn hóa dữ liệu... 30%")
        progress.progress(0.3)
        time.sleep(0.1)
        scaler = get_scaler()
        X_train_scaled = scaler.fit_transform(X_train_flat)
        X_valid_scaled = scaler.transform(X_valid_flat)

        status_text.text("Huấn luyện mô hình... 80%")
        progress.progress(0.8)
        time.sleep(0.1)
        model = get_model(model_choice, **model_params)
        model.fit(X_train_scaled)
        labels_train = model.predict(X_train_scaled) if model_choice == 'K-means' else model.fit_predict(X_train_scaled)
        labels_valid = model.predict(X_valid_scaled) if model_choice == 'K-means' else model.fit_predict(X_valid_scaled)
        n_clusters_found = len(np.unique(labels_valid)) - (1 if -1 in labels_valid else 0)

        status_text.text("Tính toán Silhouette Score... 100%")
        progress.progress(1.0)
        time.sleep(0.1)
        silhouette_train = silhouette_score(X_train_scaled, labels_train) if n_clusters_found > 1 else -1
        silhouette_valid = silhouette_score(X_valid_scaled, labels_valid) if n_clusters_found > 1 else -1

        training_duration = (datetime.datetime.now() - start_time).total_seconds()
        log_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        with mlflow.start_run(run_name=run_name):
            mlflow.log_param('model_type', model_choice)
            mlflow.log_params(model_params)
            mlflow.log_param('log_time', log_time)
            mlflow.log_metric('silhouette_train', silhouette_train)
            mlflow.log_metric('silhouette_valid', silhouette_valid)
            mlflow.log_metric('training_duration', training_duration)
            mlflow.log_metric('n_clusters_found', n_clusters_found)  # Log số cụm thực tế
            mlflow.sklearn.log_model(model, 'model', input_example=X_train_scaled[:1])
            run_id = mlflow.active_run().info.run_id

        status_text.text("Huấn luyện hoàn tất!")
        st.write(f'**Thuật toán**: {model_choice}')
        st.write(f'Silhouette Score: Train: {silhouette_train:.4f}, Valid: {silhouette_valid:.4f}')
        st.write(f'Thời gian huấn luyện: {training_duration:.2f} giây')
        st.write(f'Thời gian log: {log_time}')
        st.markdown(
            "**Ý nghĩa Silhouette Score**: \n"
            "- Giá trị dao động từ **-1 đến 1**. \n"
            "- **> 0.5**: Cụm tốt, các điểm được phân loại rõ ràng. \n"
            "- **0.25 - 0.5**: Cụm chấp nhận được, nhưng có thể cải thiện. \n"
            "- **< 0**: Cụm không hợp lệ, các điểm có thể thuộc cụm khác. \n"
            "- Giá trị **-1**: Điểm dữ liệu rất xa với cụm được gán."
        )
        st.success(f'Huấn luyện hoàn tất! Run ID: {run_id}')