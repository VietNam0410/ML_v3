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
import pickle
from tensorflow.keras.datasets import mnist

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = 'https://dagshub.com/VietNam0410/ML_v3.mlflow'
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    
    os.environ['MLFLOW_TRACKING_USERNAME'] = 'VietNam0410'
    os.environ['MLFLOW_TRACKING_PASSWORD'] = 'c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b'
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if experiments:
            st.success('Kết nối MLflow với DagsHub thành công! ✅')
        else:
            st.warning('Không tìm thấy experiment nào, nhưng kết nối MLflow vẫn hoạt động.')
        return 'ML_v3'
    except mlflow.exceptions.MlflowException as e:
        st.error(f'Lỗi xác thực MLflow: {str(e)}. Vui lòng kiểm tra token tại https://dagshub.com/user/settings/tokens.')
        return None

# Cache scaler và mô hình để tái sử dụng
@st.cache_resource
def get_scaler():
    return StandardScaler()

@st.cache_resource
def get_model(model_choice, **params):
    if model_choice == 'SVM':
        return SVC(**params)
    else:
        return DecisionTreeClassifier(**params)

# Hàm tải dữ liệu MNIST trực tiếp từ Keras
@st.cache_data(ttl=86400, show_spinner=False)
def load_mnist_data(max_samples: int = 70000):
    """Tải dữ liệu MNIST trực tiếp từ Keras và chuẩn hóa"""
    with st.spinner('Đang tải dữ liệu MNIST từ Keras...'):
        try:
            # Load dữ liệu trực tiếp từ Keras
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            
            # Gộp train và test thành một tập dữ liệu duy nhất
            X_full = np.concatenate([X_train, X_test], axis=0)
            y_full = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
            
            # Chuẩn hóa dữ liệu: chuyển pixel từ [0, 255] về [0, 1]
            X_full = X_full.astype('float32') / 255.0
            
            # Giới hạn số mẫu nếu cần
            total_samples = len(X_full)
            if max_samples == 0 or max_samples > total_samples:
                max_samples = total_samples
            if max_samples < total_samples:
                indices = np.random.choice(total_samples, max_samples, replace=False)
                X_full = X_full[indices]
                y_full = y_full[indices]
                
            st.success('Đã tải dữ liệu MNIST thành công từ Keras! ✅')
            return X_full, y_full
        except Exception as e:
            st.error(f'Lỗi khi tải dữ liệu MNIST: {str(e)}. Vui lòng kiểm tra kết nối internet hoặc thư viện tensorflow.')
            return None, None

def train_mnist():
    st.header('Huấn luyện Mô hình Nhận diện trên MNIST 🧮')

    # Đóng run MLflow nếu đang hoạt động
    if mlflow.active_run():
        mlflow.end_run()
        st.info('Đã đóng run MLflow trước đó.')

    # Thiết lập MLflow
    DAGSHUB_REPO = mlflow_input()
    if DAGSHUB_REPO is None:
        st.error('Không thể tiếp tục do lỗi kết nối MLflow.')
        return

    # Container cho thiết lập Experiment
    setup_container = st.container()
    with setup_container:
        experiment_name = st.text_input('Nhập Tên Experiment', value='MNIST_Training', key='exp_name', disabled=True)
        if 'experiment_set' not in st.session_state:
            with st.spinner('Đang thiết lập Experiment...'):
                try:
                    client = mlflow.tracking.MlflowClient()
                    experiment = client.get_experiment_by_name('MNIST_Training')
                    if experiment and experiment.lifecycle_stage == 'deleted':
                        new_exp_name = f'MNIST_Training_Restored_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
                        st.warning(f'Experiment MNIST_Training đã bị xóa. Sử dụng {new_exp_name} thay thế.')
                        mlflow.set_experiment(new_exp_name)
                    else:
                        if not client.get_experiment_by_name('MNIST_Training'):
                            client.create_experiment('MNIST_Training')
                        mlflow.set_experiment('MNIST_Training')
                    st.session_state['experiment_set'] = True
                except mlflow.exceptions.MlflowException as e:
                    st.error(f'Lỗi khi thiết lập experiment: {str(e)}.')
                    return

    # Tải dữ liệu
    X_full, y_full = load_mnist_data(max_samples=10000)
    if X_full is None or y_full is None:
        st.error('Không thể tải dữ liệu MNIST. Ứng dụng sẽ tiếp tục chạy, nhưng không huấn luyện được.')
        return

    total_samples = len(X_full)

    # Container cho thông tin dữ liệu
    data_container = st.container()
    with data_container:
        st.subheader('Chia tập dữ liệu MNIST 🔀')
        max_samples = st.slider('Số mẫu tối đa (0 = toàn bộ, tối đa 70.000)', 0, 70000, 10000, step=100)
        if max_samples == 0 or max_samples > total_samples:
            st.warning(f'Số mẫu {max_samples} vượt quá {total_samples}. Dùng toàn bộ nhưng giới hạn tối đa 10,000 để tiết kiệm tài nguyên.')
            max_samples = min(total_samples, 10000)
        elif max_samples < total_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            X_full = X_full[indices]
            y_full = y_full[indices]

        test_size = st.slider('Tỷ lệ tập kiểm tra (%)', 10, 50, 20, step=5) / 100
        train_size_relative = st.slider('Tỷ lệ tập huấn luyện (%)', 50, 90, 70, step=5) / 100
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

    # Làm phẳng dữ liệu từ (n_samples, 28, 28) thành (n_samples, 28*28) để phù hợp với StandardScaler
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_valid_flat = X_valid.reshape(X_valid.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Chuẩn hóa dữ liệu
    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train_flat)
    X_valid_scaled = scaler.transform(X_valid_flat)
    X_test_scaled = scaler.transform(X_test_flat)

    # Container cho huấn luyện mô hình
    train_container = st.container()
    with train_container:
        st.subheader('Giới thiệu thuật toán')
        st.write('### Phân loại (Classification)')
        st.write('- **SVM (Support Vector Machine):** Một mô hình học máy mạnh mẽ, sử dụng các siêu phẳng để phân loại dữ liệu. Phù hợp với dữ liệu MNIST nhờ khả năng xử lý không gian đặc trưng cao.')
        st.write('- **Decision Tree:** Một mô hình dựa trên cây quyết định, dễ hiểu và giải thích, nhưng có thể dễ bị overfitting nếu không tối ưu độ sâu.')

        st.subheader('Huấn luyện Mô hình 🎯')
        model_choice = st.selectbox('Chọn thuật toán', ['SVM', 'Decision Tree'], key='model_choice')

        # Tham số tối ưu cho từng mô hình
        model_params = {}
        if model_choice == 'SVM':
            st.write("### Tham số cho SVM")
            kernel = st.selectbox('Kernel', ['linear', 'rbf', 'poly'], index=1, key='svm_kernel')
            model_params = {
                'kernel': kernel,
                'probability': True,  # Để tính predict_proba trong demo
                'random_state': 42
            }
        else:  # Decision Tree
            st.write("### Tham số cho Decision Tree")
            max_depth = st.slider('Độ sâu tối đa', 3, 20, 10, step=1, key='dt_max_depth')
            model_params = {
                'max_depth': max_depth,
                'random_state': 42
            }

        # Tên run
        run_name = st.text_input('Nhập tên Run ID (để trống để tự tạo)', value='', max_chars=20, key='run_name')
        if not run_name.strip():
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            run_name = f'MNIST_{model_choice}_{timestamp}'

        # Container để hiển thị kết quả
        result_container = st.container()

        if st.button('Huấn luyện', key='train_button'):
            with st.spinner('Đang huấn luyện mô hình...'):
                try:
                    model = get_model(model_choice, **model_params)

                    # Huấn luyện phân loại
                    model.fit(X_train_scaled, y_train)
                    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
                    valid_acc = accuracy_score(y_valid, model.predict(X_valid_scaled))

                    # Log vào MLflow
                    with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name('MNIST_Training').experiment_id) as run:
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        mlflow.log_param('timestamp', timestamp)
                        mlflow.log_param('model_type', model_choice)
                        mlflow.log_params(model_params)
                        mlflow.log_metric('train_accuracy', train_acc)
                        mlflow.log_metric('valid_accuracy', valid_acc)
                        mlflow.sklearn.log_model(model, 'model', input_example=X_train_scaled[:1])
                        scaler_file = 'scaler.pkl'
                        with open(scaler_file, 'wb') as f:
                            pickle.dump(scaler, f)
                        mlflow.log_artifact(scaler_file, 'scaler')
                        os.remove(scaler_file)

                        run_id = run.info.run_id
                        mlflow_uri = st.session_state['mlflow_url']

                    # Hiển thị kết quả
                    with result_container:
                        st.write('### Kết quả Huấn luyện')
                        st.write(f'- **Mô hình**: {model_choice}')
                        st.write(f'- **Tham số**: {model_params}')
                        st.write(f'- **Độ chính xác**:')
                        st.write(f'  - Train: {train_acc:.4f}')
                        st.write(f'  - Valid: {valid_acc:.4f}')
                        st.write(f'- **Experiment**: MNIST_Training')
                        st.write(f'- **Run ID**: {run_id}')
                        st.write(f'- **Thời gian**: {timestamp}')
                        st.success('Huấn luyện và log vào MLflow hoàn tất!')

                    # Lưu vào session_state
                    st.session_state['mnist_model'] = model
                    st.session_state['training_metrics'] = {'train_accuracy': train_acc, 'valid_accuracy': valid_acc}
                    st.session_state['run_id'] = run_id
                    st.session_state['training_done'] = True

                    st.markdown(f'Xem chi tiết tại: [DagsHub MLflow]({mlflow_uri})')

                except Exception as e:
                    st.error(f'Lỗi khi huấn luyện hoặc log mô hình: {str(e)}')
                    
        # Reset trạng thái khi thay đổi mô hình hoặc tham số
        if st.button('Reset tham số', key='reset_button'):
            if 'training_done' in st.session_state:
                del st.session_state['training_done']
            st.success('Đã reset trạng thái huấn luyện.')

if __name__ == '__main__':
    train_mnist()