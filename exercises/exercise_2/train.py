import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
import mlflow.sklearn
import os
import dagshub
import datetime
import pickle
import logging

# Tắt cảnh báo từ dagshub
logging.getLogger("dagshub.auth.tokens").setLevel(logging.ERROR)

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

# Hàm chuẩn hóa dữ liệu với cache
@st.cache_data
def scale_data(X_train, X_valid):
    """Chuẩn hóa dữ liệu bằng StandardScaler và lưu vào bộ nhớ đệm."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    return X_train_scaled, X_valid_scaled, scaler

def train_mnist():
    st.header("Huấn luyện Mô hình Nhận diện Chữ số MNIST 🧮")

    # Khởi tạo DagsHub/MLflow chỉ một lần
    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_REPO = mlflow_input()
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_REPO
    else:
        DAGSHUB_REPO = st.session_state['mlflow_url']

    # Đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment", value="MNIST_Training", key="exp_name")
    with st.spinner("Đang thiết lập Experiment..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                new_exp_name = f"{experiment_name}_Restored_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
                st.warning(f"Experiment '{experiment_name}' đã bị xóa. Sử dụng '{new_exp_name}' thay thế.")
                experiment_name = new_exp_name
            if not client.get_experiment_by_name(experiment_name):
                client.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
        except Exception as e:
            st.error(f"Lỗi thiết lập experiment: {str(e)}")
            return

    # Kiểm tra dữ liệu MNIST
    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.error("Dữ liệu MNIST chưa được xử lý. Vui lòng chạy 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    mnist_data = st.session_state['mnist_data']
    if 'X_train' not in mnist_data or 'y_train' not in mnist_data:
        st.error("Dữ liệu huấn luyện không đầy đủ. Vui lòng chạy lại tiền xử lý.")
        return

    # Hiển thị thông tin dữ liệu ngắn gọn
    st.subheader("Dữ liệu Đã Xử lý 📝")
    col1, col2, col3 = st.columns(3)
    col1.write(f"Train: {len(mnist_data['X_train'])} mẫu")
    col2.write(f"Valid: {len(mnist_data.get('X_valid', []))} mẫu")
    col3.write(f"Test: {len(mnist_data['X_test'])} mẫu")

    # Reshape dữ liệu
    X_train = mnist_data['X_train'].reshape(-1, 28 * 28)
    y_train = mnist_data['y_train']
    X_valid = mnist_data.get('X_valid', mnist_data['X_test']).reshape(-1, 28 * 28)
    y_valid = mnist_data.get('y_valid', mnist_data['y_test'])

    # Chuẩn hóa dữ liệu
    X_train_scaled, X_valid_scaled, scaler = scale_data(X_train, X_valid)

    st.subheader("Huấn luyện Mô hình 🎯")

    # Chọn mô hình
    model_choice = st.selectbox("Chọn mô hình", ["SVM", "Decision Tree"], key="model_choice")

    # Tham số tối ưu hơn
    if model_choice == "SVM":
        kernel = st.selectbox("Kernel SVM", ["linear", "rbf", "poly"], index=1, key="svm_kernel")
        C = st.slider("Tham số C (Regularization)", 0.1, 10.0, 1.0, step=0.1, key="svm_C")
        model_params = {"kernel": kernel, "C": C}
    else:
        max_depth = st.slider("Độ sâu tối đa", 3, 20, 10, step=1, key="dt_depth")
        min_samples_split = st.slider("Số mẫu tối thiểu để split", 2, 10, 2, step=1, key="dt_min_samples")
        model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}

    # Tên run
    run_name = st.text_input("Tên Run ID (để trống để tự động tạo)", value="", max_chars=50, key="run_name")
    if not run_name.strip():
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"MNIST_{model_choice}_{timestamp}"

    # Container để hiển thị kết quả, tránh reload
    result_container = st.container()

    if st.button("Huấn luyện", key="train_button"):
        with st.spinner("Đang huấn luyện..."):
            # Khởi tạo mô hình với tham số tối ưu
            if model_choice == "SVM":
                model = SVC(kernel=model_params["kernel"], C=model_params["C"], random_state=42, probability=True, max_iter=1000)
            else:
                model = DecisionTreeClassifier(max_depth=model_params["max_depth"], min_samples_split=model_params["min_samples_split"], random_state=42)

            # Huấn luyện
            model.fit(X_train_scaled, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
            valid_acc = accuracy_score(y_valid, model.predict(X_valid_scaled))

            # Log vào MLflow
            with mlflow.start_run(run_name=run_name) as run:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("model_type", model_choice)
                mlflow.log_params(model_params)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("valid_accuracy", valid_acc)

                # Log mô hình và scaler với input_example để tránh cảnh báo MLflow
                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=X_train_scaled[:1]
                )
                scaler_file = "scaler.pkl"
                with open(scaler_file, "wb") as f:
                    pickle.dump(scaler, f)
                mlflow.log_artifact(scaler_file, artifact_path="scaler")
                os.remove(scaler_file)

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']

            # Hiển thị kết quả trong container
            with result_container:
                st.write("### Kết quả Huấn luyện")
                st.write(f"- **Mô hình**: {model_choice}")
                st.write(f"- **Tham số**: {model_params}")
                st.write(f"- **Độ chính xác**:")
                st.write(f"  - Train: {train_acc:.4f}")
                st.write(f"  - Valid: {valid_acc:.4f}")
                st.write(f"- **Run ID**: {run_id}")
                st.write(f"- **Thời gian**: {timestamp}")
                st.success("Huấn luyện và log vào MLflow hoàn tất!")
                st.markdown(f"[Xem chi tiết trên MLflow]({mlflow_uri})", unsafe_allow_html=True)

            # Lưu vào session_state chỉ sau khi hoàn tất, tránh reload giữa chừng
            st.session_state['mnist_model'] = model
            st.session_state['mnist_scaler'] = scaler
            st.session_state['training_metrics'] = {"train_accuracy": train_acc, "valid_accuracy": valid_acc}
            st.session_state['run_id'] = run_id

if __name__ == "__main__":
    train_mnist()