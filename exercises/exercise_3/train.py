import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
import mlflow
import os
import dagshub
import datetime
from tensorflow.keras.datasets import mnist
import openml

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if experiments:
            st.success("Kết nối MLflow với DagsHub thành công! ✅")
        else:
            st.warning("Không tìm thấy experiment nào, nhưng kết nối MLflow vẫn hoạt động.")
        return "ML_v3"
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Lỗi xác thực MLflow: {str(e)}. Vui lòng kiểm tra token tại https://dagshub.com/user/settings/tokens.")
        return None

# Hàm tải dữ liệu MNIST
@st.cache_data
def load_mnist():
    with st.spinner("Đang tải dữ liệu MNIST..."):
        try:
            dataset = openml.datasets.get_dataset(554)
            X, y, _, _ = dataset.get_data(target='class')
            X = X.values.reshape(-1, 28 * 28) / 255.0
            y = y.astype(np.int32)
        except Exception as e:
            st.error(f"Không thể tải từ OpenML: {str(e)}. Sử dụng TensorFlow.")
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X = np.concatenate([X_train, X_test], axis=0) / 255.0
            y = np.concatenate([y_train, y_test], axis=0)
            X = X.reshape(-1, 28 * 28)
        return X, y

# Cache scaler và mô hình để tái sử dụng
@st.cache_resource
def get_scaler():
    return StandardScaler()

@st.cache_resource
def get_model(model_choice, **params):
    if model_choice == "K-means":
        return KMeans(**params)
    else:
        return DBSCAN(**params)

def train_clustering():
    st.header("Huấn luyện Mô hình Clustering trên MNIST 🧮")

    # Đóng run MLflow nếu đang hoạt động
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow trước đó.")

    # Thiết lập MLflow
    DAGSHUB_REPO = mlflow_input()
    if DAGSHUB_REPO is None:
        st.error("Không thể tiếp tục do lỗi kết nối MLflow.")
        return

    experiment_name = "MNIST_Train_Clustering"
    with st.spinner("Đang thiết lập Experiment..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                st.warning(f"Experiment '{experiment_name}' đã bị xóa. Vui lòng khôi phục qua DagsHub UI.")
                return
            else:
                mlflow.set_experiment(experiment_name)
        except mlflow.exceptions.MlflowException as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}.")
            return

    # Tải dữ liệu
    X_full, y_full = load_mnist()
    total_samples = len(X_full)

    st.subheader("Chia tập dữ liệu MNIST 🔀")
    max_samples = st.slider("Số mẫu tối đa (0 = toàn bộ, tối đa 70.000)", 0, 70000, 70000, step=100)  # Giữ nguyên mặc định 70,000
    
    if max_samples == 0 or max_samples > total_samples:
        st.warning(f"Số mẫu {max_samples} vượt quá {total_samples}. Dùng toàn bộ.")
        max_samples = total_samples
    elif max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X_full = X_full[indices]
        y_full = y_full[indices]

    test_size = st.slider("Tỷ lệ tập kiểm tra (%)", 10, 50, 20, step=5) / 100
    train_size_relative = st.slider("Tỷ lệ tập huấn luyện (%)", 50, 90, 70, step=5) / 100
    val_size_relative = 1 - train_size_relative

    X_remaining, X_test, y_remaining, y_test = train_test_split(
        X_full, y_full, test_size=test_size, random_state=42, stratify=y_full
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_remaining, y_remaining, train_size=train_size_relative, random_state=42, stratify=y_remaining
    )

    st.subheader("Thông tin tập dữ liệu 📊")
    st.write(f"Tổng số mẫu: {max_samples}")
    st.write(f"Tỷ lệ: Huấn luyện {train_size_relative*100:.1f}%, Validation {val_size_relative*100:.1f}%, Kiểm tra {test_size*100:.1f}%")
    st.write(f"Tập huấn luyện: {len(X_train)} mẫu, Validation: {len(X_valid)} mẫu, Kiểm tra: {len(X_test)} mẫu")

    # Chuẩn hóa dữ liệu
    scaler = get_scaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    st.subheader("Giới thiệu thuật toán Clustering")
    st.write("### K-means")
    st.write("Phân cụm dữ liệu thành K cụm dựa trên khoảng cách tới tâm cụm.")
    st.write("### DBSCAN")
    st.write("Phân cụm dựa trên mật độ, tự động phát hiện nhiễu.")

    st.subheader("Huấn luyện Mô hình 🎯")
    model_choice = st.selectbox("Chọn thuật toán", ["K-means", "DBSCAN"])

    if model_choice == "K-means":
        n_clusters = st.slider("Số cụm (K)", 2, 20, 10, step=1)
        model_params = {"n_clusters": n_clusters, "random_state": 42}  # Không tối ưu n_init, max_iter
    else:
        eps = st.slider("Khoảng cách tối đa (eps)", 0.1, 2.0, 0.5, step=0.1)
        min_samples = st.slider("Số điểm tối thiểu", 2, 20, 5, step=1)
        model_params = {"eps": eps, "min_samples": min_samples}  # Không dùng n_jobs

    run_name = st.text_input("Nhập tên Run ID (để trống để tự tạo)", value="", max_chars=20)
    if not run_name.strip():
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"MNIST_{model_choice}_{timestamp}"

    if st.button("Huấn luyện và đánh giá"):
        if mlflow.active_run():
            mlflow.end_run()

        with st.spinner(f"Đang huấn luyện {model_choice}..."):
            model = get_model(model_choice, **model_params)
            model.fit(X_train_scaled)

            # Dự đoán nhãn
            labels_train = model.predict(X_train_scaled) if model_choice == "K-means" else model.fit_predict(X_train_scaled)
            labels_valid = model.predict(X_valid_scaled) if model_choice == "K-means" else model.fit_predict(X_valid_scaled)
            labels_test = model.predict(X_test_scaled) if model_choice == "K-means" else model.fit_predict(X_test_scaled)
            n_clusters_found = len(np.unique(labels_valid)) - (1 if -1 in labels_valid else 0)

            # Tính Silhouette Score
            silhouette_train = silhouette_score(X_train_scaled, labels_train) if n_clusters_found > 1 else -1
            silhouette_valid = silhouette_score(X_valid_scaled, labels_valid) if n_clusters_found > 1 else -1

            # Hiển thị thông tin
            st.write(f"**Thuật toán**: {model_choice}")
            st.write(f"**Tham số**: {model_params}")
            st.write("**Chỉ số đánh giá**:")
            st.write(f"- Silhouette Score (Train): {silhouette_train:.4f}")
            st.write(f"- Silhouette Score (Valid): {silhouette_valid:.4f}")
            st.write("""
                **Thông tin về Silhouette Score**:  
                - Là chỉ số đánh giá chất lượng phân cụm, đo lường mức độ tương đồng của một điểm trong cụm của nó so với các cụm khác.  
                - Giá trị từ -1 đến 1:  
                  + Gần 1: Các cụm được phân tách tốt, điểm nằm gần cụm của nó.  
                  + Gần 0: Các cụm chồng lấp nhau.  
                  + Gần -1: Điểm có thể bị phân cụm sai.  
                - Chỉ tính khi số cụm > 1.
            """)

            # Logging vào MLflow
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("run_id", run.info.run_id)
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                mlflow.log_param("max_samples", max_samples)
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("train_size", train_size_relative)
                mlflow.log_param("val_size", val_size_relative)
                mlflow.log_metric("n_clusters_found", n_clusters_found)
                mlflow.log_metric("silhouette_train", silhouette_train)
                mlflow.log_metric("silhouette_valid", silhouette_valid)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
                mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']
                st.success(f"Huấn luyện {model_choice} hoàn tất! ✅ (Run: {run_name}, ID: {run_id}, Thời gian: {timestamp})")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow]({mlflow_uri})")

            # Lưu vào session_state
            st.session_state['clustering_model'] = model
            st.session_state['clustering_scaler'] = scaler
            st.session_state['clustering_labels_train'] = labels_train
            st.session_state['clustering_labels_valid'] = labels_valid
            st.session_state['clustering_labels_test'] = labels_test

if __name__ == "__main__":
    train_clustering()