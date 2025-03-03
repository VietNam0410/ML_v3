import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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
@st.cache_data
def train_clustering():
    st.header("Huấn luyện Mô hình Clustering trên MNIST 🧮")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow tại DAGSHUB_MLFLOW_URI
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment cho Huấn luyện", value="MNIST_Clustering")
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

    # Kiểm tra dữ liệu từ preprocess_mnist_clustering.py
    if 'mnist_clustering_data' not in st.session_state or st.session_state['mnist_clustering_data'] is None:
        st.error("Dữ liệu MNIST cho clustering không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Clustering Preprocess' trước.")
        return

    mnist_data = st.session_state['mnist_clustering_data']
    if 'X_train' not in mnist_data:
        st.error("Dữ liệu 'X_train' không tồn tại trong session. Vui lòng hoàn tất tiền xử lý trước.")
        return

    st.subheader("Dữ liệu MNIST đã xử lý 📝")
    st.write(f"Số lượng mẫu huấn luyện: {len(mnist_data['X_train'])}")
    st.write(f"Số lượng mẫu validation: {len(mnist_data.get('X_valid', []))}")
    st.write(f"Số lượng mẫu kiểm tra: {len(mnist_data['X_test'])}")

    X_train = mnist_data['X_train'].reshape(-1, 28 * 28)
    X_valid = mnist_data.get('X_valid', mnist_data['X_test']).reshape(-1, 28 * 28)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Giảm chiều dữ liệu để hiển thị biểu đồ
    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_valid_pca = pca.transform(X_valid_scaled)

    st.subheader("Giới thiệu thuật toán Clustering")
    st.write("### K-means")
    st.write("K-means là một thuật toán phân cụm không giám sát, chia dữ liệu thành K cụm sao cho tổng bình phương khoảng cách từ mỗi điểm đến tâm cụm gần nhất là nhỏ nhất.")
    st.write("### DBSCAN")
    st.write("DBSCAN (Density-Based Spatial Clustering of Applications with Noise) phân cụm dựa trên mật độ điểm, không yêu cầu xác định trước số cụm, và có thể phát hiện nhiễu.")

    st.subheader("Huấn luyện Mô hình Clustering 🎯")
    model_choice = st.selectbox("Chọn thuật toán clustering", ["K-means", "DBSCAN"])

    if model_choice == "K-means":
        n_clusters = st.slider("Số lượng cụm (K)", 2, 20, 10, step=1)
        max_iter = st.slider("Số lần lặp tối đa", 100, 1000, 300, step=100)
        model_params = {"n_clusters": n_clusters, "max_iter": max_iter, "random_state": 42}
    else:  # DBSCAN
        eps = st.slider("Khoảng cách tối đa (eps)", 0.1, 2.0, 0.5, step=0.1)
        min_samples = st.slider("Số mẫu tối thiểu trong một cụm", 2, 20, 5, step=1)
        model_params = {"eps": eps, "min_samples": min_samples}

    # Cho phép người dùng đặt tên run ID cho mô hình
    run_name = st.text_input("Nhập tên Run ID cho mô hình clustering (để trống để tự động tạo)", value="", max_chars=20, key="clustering_run_name_input")
    if run_name.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_{model_choice}_Clustering_{timestamp.replace(' ', '_').replace(':', '-')}"  # Định dạng tên run hợp lệ cho MLflow

    if st.button("Huấn luyện và hiển thị kết quả"):
        # Đóng bất kỳ run nào đang hoạt động trước khi bắt đầu
        if mlflow.active_run():
            mlflow.end_run()

        with st.spinner("Đang huấn luyện mô hình clustering..."):
            if model_choice == "K-means":
                model = KMeans(**model_params)
            else:
                model = DBSCAN(**model_params)

            model.fit(X_train_scaled)
            labels = model.predict(X_valid_scaled) if model_choice == "K-means" else model.fit_predict(X_valid_scaled)
            n_clusters_found = len(np.unique(labels)) - (1 if -1 in labels else 0)  # Đếm số cụm, trừ nhiễu nếu có

            st.write(f"Thuật toán: {model_choice}")
            st.write(f"Tham số: {model_params}")
            st.write(f"Số cụm tìm thấy: {n_clusters_found}")

            # Vẽ biểu đồ
            fig, ax = plt.subplots()
            scatter = ax.scatter(X_valid_pca[:, 0], X_valid_pca[:, 1], c=labels, cmap="viridis", s=10)
            plt.colorbar(scatter)
            plt.title(f"{model_choice} Clustering trên MNIST (PCA 2D)")
            plt.xlabel("PCA Component 1")
            plt.ylabel("PCA Component 2")
            st.pyplot(fig)

            # Lưu biểu đồ cục bộ trước khi log vào MLflow
            plot_file = "clustering_plot.png"
            fig.savefig(plot_file)

            # Logging vào MLflow tại DAGSHUB_MLFLOW_URI
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("run_id", run.info.run_id)
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                mlflow.log_metric("n_clusters_found", n_clusters_found)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
                mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])
                mlflow.sklearn.log_model(pca, "pca", input_example=X_train_scaled[:1])

                # Log biểu đồ làm artifact
                mlflow.log_artifact(plot_file, artifact_path="plots")
                os.remove(plot_file)  # Xóa file cục bộ sau khi log

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']
                st.success(f"Huấn luyện {model_choice} hoàn tất và log vào DagsHub MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {timestamp})")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

            # Lưu mô hình, scaler, PCA, và nhãn vào session_state để sử dụng sau
            st.session_state['clustering_model'] = model
            st.session_state['clustering_scaler'] = scaler
            st.session_state['clustering_pca'] = pca
            st.session_state['clustering_labels'] = labels

if __name__ == "__main__":
    train_clustering()