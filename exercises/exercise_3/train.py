import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

def train_clustering():
    st.header("Huấn luyện Mô hình Clustering trên MNIST 🧮")

    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    experiment_name = st.text_input("Nhập Tên Experiment cho Huấn luyện", value="MNIST_Clustering")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

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

    if st.button("Huấn luyện và hiển thị kết quả"):
        with mlflow.start_run(run_name=f"{model_choice}_MNIST_Clustering") as run:
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

            # Log vào MLflow
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", model_choice)
            mlflow.log_metric("n_clusters_found", n_clusters_found)
            mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
            mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])
            mlflow.sklearn.log_model(pca, "pca", input_example=X_train_scaled[:1])

            # Lưu biểu đồ
            plt.savefig("clustering_plot.png")
            mlflow.log_artifact("clustering_plot.png", artifact_path="plots")
            os.remove("clustering_plot.png")

            run_id = run.info.run_id
            dagshub_link = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments/#/experiment/{experiment_name}/{run_id}"
            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow ✅ (Run ID: {run_id})")
            st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")

            st.session_state['clustering_model'] = model
            st.session_state['clustering_scaler'] = scaler
            st.session_state['clustering_pca'] = pca
            st.session_state['clustering_labels'] = labels

if __name__ == "__main__":
    train_clustering()