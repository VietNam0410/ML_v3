import streamlit as st
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
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

def plot_and_log_reduction(X_reduced, y, method, params, run):
    """Vẽ biểu đồ scatter và log vào MLflow."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="tab10", ax=ax, s=10)
    ax.set_title(f"{method} Visualization of MNIST (Train Set)")
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    st.pyplot(fig)

    # Lưu plot vào file tạm để log vào MLflow
    plot_file = f"{method.lower()}_plot.png"
    fig.savefig(plot_file)
    if run:
        mlflow.log_artifact(plot_file, artifact_path="visualizations")
    os.remove(plot_file)  # Xóa file tạm sau khi log

    st.info(f"Biểu đồ {method} đã được log vào DagsHub MLflow.")

def train():
    st.header("Huấn luyện PCA và t-SNE trên MNIST 🧮")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow tại DAGSHUB_MLFLOW_URI
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment cho Giảm Chiều", value="MNIST_DimReduction")
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

    # Kiểm tra dữ liệu từ preprocess.py (chỉ chia dữ liệu, không giảm chiều)
    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'preprocess.py' trước.")
        return

    mnist_data = st.session_state['mnist_data']
    X_train = mnist_data['X_train']
    y_train = mnist_data['y_train']
    X_valid = mnist_data['X_valid']
    y_valid = mnist_data['y_valid']
    X_test = mnist_data['X_test']
    y_test = mnist_data['y_test']

    st.write(f"Train samples: {len(X_train)}, Validation samples: {len(X_valid)}, Test samples: {len(X_test)}")

    # Chuẩn hóa dữ liệu
    with st.spinner("Đang chuẩn hóa dữ liệu train..."):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_test_scaled = scaler.transform(X_test)

    # Chọn phương pháp giảm chiều
    method = st.selectbox("Chọn phương pháp giảm chiều để huấn luyện", ["PCA", "t-SNE"])

    # Cho phép người dùng đặt tên run ID
    run_name = st.text_input("Nhập tên Run ID cho giảm chiều (để trống để tự động tạo)", value="", max_chars=20, key="run_name_input")
    if run_name.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_{method}_{timestamp.replace(' ', '_').replace(':', '-')}"  # Định dạng tên run hợp lệ cho MLflow

    if method == "PCA":
        n_components = st.slider("Số thành phần PCA", 2, min(50, X_train_scaled.shape[1]), 2, key="pca_n_components")
        if st.button("Huấn luyện PCA"):
            # Đóng bất kỳ run nào đang hoạt động trước khi bắt đầu
            if mlflow.active_run():
                mlflow.end_run()

            with st.spinner("Đang huấn luyện PCA..."):
                pca = PCA(n_components=n_components, random_state=42)
                X_train_pca = pca.fit_transform(X_train_scaled)
                explained_variance_ratio = pca.explained_variance_ratio_.sum()

                # Hiển thị kết quả
                st.success(f"PCA hoàn tất! Tỷ lệ phương sai giải thích: {explained_variance_ratio:.4f}")
                plot_and_log_reduction(X_train_pca, y_train, "PCA", {"n_components": n_components}, None)

                # Lưu mô hình và dữ liệu giảm chiều vào session_state
                st.session_state['pca_model'] = pca
                st.session_state['X_train_pca'] = X_train_pca
                st.session_state['X_valid_pca'] = pca.transform(X_valid_scaled)
                st.session_state['X_test_pca'] = pca.transform(X_test_scaled)

                # Logging vào MLflow tại DAGSHUB_MLFLOW_URI
                with mlflow.start_run(run_name=run_name) as run:
                    mlflow.log_param("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    mlflow.log_param("run_id", run.info.run_id)
                    mlflow.log_param("method", "PCA")
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_param("n_samples", X_train_scaled.shape[0])
                    mlflow.log_metric("explained_variance_ratio", explained_variance_ratio)
                    mlflow.sklearn.log_model(pca, "pca_model", input_example=X_train_scaled[:1])
                    plot_and_log_reduction(X_train_pca, y_train, "PCA", {"n_components": n_components}, run)

                    run_id = run.info.run_id
                    mlflow_uri = st.session_state['mlflow_url']
                    st.success(f"PCA đã được huấn luyện và log vào DagsHub MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                    st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

    elif method == "t-SNE":
        perplexity = st.slider("Perplexity", 5, 50, 30, key="tsne_perplexity")
        n_iter = st.slider("Số lần lặp", 250, 1000, 500, key="tsne_n_iter")
        if st.button("Huấn luyện t-SNE"):
            # Đóng bất kỳ run nào đang hoạt động trước khi bắt đầu
            if mlflow.active_run():
                mlflow.end_run()

            with st.spinner("Đang huấn luyện t-SNE (có thể lâu với dữ liệu lớn)..."):
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
                X_train_tsne = tsne.fit_transform(X_train_scaled)

                # Hiển thị kết quả
                st.success("t-SNE hoàn tất!")
                plot_and_log_reduction(X_train_tsne, y_train, "t-SNE", {"perplexity": perplexity, "n_iter": n_iter}, None)

                # Lưu mô hình và dữ liệu giảm chiều vào session_state
                st.session_state['tsne_model'] = tsne
                st.session_state['X_train_tsne'] = X_train_tsne
                # t-SNE không có transform, dùng trên valid/test cần tính lại
                st.session_state['X_valid_tsne'] = tsne.fit_transform(X_valid_scaled)
                st.session_state['X_test_tsne'] = tsne.fit_transform(X_test_scaled)

                # Logging vào MLflow tại DAGSHUB_MLFLOW_URI
                with mlflow.start_run(run_name=run_name) as run:
                    mlflow.log_param("timestamp", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    mlflow.log_param("run_id", run.info.run_id)
                    mlflow.log_param("method", "t-SNE")
                    mlflow.log_param("perplexity", perplexity)
                    mlflow.log_param("n_iter", n_iter)
                    mlflow.log_param("n_samples", X_train_scaled.shape[0])
                    mlflow.sklearn.log_model(tsne, "tsne_model", input_example=X_train_scaled[:1])
                    plot_and_log_reduction(X_train_tsne, y_train, "t-SNE", {"perplexity": perplexity, "n_iter": n_iter}, run)

                    run_id = run.info.run_id
                    mlflow_uri = st.session_state['mlflow_url']
                    st.success(f"t-SNE đã được huấn luyện và log vào DagsHub MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                    st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

if __name__ == "__main__":
    train()