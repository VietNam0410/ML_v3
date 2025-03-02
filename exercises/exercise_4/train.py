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

# Thiết lập thông tin DagsHub
DAGSHUB_USERNAME = "VietNam0410"
DAGSHUB_REPO = "vn0410"

try:
    with st.spinner("Đang kết nối với DagsHub..."):
        dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
        mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
        os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    st.success("Đã kết nối với DagsHub thành công!")
except Exception as e:
    st.error(f"Không thể kết nối với DagsHub: {str(e)}. Sử dụng MLflow cục bộ.")
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def plot_and_log_reduction(X_reduced, y, method, params, run_name):
    """Vẽ biểu đồ scatter và log kết quả với MLflow."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="tab10", ax=ax, s=10)
    ax.set_title(f"{method} Visualization of MNIST")
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    st.pyplot(fig)

    # Lưu plot vào file tạm và log với MLflow
    plot_file = f"{method.lower()}_plot.png"
    fig.savefig(plot_file)
    mlflow.log_artifact(plot_file)
    os.remove(plot_file)

def train():
    st.header("Huấn luyện PCA và t-SNE trên MNIST 🧮")

    # Kiểm tra dữ liệu từ preprocess
    if 'mnist_data' not in st.session_state:
        st.error("Vui lòng chạy tiền xử lý dữ liệu trong 'preprocess.py' trước.")
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

    if method == "PCA":
        n_components = st.slider("Số thành phần PCA", 2, 50, 2)
        if st.button("Huấn luyện PCA"):
            with st.spinner("Đang huấn luyện PCA..."):
                pca = PCA(n_components=n_components)
                X_train_pca = pca.fit_transform(X_train_scaled)
                explained_variance_ratio = pca.explained_variance_ratio_.sum()

                # Logging với MLflow
                with mlflow.start_run(run_name=f"PCA_{n_components}_Components"):
                    mlflow.log_param("method", "PCA")
                    mlflow.log_param("n_components", n_components)
                    mlflow.log_param("n_samples", X_train_scaled.shape[0])
                    mlflow.log_metric("explained_variance_ratio", explained_variance_ratio)
                    mlflow.sklearn.log_model(pca, "pca_model")
                    plot_and_log_reduction(X_train_pca, y_train, "PCA", {"n_components": n_components}, "PCA")

                    # Lưu mô hình vào session_state
                    st.session_state['pca_model'] = pca
                    st.session_state['X_train_pca'] = X_train_pca
                    st.session_state['X_valid_pca'] = pca.transform(X_valid_scaled)
                    st.session_state['X_test_pca'] = pca.transform(X_test_scaled)

                st.success(f"PCA hoàn tất! Tỷ lệ phương sai giải thích: {explained_variance_ratio:.4f}")

    elif method == "t-SNE":
        perplexity = st.slider("Perplexity", 5, 50, 30)
        n_iter = st.slider("Số lần lặp", 250, 1000, 500)
        if st.button("Huấn luyện t-SNE"):
            with st.spinner("Đang huấn luyện t-SNE (có thể lâu với dữ liệu lớn)..."):
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
                X_train_tsne = tsne.fit_transform(X_train_scaled)

                # Logging với MLflow
                with mlflow.start_run(run_name=f"tSNE_Perplexity_{perplexity}"):
                    mlflow.log_param("method", "t-SNE")
                    mlflow.log_param("perplexity", perplexity)
                    mlflow.log_param("n_iter", n_iter)
                    mlflow.log_param("n_samples", X_train_scaled.shape[0])
                    mlflow.sklearn.log_model(tsne, "tsne_model")
                    plot_and_log_reduction(X_train_tsne, y_train, "t-SNE", {"perplexity": perplexity, "n_iter": n_iter}, "t-SNE")

                    # Lưu mô hình vào session_state
                    st.session_state['tsne_model'] = tsne
                    st.session_state['X_train_tsne'] = X_train_tsne
                    # t-SNE không có transform, dùng trên valid/test cần tính lại
                    st.session_state['X_valid_tsne'] = tsne.fit_transform(X_valid_scaled)
                    st.session_state['X_test_tsne'] = tsne.fit_transform(X_test_scaled)

                st.success("t-SNE hoàn tất!")

if __name__ == "__main__":
    train()