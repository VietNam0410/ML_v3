import streamlit as st
import numpy as np
import pandas as pd
import openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import os
import dagshub

# Phần khởi tạo kết nối với DagsHub được comment để không truy cập ngay lập tức
# with st.spinner("Đang kết nối với DagsHub..."):
#     dagshub.init(repo_owner='VietNam0410', repo_name='vn0410', mlflow=True)
#     mlflow.set_tracking_uri(f"https://dagshub.com/VietNam0410/vn0410.mlflow")
# st.success("Đã kết nối với DagsHub thành công!")

# Cache dữ liệu MNIST
@st.cache_data
def load_mnist_data():
    with st.spinner("Đang tải dữ liệu MNIST từ OpenML..."):
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        X = X.values.reshape(-1, 28 * 28) / 255.0  # Chuẩn hóa về [0, 1]
        y = y.astype(np.int32)
    return X, y

def plot_reduction(X_reduced, y, method, params):
    """Vẽ biểu đồ scatter và lưu cục bộ."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=X_reduced[:, 0], y=X_reduced[:, 1], hue=y, palette="tab10", ax=ax, s=10)
    ax.set_title(f"{method} Visualization of MNIST")
    ax.set_xlabel(f"{method} Component 1")
    ax.set_ylabel(f"{method} Component 2")
    st.pyplot(fig)

    # Lưu plot vào file cục bộ
    plot_file = f"{method.lower()}_plot.png"
    fig.savefig(plot_file)
    st.info(f"Biểu đồ {method} đã được lưu cục bộ tại: {plot_file}")

def preprocess():
    st.header("Tiền xử lý dữ liệu MNIST: PCA và t-SNE 🧮")

    # Giới thiệu PCA
    st.subheader("1. PCA (Principal Component Analysis)")
    st.write("""
    PCA là một phương pháp giảm chiều tuyến tính, biến đổi dữ liệu thành các thành phần chính (principal components) 
    sao cho giữ được tối đa phương sai của dữ liệu. Các thông số điều chỉnh:
    - **Số thành phần (n_components):** Số chiều dữ liệu giảm xuống (mặc định 2 để trực quan hóa).
    """)

    # Giới thiệu t-SNE
    st.subheader("2. t-SNE (t-Distributed Stochastic Neighbor Embedding)")
    st.write("""
    t-SNE là một phương pháp phi tuyến tính, tập trung vào việc bảo tồn cấu trúc cục bộ của dữ liệu trong không gian 
    chiều thấp. Các thông số điều chỉnh:
    - **Perplexity:** Độ cân bằng giữa cấu trúc cục bộ và toàn cục (thường từ 5-50).
    - **Số lần lặp (n_iter):** Số lần tối ưu hóa (thường từ 250-1000).
    """)

    # Thiết lập experiment (không cần kết nối DagsHub)
    experiment_name = st.text_input("Nhập tên Experiment", value="MNIST_DimReduction")
    # if experiment_name:
    #     with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
    #         mlflow.set_experiment(experiment_name)

    # Tải dữ liệu MNIST
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_data()
        st.success(f"Đã tải dữ liệu MNIST: {st.session_state['X_full'].shape[0]} mẫu, {st.session_state['X_full'].shape[1]} đặc trưng")

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']

    # Chọn số mẫu
    max_samples = st.slider("Chọn số mẫu để xử lý (ít hơn để nhanh hơn)", 100, X_full.shape[0], 1000)
    if max_samples < X_full.shape[0]:
        indices = np.random.choice(X_full.shape[0], max_samples, replace=False)
        X_subset = X_full[indices]
        y_subset = y_full[indices]
    else:
        X_subset = X_full
        y_subset = y_full

    # Chia tập dữ liệu: test trước, sau đó train/validation
    st.subheader("Chia tập dữ liệu")
    test_size = st.slider("Tỷ lệ tập kiểm tra (%)", 10, 50, 20) / 100
    remaining_size = 1 - test_size
    train_size_relative = st.slider("Tỷ lệ tập huấn luyện (trong phần còn lại) (%)", 10, 90, 70) / 100

    # Tính toán tỷ lệ thực tế
    train_size = remaining_size * train_size_relative
    valid_size = remaining_size * (1 - train_size_relative)

    # Hiển thị tỷ lệ thực tế
    st.write(f"Tỷ lệ thực tế: Huấn luyện {train_size*100:.1f}%, Validation {valid_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%")
    st.write(f"Kiểm tra tổng tỷ lệ: {train_size*100 + valid_size*100 + test_size*100:.1f}% (phải luôn bằng 100%)")

    if st.button("Chia dữ liệu và giảm chiều"):
        with st.spinner("Đang chia dữ liệu..."):
            # Chia tập test trước
            X_temp, X_test, y_temp, y_test = train_test_split(X_subset, y_subset, test_size=test_size, random_state=42)
            # Chia tập train và validation từ phần còn lại
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, train_size=train_size_relative, random_state=42)

            # Lưu vào session_state
            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

            st.success(f"Đã chia dữ liệu: Train {len(X_train)}, Validation {len(X_valid)}, Test {len(X_test)}")

            # Chuẩn hóa dữ liệu
            with st.spinner("Đang chuẩn hóa dữ liệu..."):
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_valid)
                X_test_scaled = scaler.transform(X_test)

            # Giảm chiều bằng PCA
            with st.spinner("Đang giảm chiều bằng PCA..."):
                pca = PCA(n_components=2)
                X_train_pca = pca.fit_transform(X_train_scaled)
                X_valid_pca = pca.transform(X_valid_scaled)
                X_test_pca = pca.transform(X_test_scaled)
                explained_variance_ratio = pca.explained_variance_ratio_.sum()

                st.success(f"PCA hoàn tất! Tỷ lệ phương sai giải thích: {explained_variance_ratio:.4f}")
                plot_reduction(X_train_pca, y_train, "PCA", {"n_components": 2})

                st.session_state['pca_model'] = pca
                st.session_state['X_train_pca'] = X_train_pca
                st.session_state['X_valid_pca'] = X_valid_pca
                st.session_state['X_test_pca'] = X_test_pca

            # Giảm chiều bằng t-SNE
            with st.spinner("Đang giảm chiều bằng t-SNE (có thể lâu với dữ liệu lớn)..."):
                tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
                X_train_tsne = tsne.fit_transform(X_train_scaled)
                X_valid_tsne = tsne.fit_transform(X_valid_scaled)  # t-SNE không có transform, cần tính lại
                X_test_tsne = tsne.fit_transform(X_test_scaled)

                st.success("t-SNE hoàn tất!")
                plot_reduction(X_train_tsne, y_train, "t-SNE", {"perplexity": 30, "n_iter": 500})

                st.session_state['tsne_model'] = tsne
                st.session_state['X_train_tsne'] = X_train_tsne
                st.session_state['X_valid_tsne'] = X_valid_tsne
                st.session_state['X_test_tsne'] = X_test_tsne

            # Comment phần logging với MLflow
            # with mlflow.start_run(run_name=f"Data_Split_{max_samples}") as run:
            #     mlflow.log_param("max_samples", max_samples)
            #     mlflow.log_param("test_size", test_size)
            #     mlflow.log_param("valid_size", valid_size)
            #     mlflow.log_metric("train_samples", len(X_train))
            #     mlflow.log_metric("valid_samples", len(X_valid))
            #     mlflow.log_metric("test_samples", len(X_test))

            # with mlflow.start_run(run_name=f"PCA_{max_samples}") as run:
            #     mlflow.log_param("method", "PCA")
            #     mlflow.log_param("n_components", 2)
            #     mlflow.log_param("n_samples", X_train_scaled.shape[0])
            #     mlflow.log_metric("explained_variance_ratio", explained_variance_ratio)
            #     mlflow.sklearn.log_model(pca, "pca_model")
            #     plot_reduction(X_train_pca, y_train, "PCA", {"n_components": 2})

            # with mlflow.start_run(run_name=f"tSNE_{max_samples}") as run:
            #     mlflow.log_param("method", "t-SNE")
            #     mlflow.log_param("perplexity", 30)
            #     mlflow.log_param("n_iter", 500)
            #     mlflow.log_param("n_samples", X_train_scaled.shape[0])
            #     mlflow.sklearn.log_model(tsne, "tsne_model")
            #     plot_reduction(X_train_tsne, y_train, "t-SNE", {"perplexity": 30, "n_iter": 500})

if __name__ == "__main__":
    preprocess()