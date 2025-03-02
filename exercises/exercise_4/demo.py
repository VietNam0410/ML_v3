import streamlit as st
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
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

def demo():
    st.header("Demo Giảm chiều MNIST với PCA và t-SNE 🖌️")
    experiment_name = "MNIST_DimReduction"

    # Kiểm tra dữ liệu và mô hình từ train
    if 'mnist_data' not in st.session_state:
        st.error("Vui lòng chạy tiền xử lý dữ liệu trong 'preprocess.py' trước.")
        return
    if 'pca_model' not in st.session_state and 'tsne_model' not in st.session_state:
        st.error("Vui lòng huấn luyện ít nhất một mô hình trong 'train.py' trước.")
        return

    mnist_data = st.session_state['mnist_data']
    y_valid = mnist_data['y_valid']
    y_test = mnist_data['y_test']

    # Chọn tập dữ liệu để demo
    dataset_choice = st.selectbox("Chọn tập dữ liệu để trực quan hóa", ["Validation", "Test"])
    if dataset_choice == "Validation":
        y_subset = y_valid
    else:
        y_subset = y_test

    # Hiển thị kết quả PCA
    if 'pca_model' in st.session_state:
        st.subheader("Kết quả PCA")
        X_valid_pca = st.session_state['X_valid_pca'] if dataset_choice == "Validation" else st.session_state['X_test_pca']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_valid_pca[:, 0], y=X_valid_pca[:, 1], hue=y_subset, palette="tab10", ax=ax, s=10)
        ax.set_title(f"PCA Visualization ({dataset_choice} Set)")
        ax.set_xlabel("PCA Component 1")
        ax.set_ylabel("PCA Component 2")
        st.pyplot(fig)

    # Hiển thị kết quả t-SNE
    if 'tsne_model' in st.session_state:
        st.subheader("Kết quả t-SNE")
        X_valid_tsne = st.session_state['X_valid_tsne'] if dataset_choice == "Validation" else st.session_state['X_test_tsne']
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=X_valid_tsne[:, 0], y=X_valid_tsne[:, 1], hue=y_subset, palette="tab10", ax=ax, s=10)
        ax.set_title(f"t-SNE Visualization ({dataset_choice} Set)")
        ax.set_xlabel("t-SNE Component 1")
        ax.set_ylabel("t-SNE Component 2")
        st.pyplot(fig)

    # Tải runs từ MLflow để xem lịch sử
    if st.button("Xem lịch sử huấn luyện từ DagsHub"):
        with st.spinner("Đang tải lịch sử Runs từ DagsHub..."):
            runs = mlflow.search_runs(experiment_names=[experiment_name])
            if not runs.empty:
                st.write("Danh sách Runs đã huấn luyện:")
                st.write(runs[['run_id', 'tags.mlflow.runName', 'params.method', 'start_time']])
            else:
                st.write("Chưa có run nào được log.")

if __name__ == "__main__":
    demo()