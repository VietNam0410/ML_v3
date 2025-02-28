import streamlit as st
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import mlflow
import os

# Thiết lập MLflow Tracking URI cục bộ
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def train_mnist_clustering():
    st.header("Huấn luyện Mô hình Phân Cụm MNIST 🧮")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment cho huấn luyện phân cụm", value="MNIST_Clustering")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Kiểm tra dữ liệu đã tiền xử lý trong session từ preprocessing.py
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.error("Dữ liệu MNIST chưa được tiền xử lý. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']
    total_samples = len(X_full)

    st.subheader("Dữ liệu MNIST đã xử lý 📝")
    st.write(f"Tổng số lượng mẫu: {total_samples}")

    # Cho người dùng chọn số lượng mẫu để tránh chậm trang
    max_samples = st.slider("Chọn số lượng mẫu tối đa (0 để dùng toàn bộ)", 0, total_samples, total_samples, step=100)
    if max_samples == 0:
        max_samples = total_samples
    elif max_samples > total_samples:
        st.error(f"Số lượng mẫu ({max_samples}) vượt quá tổng số mẫu có sẵn ({total_samples}). Đặt lại về {total_samples}.")
        max_samples = total_samples

    # Lấy mẫu ngẫu nhiên nếu max_samples < total_samples
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X_subset = X_full[indices]
        y_subset = y_full[indices]
    else:
        X_subset = X_full
        y_subset = y_full

    # Chuẩn bị dữ liệu: Flatten hình ảnh 28x28 thành vector 784 chiều
    X_data = X_subset.reshape(-1, 28 * 28)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data)

    # Chia dữ liệu train/test
    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    test_size = st.slider("Chọn kích thước tập kiểm tra (%)", min_value=10, max_value=40, value=20, step=5) / 100
    if st.button("Chia dữ liệu"):
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_subset, test_size=test_size, random_state=42)
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.success(f"Đã chia dữ liệu với kích thước: Huấn luyện {100-test_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%! ✅")
        st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
        st.write(f"Tập kiểm tra: {len(X_test)} mẫu")

    # Kiểm tra dữ liệu đã chia chưa
    if 'X_train' not in st.session_state or 'X_test' not in st.session_state:
        st.warning("Vui lòng chia dữ liệu trước khi huấn luyện mô hình. ⚠️")
        return

    X_train = st.session_state['X_train']
    X_test = st.session_state['X_test']

    # Xây dựng và huấn luyện mô hình phân cụm
    st.subheader("Xây dựng và Huấn luyện Mô hình Phân Cụm 🎯")
    model_choice = st.selectbox(
        "Chọn loại mô hình phân cụm",
        ["K-means", "DBSCAN"]
    )

    # Tham số huấn luyện cho từng mô hình
    if model_choice == "K-means":
        n_clusters = st.slider("Số cụm", 2, 20, 10, step=1)
        model_params = {"n_clusters": n_clusters}
    else:  # DBSCAN
        eps = st.slider("Epsilon (khoảng cách tối đa)", 0.1, 1.0, 0.5, step=0.1)
        min_samples = st.slider("Số mẫu tối thiểu", 1, 10, 5, step=1)
        model_params = {"eps": eps, "min_samples": min_samples}

    if st.button("Huấn luyện mô hình"):
        # Kiểm tra và kết thúc run hiện tại nếu có
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"{model_choice}_MNIST_{experiment_name}") as run:
            if model_choice == "K-means":
                model = KMeans(**model_params, random_state=42)
                clusters_train = model.fit_predict(X_train)
                clusters_test = model.predict(X_test)

                # Đánh giá bằng silhouette score (nếu có đủ cụm)
                silhouette_train = silhouette_score(X_train, clusters_train) if len(np.unique(clusters_train)) > 1 else None
                silhouette_test = silhouette_score(X_test, clusters_test) if len(np.unique(clusters_test)) > 1 else None

                st.write(f"Mô hình đã chọn: {model_choice}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Silhouette Score (Tập huấn luyện): {silhouette_train:.4f}" if silhouette_train else "Không thể tính Silhouette Score do số cụm không đủ đa dạng.")
                st.write(f"Silhouette Score (Tập kiểm tra): {silhouette_test:.4f}" if silhouette_test else "Không thể tính Silhouette Score do số cụm không đủ đa dạng.")

                # Log mô hình và metrics vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                if silhouette_train:
                    mlflow.log_metric("silhouette_train", silhouette_train)
                if silhouette_test:
                    mlflow.log_metric("silhouette_test", silhouette_test)
                mlflow.sklearn.log_model(model, "model", input_example=X_train[:1])
                mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])

            else:  # DBSCAN
                model = DBSCAN(**model_params)
                clusters_train = model.fit_predict(X_train)
                clusters_test = model.fit_predict(X_test)

                # Đánh giá bằng silhouette score (nếu có đủ cụm, loại bỏ noise)
                mask_train = clusters_train != -1  # Loại bỏ noise (-1)
                mask_test = clusters_test != -1
                silhouette_train = silhouette_score(X_train[mask_train], clusters_train[mask_train]) if len(np.unique(clusters_train[mask_train])) > 1 else None
                silhouette_test = silhouette_score(X_test[mask_test], clusters_test[mask_test]) if len(np.unique(clusters_test[mask_test])) > 1 else None

                st.write(f"Mô hình đã chọn: {model_choice}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Silhouette Score (Tập huấn luyện): {silhouette_train:.4f}" if silhouette_train else "Không thể tính Silhouette Score do số cụm không đủ đa dạng hoặc toàn bộ là noise.")
                st.write(f"Silhouette Score (Tập kiểm tra): {silhouette_test:.4f}" if silhouette_test else "Không thể tính Silhouette Score do số cụm không đủ đa dạng hoặc toàn bộ là noise.")
                st.write("Lưu ý: DBSCAN có thể phân loại một số điểm là noise (-1).")

                # Log mô hình và metrics vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                if silhouette_train:
                    mlflow.log_metric("silhouette_train", silhouette_train)
                if silhouette_test:
                    mlflow.log_metric("silhouette_test", silhouette_test)
                mlflow.sklearn.log_model(model, "model", input_example=X_train[:1])
                mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])

            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow thành công! ✅ (Run ID: {run.info.run_id})")

            # Lưu mô hình và scaler trong session để dùng cho demo
            st.session_state['mnist_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['clustering_metrics'] = {
                "silhouette_train": silhouette_train if silhouette_train else None,
                "silhouette_test": silhouette_test if silhouette_test else None
            }

if __name__ == "__main__":
    train_mnist_clustering()