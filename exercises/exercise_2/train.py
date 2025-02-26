import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import mlflow
import os

# Thiết lập MLflow Tracking URI cục bộ
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def train_mnist():
    st.header("Huấn luyện Mô hình Nhận diện Chữ số MNIST 🧮")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment cho Huấn luyện", value="MNIST_Training")
    if experiment_name:
        mlflow.set_experiment(experiment_name)
    
    # Kiểm tra dữ liệu đã tiền xử lý trong MLflow (thay vì session_state)
    # Sửa đổi bởi Grok 3: Loại bỏ session_state, chỉ dùng MLflow
    preprocess_runs = mlflow.search_runs(experiment_names=["MNIST_Preprocessing"])
    if preprocess_runs.empty:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy trong MLflow. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    latest_preprocess_run_id = preprocess_runs['run_id'].iloc[0]
    split_runs = mlflow.search_runs(experiment_names=["MNIST_Preprocessing"], filter_string=f"tags.mlflow.runName LIKE '%Split%'")
    if split_runs.empty:
        st.error("Dữ liệu chia tách không tìm thấy trong MLflow. Vui lòng chia tách dữ liệu trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    latest_split_run_id = split_runs['run_id'].iloc[0]

    # Tải dữ liệu từ MLflow
    X_train_path = mlflow.artifacts.download_artifacts(run_id=latest_split_run_id, path="X_train.npy")
    y_train_path = mlflow.artifacts.download_artifacts(run_id=latest_split_run_id, path="y_train.npy")
    X_valid_path = mlflow.artifacts.download_artifacts(run_id=latest_split_run_id, path="X_valid.npy")
    y_valid_path = mlflow.artifacts.download_artifacts(run_id=latest_split_run_id, path="y_valid.npy")
    X_test_path = mlflow.artifacts.download_artifacts(run_id=latest_split_run_id, path="X_test.npy")

    X_train = np.load(X_train_path).reshape(-1, 28 * 28)
    y_train = np.load(y_train_path)
    X_valid = np.load(X_valid_path).reshape(-1, 28 * 28)
    y_valid = np.load(y_valid_path)
    X_test = np.load(X_test_path).reshape(-1, 28 * 28)

    st.subheader("Dữ liệu MNIST đã xử lý từ MLflow 📝")
    st.write(f"Số lượng mẫu huấn luyện: {len(X_train)}")
    st.write(f"Số lượng mẫu validation: {len(X_valid)}")
    st.write(f"Số lượng mẫu kiểm tra: {len(X_test)}")

    # Chuẩn hóa dữ liệu cho các mô hình
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    X_test_scaled = scaler.transform(X_test)

    # Xây dựng và huấn luyện mô hình
    st.subheader("Xây dựng và Huấn luyện Mô hình 🎯")
    model_choice = st.selectbox(
        "Chọn loại mô hình",
        ["SVM (Support Vector Machine)", "Decision Tree", "K-means (Clustering)", "DBSCAN (Clustering)"]
    )

    # Tham số huấn luyện cho từng mô hình
    if model_choice == "SVM (Support Vector Machine)":
        kernel = st.selectbox("Chọn kernel SVM", ["linear", "rbf", "poly"], index=1)
        C = st.slider("C (Regularization parameter)", 0.1, 10.0, 1.0, step=0.1)
        model_params = {"kernel": kernel, "C": C}
    elif model_choice == "Decision Tree":
        max_depth = st.slider("Độ sâu tối đa", 1, 20, 10, step=1)
        min_samples_split = st.slider("Số mẫu tối thiểu để chia", 2, 20, 2, step=1)
        model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}
    elif model_choice == "K-means (Clustering)":
        n_clusters = st.slider("Số cụm", 2, 20, 10, step=1)
        model_params = {"n_clusters": n_clusters}
    else:  # DBSCAN
        eps = st.slider("Epsilon (khoảng cách tối đa)", 0.1, 1.0, 0.5, step=0.1)
        min_samples = st.slider("Số mẫu tối thiểu", 1, 10, 5, step=1)
        model_params = {"eps": eps, "min_samples": min_samples}

    if st.button("Huấn luyện mô hình"):
        # Kiểm tra và kết thúc run hiện tại nếu có
        # Sửa đổi bởi Grok 3: Loại bỏ session_state, chỉ dùng MLflow
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"{model_choice}_MNIST_{experiment_name}"):
            if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"]:
                # Huấn luyện mô hình phân loại
                if model_choice == "SVM (Support Vector Machine)":
                    model = SVC(**model_params, random_state=42)
                else:  # Decision Tree
                    model = DecisionTreeClassifier(**model_params, random_state=42)

                model.fit(X_train_scaled, y_train)
                train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
                valid_acc = accuracy_score(y_valid, model.predict(X_valid_scaled))

                st.write(f"Mô hình đã chọn: {model_choice}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Độ chính xác huấn luyện: {train_acc:.4f}")
                st.write(f"Độ chính xác validation: {valid_acc:.4f}")
                
                # Log mô hình, metrics, và scaler vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("valid_accuracy", valid_acc)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
                mlflow.sklearn.log_model(scaler, "scaler")  # Log scaler để dùng trong demo

            elif model_choice == "K-means (Clustering)":
                # Huấn luyện mô hình phân cụm (không dùng nhãn)
                model = KMeans(**model_params, random_state=42)
                clusters = model.fit_predict(X_train_scaled)

                # Đánh giá bằng silhouette score (nếu có nhãn, chỉ để minh họa)
                silhouette = silhouette_score(X_train_scaled, clusters) if len(np.unique(y_train)) > 1 else None
                st.write(f"Mô hình đã chọn: {model_choice}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Silhouette Score (nếu có): {silhouette:.4f}" if silhouette else "Không thể tính Silhouette Score do nhãn không đủ đa dạng.")
                st.write("Lưu ý: K-means không dự đoán nhãn, chỉ phân chia thành cụm.")

                # Log mô hình, metrics, và model_choice vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                if silhouette:
                    mlflow.log_metric("silhouette_score", silhouette)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])

            elif model_choice == "DBSCAN (Clustering)":
                # Huấn luyện mô hình phân cụm (không dùng nhãn)
                model = DBSCAN(**model_params)
                clusters = model.fit_predict(X_train_scaled)

                # Đánh giá bằng silhouette score (nếu có nhãn, chỉ để minh họa)
                mask = clusters != -1  # Loại bỏ noise (-1)
                silhouette = silhouette_score(X_train_scaled[mask], clusters[mask]) if len(np.unique(clusters[mask])) > 1 else None
                st.write(f"Mô hình đã chọn: {model_choice}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Silhouette Score (nếu có): {silhouette:.4f}" if silhouette else "Không thể tính Silhouette Score do nhãn không đủ đa dạng.")
                st.write("Lưu ý: DBSCAN không dự đoán nhãn, chỉ phân chia thành cụm. Giá trị -1 là noise.")

                # Log mô hình, metrics, và model_choice vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                if silhouette:
                    mlflow.log_metric("silhouette_score", silhouette)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])

            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow thành công! ✅")

if __name__ == "__main__":
    train_mnist()