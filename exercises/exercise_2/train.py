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
    
    # Kiểm tra dữ liệu đã tiền xử lý trong session
    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    # Kiểm tra key 'X_train' trước khi truy cập
    # Sửa đổi bởi Grok 3: Thêm kiểm tra chi tiết hơn cho mnist_data và các key
    mnist_data = st.session_state['mnist_data']
    if 'X_train' not in mnist_data or 'y_train' not in mnist_data:
        st.error("Dữ liệu 'X_train' hoặc 'y_train' không tồn tại trong session. Vui lòng hoàn tất tiền xử lý và chia dữ liệu trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    st.subheader("Dữ liệu MNIST đã xử lý 📝")
    st.write("Đây là dữ liệu sau các bước tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST':")
    st.write(f"Số lượng mẫu huấn luyện: {len(mnist_data['X_train'])}")
    st.write(f"Số lượng mẫu validation: {len(mnist_data.get('X_valid', []))}")
    st.write(f"Số lượng mẫu kiểm tra: {len(mnist_data['X_test'])}")

    # Chuẩn bị dữ liệu: Flatten hình ảnh 28x28 thành vector 784 chiều
    X_train = mnist_data['X_train'].reshape(-1, 28 * 28)
    y_train = mnist_data['y_train']
    X_valid = mnist_data.get('X_valid', mnist_data['X_test']).reshape(-1, 28 * 28)
    y_valid = mnist_data.get('y_valid', mnist_data['y_test'])
    X_test = mnist_data['X_test'].reshape(-1, 28 * 28)

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
        # Sửa đổi bởi Grok 3: Thêm log huấn luyện vào MLflow
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
                
                # Log mô hình và metrics vào MLflow
                mlflow.log_params(model_params)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("valid_accuracy", valid_acc)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])

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

                # Log mô hình và metrics vào MLflow
                mlflow.log_params(model_params)
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

                # Log mô hình và metrics vào MLflow
                mlflow.log_params(model_params)
                if silhouette:
                    mlflow.log_metric("silhouette_score", silhouette)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])

            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow thành công! ✅")

            # Lưu mô hình và metrics trong session để dùng cho demo (tùy chọn)
            st.session_state['mnist_model'] = model
            st.session_state['model_params'] = model_params
            st.session_state['training_metrics'] = {"train_accuracy": train_acc if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None, 
                                                  "valid_accuracy": valid_acc if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None,
                                                  "silhouette_score": silhouette if model_choice in ["K-means (Clustering)", "DBSCAN (Clustering)"] else None}

if __name__ == "__main__":
    train_mnist()