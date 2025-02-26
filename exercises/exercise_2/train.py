import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, silhouette_score  # Dùng silhouette_score cho clustering
from sklearn.preprocessing import StandardScaler, LabelEncoder

def train_mnist():
    st.header("Huấn luyện Mô hình Nhận diện Chữ số MNIST 🧮")

    # Cho người dùng đặt tên Experiment (giờ chỉ để hiển thị, không dùng MLflow)
    experiment_name = st.text_input("Nhập Tên Experiment cho Huấn luyện", value="MNIST_Training")
    
    # Kiểm tra dữ liệu đã tiền xử lý trong session
    if 'processed_mnist' not in st.session_state:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return
    mnist_data = st.session_state['processed_mnist']

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

    # Chuẩn hóa dữ liệu cho các mô hình (SVM, Decision Tree, K-means, DBSCAN)
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
            
            # Lưu mô hình trong session
            # Sửa đổi bởi Grok 3: Loại bỏ MLflow, lưu trong session_state
            st.session_state['mnist_model'] = model
            st.session_state['model_params'] = model_params
            st.session_state['training_metrics'] = {"train_accuracy": train_acc, "valid_accuracy": valid_acc}
            st.success(f"Huấn luyện {model_choice} hoàn tất và lưu trong session! ✅")
        
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

            # Lưu mô hình trong session
            # Sửa đổi bởi Grok 3: Loại bỏ MLflow, lưu trong session_state
            st.session_state['mnist_model'] = model
            st.session_state['model_params'] = model_params
            st.session_state['clustering_result'] = clusters
            st.success(f"Huấn luyện {model_choice} hoàn tất và lưu trong session! ✅")

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

            # Lưu mô hình trong session
            # Sửa đổi bởi Grok 3: Loại bỏ MLflow, lưu trong session_state
            st.session_state['mnist_model'] = model
            st.session_state['model_params'] = model_params
            st.session_state['clustering_result'] = clusters
            st.success(f"Huấn luyện {model_choice} hoàn tất và lưu trong session! ✅")

if __name__ == "__main__":
    train_mnist()