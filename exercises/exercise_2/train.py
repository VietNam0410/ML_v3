import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
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
    
    # Kiểm tra dữ liệu đã tiền xử lý trong session từ preprocess_mnist.py
    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    # Kiểm tra key 'X_train' trước khi truy cập
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

    # Chuẩn hóa dữ liệu cho các mô hình
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)

    # Xây dựng và huấn luyện mô hình
    st.subheader("Xây dựng và Huấn luyện Mô hình 🎯")
    model_choice = st.selectbox(
        "Chọn loại mô hình",
        ["SVM (Support Vector Machine)", "Decision Tree"]
    )

    # Tham số huấn luyện cho từng mô hình
    if model_choice == "SVM (Support Vector Machine)":
        kernel = st.selectbox("Chọn kernel SVM", ["linear", "rbf", "poly"], index=1)
        C = st.slider("C (Tham số điều chỉnh)", 0.1, 10.0, 1.0, step=0.1)
        model_params = {"kernel": kernel, "C": C}
    else:  # Decision Tree
        max_depth = st.slider("Độ sâu tối đa", 1, 20, 10, step=1)
        min_samples_split = st.slider("Số mẫu tối thiểu để chia", 2, 20, 2, step=1)
        model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}

    if st.button("Huấn luyện mô hình"):
        # Kiểm tra và kết thúc run hiện tại nếu có
        active_run = mlflow.active_run()
        if active_run:
            mlflow.end_run()

        with mlflow.start_run(run_name=f"{model_choice}_MNIST_{experiment_name}") as run:
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
            
            # Log mô hình, scaler, và metrics vào MLflow
            mlflow.log_params(model_params)
            mlflow.log_param("model_type", model_choice)
            mlflow.log_metric("train_accuracy", train_acc)
            mlflow.log_metric("valid_accuracy", valid_acc)
            mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
            mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])

            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow thành công! ✅ (Run ID: {run.info.run_id})")

            # Lưu mô hình và scaler trong session để dùng cho demo
            st.session_state['mnist_model'] = model
            st.session_state['scaler'] = scaler
            st.session_state['training_metrics'] = {"train_accuracy": train_acc, "valid_accuracy": valid_acc}

if __name__ == "__main__":
    train_mnist()