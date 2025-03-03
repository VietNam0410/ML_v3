import streamlit as st
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import mlflow
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

# Hàm chuẩn hóa dữ liệu với cache
@st.cache_data
def scale_data(X_train, X_valid):
    """Chuẩn hóa dữ liệu bằng StandardScaler và lưu vào bộ nhớ đệm."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    return X_train_scaled, X_valid_scaled, scaler

def train_mnist():
    st.header("Huấn luyện Mô hình Nhận diện Chữ số MNIST 🧮")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập Tên Experiment cho Huấn luyện", value="MNIST_Training")
    with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
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

    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    mnist_data = st.session_state['mnist_data']
    if 'X_train' not in mnist_data or 'y_train' not in mnist_data:
        st.error("Dữ liệu 'X_train' hoặc 'y_train' không tồn tại trong session. Vui lòng hoàn tất tiền xử lý và chia dữ liệu trong 'Tiền xử lý Dữ liệu MNIST' trước.")
        return

    st.subheader("Dữ liệu MNIST đã xử lý 📝")
    st.write("Đây là dữ liệu sau các bước tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST':")
    st.write(f"Số lượng mẫu huấn luyện: {len(mnist_data['X_train'])}")
    st.write(f"Số lượng mẫu validation: {len(mnist_data.get('X_valid', []))}")
    st.write(f"Số lượng mẫu kiểm tra: {len(mnist_data['X_test'])}")

    # Reshape dữ liệu từ (n, 28, 28, 1) thành (n, 784)
    X_train = mnist_data['X_train'].reshape(-1, 28 * 28)
    y_train = mnist_data['y_train']
    X_valid = mnist_data.get('X_valid', mnist_data['X_test']).reshape(-1, 28 * 28)
    y_valid = mnist_data.get('y_valid', mnist_data['y_test'])

    # Chuẩn hóa dữ liệu với cache
    X_train_scaled, X_valid_scaled, scaler = scale_data(X_train, X_valid)

    st.subheader("Xây dựng và Huấn luyện Mô hình 🎯")

    # Giới thiệu các mô hình và tham số
    st.write("### Giới thiệu các mô hình và tham số")
    st.write("#### SVM (Support Vector Machine)")
    st.write("- **Kernel**: Hàm kernel (linear, rbf, poly) xác định cách phân tách dữ liệu (linear: đường thẳng, rbf: không tuyến tính, poly: đa thức).")
    st.write("- **C**: Tham số điều chỉnh (giá trị lớn giảm regularization, dễ overfitting; giá trị nhỏ tăng regularization, giảm overfitting).")
    st.write("#### Decision Tree")
    st.write("- **max_depth**: Độ sâu tối đa của cây (giới hạn để tránh overfitting).")
    st.write("- **min_samples_split**: Số mẫu tối thiểu để chia một node (giúp kiểm soát độ phức tạp cây).")

    model_choice = st.selectbox(
        "Chọn loại mô hình",
        ["SVM (Support Vector Machine)", "Decision Tree"]
    )

    if model_choice == "SVM (Support Vector Machine)":
        kernel = st.selectbox("Chọn kernel SVM", ["linear", "rbf", "poly"], index=1)
        C = st.slider("C (Tham số điều chỉnh)", 0.1, 10.0, 1.0, step=0.1)
        model_params = {"kernel": kernel, "C": C}
    else:
        max_depth = st.slider("Độ sâu tối đa", 1, 20, 10, step=1)
        min_samples_split = st.slider("Số mẫu tối thiểu để chia", 2, 20, 2, step=1)
        model_params = {"max_depth": max_depth, "min_samples_split": min_samples_split}

    # Cho phép người dùng đặt tên run ID
    run_name = st.text_input("Nhập tên Run ID cho mô hình (để trống để tự động tạo)", value="", max_chars=20, key="run_name_input")
    if run_name.strip() == "":
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        run_name = f"MNIST_Model_{timestamp.replace(' ', '_').replace(':', '-')}"  # Định dạng tên run hợp lệ cho MLflow

    if st.button("Huấn luyện mô hình"):
        with st.spinner("Đang huấn luyện mô hình..."):
            if model_choice == "SVM (Support Vector Machine)":
                model = SVC(**model_params, random_state=42)
            else:
                model = DecisionTreeClassifier(**model_params, random_state=42)

            model.fit(X_train_scaled, y_train)
            train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
            valid_acc = accuracy_score(y_valid, model.predict(X_valid_scaled))

            st.write(f"Mô hình đã chọn: {model_choice}")
            st.write(f"Tham số: {model_params}")
            st.write(f"Độ chính xác huấn luyện: {train_acc:.4f}")
            st.write(f"Độ chính xác validation: {valid_acc:.4f}")
            st.success(f"Huấn luyện {model_choice} hoàn tất cục bộ ✅.")

            # Logging và tracking vào MLflow/DagsHub với chi tiết ngày giờ và ID
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with mlflow.start_run(run_name=run_name) as run:
                mlflow.log_param("timestamp", timestamp)
                mlflow.log_param("run_id", run.info.run_id)
                mlflow.log_params(model_params)
                mlflow.log_param("model_type", model_choice)
                mlflow.log_metric("train_accuracy", train_acc)
                mlflow.log_metric("valid_accuracy", valid_acc)
                mlflow.sklearn.log_model(model, "model", input_example=X_train_scaled[:1])
                mlflow.sklearn.log_model(scaler, "scaler", input_example=X_train[:1])

                run_id = run.info.run_id
                mlflow_uri = st.session_state['mlflow_url']
                st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow thành công! ✅ (Tên Run: {run_name}, Run ID: {run_id}, Thời gian: {timestamp})")
                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

            # Lưu mô hình và scaler vào st.session_state để sử dụng trong demo
            st.session_state['mnist_model'] = model
            st.session_state['mnist_scaler'] = scaler  # Đảm bảo lưu scaler vào session_state
            st.session_state['training_metrics'] = {"train_accuracy": train_acc, "valid_accuracy": valid_acc}

if __name__ == "__main__":
    train_mnist()