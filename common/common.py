import streamlit as st
import numpy as np
from tensorflow.keras.datasets import mnist
import openml
import logging
import mlflow
from typing import Tuple, Optional, Any

# Đặt mức log của MLflow về WARNING để giảm cảnh báo không cần thiết
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Cache dữ liệu MNIST để tái sử dụng giữa các bài tập
@st.cache_data(ttl=86400)  # Làm mới sau 24 giờ
def load_mnist(max_samples: int = 70000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tải dữ liệu MNIST với giới hạn số mẫu để tránh crash, xử lý lỗi và tái sử dụng.
    """
    with st.spinner("Đang tải dữ liệu MNIST..."):
        try:
            # Thử tải từ OpenML
            dataset = openml.datasets.get_dataset(554)
            X, y, _, _ = dataset.get_data(target='class')
            X = X.values.reshape(-1, 28 * 28) / 255.0  # Làm phẳng và chuẩn hóa
            y = y.astype(np.int32)
        except Exception as e:
            st.error(f"Không thể tải từ OpenML: {str(e)}. Sử dụng TensorFlow.")
            try:
                (X_train, y_train), (X_test, y_test) = mnist.load_data()
                X = np.concatenate([X_train, X_test], axis=0) / 255.0
                y = np.concatenate([y_train, y_test], axis=0)
                X = X.reshape(-1, 28 * 28)
            except Exception as tf_e:
                st.error(f"Lỗi khi tải từ TensorFlow: {str(e)}. Ứng dụng sẽ dừng lại.")
                return None, None

        # Giới hạn số mẫu để tránh crash
        total_samples = len(X)
        if max_samples == 0 or max_samples > total_samples:
            st.warning(f"Số mẫu {max_samples} vượt quá {total_samples}. Dùng toàn bộ.")
            max_samples = total_samples
        elif max_samples < total_samples:
            indices = np.random.choice(total_samples, max_samples, replace=False)
            X, y = X[indices], y[indices]

        return X, y

# Hàm kiểm tra và xử lý lỗi chung
def handle_error(error_msg: str, action: str = "dừng") -> None:
    """
    Hiển thị thông báo lỗi và xử lý (dừng hoặc tiếp tục).
    """
    st.error(f"Lỗi: {error_msg}")
    if action == "dừng":
        st.stop()
    else:
        st.write("Ứng dụng sẽ tiếp tục chạy, nhưng một số chức năng có thể bị ảnh hưởng.")

# Hàm tải mô hình từ MLflow một cách an toàn
def load_model_safe(run_id: str, artifact_path: str = "model") -> Optional[Any]:
    """
    Tải mô hình từ MLflow với kiểm tra flavor 'python_function' và xử lý lỗi.
    """
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/{artifact_path}")
        if "python_function" not in mlflow.models.get_model_info(f"runs:/{run_id}/{artifact_path}").flavors:
            st.warning("Mô hình không có flavor 'python_function'. Vui lòng kiểm tra cách log mô hình hoặc sử dụng flavor khác.")
            return model  # Trả về mô hình dù không có python_function
        return model
    except mlflow.exceptions.MlflowException as e:
        handle_error(f"Lỗi khi tải mô hình từ MLflow (Run ID: {run_id}): {str(e)}", "tiếp tục")
        return None
    except Exception as e:
        handle_error(f"Lỗi không xác định khi tải mô hình: {str(e)}", "tiếp tục")
        return None

# Hàm cấu hình MLflow/DagsHub
def configure_mlflow() -> tuple[str, Any]:
    """
    Cấu hình MLflow và DagsHub, trả về URI và client.
    """
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

    client = mlflow.tracking.MlflowClient()
    return DAGSHUB_MLFLOW_URI, client