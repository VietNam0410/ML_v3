import streamlit as st
import numpy as np
import pickle
import os
from typing import Tuple, Optional
import logging

# Đặt mức log của MLflow về WARNING để giảm cảnh báo không cần thiết (nếu cần MLflow)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Cache dữ liệu MNIST để tái sử dụng giữa các bài tập
@st.cache_data(ttl=86400)  # Làm mới sau 24 giờ
def load_mnist(data_dir: str = "./exercises/exercise_3/data/", max_samples: int = 70000) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tải dữ liệu MNIST từ các file .pkl trong thư mục được chỉ định.
    Xử lý lỗi và giới hạn tài nguyên để tránh crash.
    """
    with st.spinner("Đang tải dữ liệu MNIST từ file .pkl, vui lòng đợi một chút..."):
        try:
            # Kiểm tra thư mục tồn tại
            if not os.path.exists(data_dir):
                st.error(f"Thư mục '{data_dir}' không tồn tại. Vui lòng kiểm tra đường dẫn.")
                return None, None

            # Tải dữ liệu X từ file X.pkl
            x_path = os.path.join(data_dir, "X.pkl")
            if not os.path.exists(x_path):
                st.error(f"File '{x_path}' không tồn tại. Vui lòng kiểm tra thư mục data.")
                return None, None

            with open(x_path, 'rb') as f:
                X = pickle.load(f)
                # Đảm bảo X có kích thước phù hợp (28x28x1 hoặc 28x28)
                if len(X.shape) == 4 and X.shape[3] == 1:  # Nếu đã là 28x28x1
                    X = X.reshape(-1, 28, 28, 1) / 255.0  # Chuẩn hóa về [0, 1]
                elif len(X.shape) == 3 and X.shape[1:] == (28, 28):  # Nếu là 28x28
                    X = X.reshape(-1, 28, 28, 1) / 255.0  # Thêm kênh và chuẩn hóa
                else:
                    st.error(f"Kích thước dữ liệu X không đúng: {X.shape}. Định dạng phải là 28x28 hoặc 28x28x1.")
                    return None, None

            # Tải dữ liệu y từ file y.pkl
            y_path = os.path.join(data_dir, "y.pkl")
            if not os.path.exists(y_path):
                st.error(f"File '{y_path}' không tồn tại. Vui lòng kiểm tra thư mục data.")
                return None, None

            with open(y_path, 'rb') as f:
                y = pickle.load(f)
                y = y.astype(np.int32)  # Đảm bảo y là kiểu int32

            # Kiểm tra kích thước dữ liệu
            if len(X) != len(y):
                st.error(f"Số lượng mẫu trong X ({len(X)}) không khớp với y ({len(y)}).")
                return None, None

            # Giới hạn số mẫu để tránh crash (nếu cần)
            total_samples = len(X)
            if max_samples == 0 or max_samples > total_samples:
                st.warning(f"Số mẫu {max_samples} vượt quá {total_samples}. Dùng toàn bộ.")
                max_samples = total_samples
            elif max_samples < total_samples:
                indices = np.random.choice(total_samples, max_samples, replace=False)
                X, y = X[indices], y[indices]

            return X, y

        except pickle.UnpicklingError as e:
            st.error(f"Lỗi khi tải file pickle: {str(e)}. Vui lòng kiểm tra file X.pkl và y.pkl.")
            return None, None
        except Exception as e:
            st.error(f"Lỗi không xác định khi tải dữ liệu: {str(e)}. Ứng dụng sẽ tiếp tục nhưng không tải được dữ liệu.")
            return None, None

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