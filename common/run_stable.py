import streamlit as st
import sys
import importlib
import traceback
import os
import logging
from typing import Optional

# Đặt mức log của MLflow về WARNING để giảm cảnh báo không cần thiết (nếu cần MLflow)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Hàm chạy một file Streamlit ổn định
def run_stable_script(script_path: str) -> None:
    st.title("🌟 Chạy Ứng dụng Streamlit Ổn Định")

    # Kiểm tra file tồn tại
    if not os.path.exists(script_path):
        st.error(f"File '{script_path}' không tồn tại. Vui lòng kiểm tra đường dẫn.")
        return

    # Hiển thị thông báo trạng thái
    st.subheader(f"Đang chạy file: {script_path}")
    st.write("Nếu có lỗi, ứng dụng sẽ không sập mà hiển thị thông báo chi tiết.")

    try:
        # Import và chạy file Streamlit
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Gọi hàm chính (nếu có) hoặc chạy trực tiếp
        if hasattr(module, "__main__"):
            module.__main__()
        elif hasattr(module, "main"):
            module.main()
        elif hasattr(module, "introduce_mnist"):  # Kiểm tra hàm cụ thể trong introduce_mnist.py
            module.introduce_mnist()
        else:
            st.warning("Không tìm thấy hàm chính trong file. Vui lòng đảm bảo file có hàm `if __name__ == '__main__':` hoặc hàm `introduce_mnist`.")

    except Exception as e:
        st.error("Ứng dụng gặp lỗi, nhưng đã được xử lý để không sập:")
        st.write(f"**Lỗi chi tiết:** {str(e)}")
        if "File './exercises/exercise_3/data/X.pkl' không tồn tại" in str(e) or "Không thể tải dữ liệu MNIST" in str(e):
            st.write("**Giải pháp:**")
            st.write("- Đảm bảo file `X.pkl` và `y.pkl` tồn tại trong thư mục `exercises/exercise_3/data/`.");
            st.write("- Kiểm tra đường dẫn thư mục và tạo dữ liệu nếu cần (xem hướng dẫn bên dưới).")
            st.write("**Hướng dẫn tạo file .pkl:**")
            st.write("""
                1. Tạo một script Python (ví dụ: `generate_mnist_pkl.py`) trong thư mục gốc với nội dung sau:
                ```python
                import numpy as np
                from tensorflow.keras.datasets import mnist
                import pickle
                import os

                def generate_mnist_pkl(output_dir: str = "./exercises/exercise_3/data/"):
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    (X_train, y_train), (X_test, y_test) = mnist.load_data()
                    X = np.concatenate([X_train, X_test], axis=0) / 255.0  # Chuẩn hóa về [0, 1]
                    y = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
                    x_path = os.path.join(output_dir, "X.pkl")
                    y_path = os.path.join(output_dir, "y.pkl")
                    with open(x_path, 'wb') as f:
                        pickle.dump(X, f)
                    with open(y_path, 'wb') as f:
                        pickle.dump(y, f)
                    print(f"Dữ liệu MNIST đã được lưu vào: {x_path} và {y_path}")

                if __name__ == "__main__":
                    generate_mnist_pkl()