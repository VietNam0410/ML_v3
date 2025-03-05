import numpy as np
from tensorflow.keras.datasets import mnist
import pickle
import os

def download_and_save_mnist(output_dir: str = "./exercises/exercise_4/data/"):
    # Tạo thư mục nếu chưa tồn tại
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Tải dữ liệu MNIST (sử dụng context không kiểm tra SSL nếu cần)
    try:
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
    except Exception as e:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X = np.concatenate([X_train, X_test], axis=0) / 255.0  # Chuẩn hóa về [0, 1]
    y = np.concatenate([y_train, y_test], axis=0).astype(np.int32)

    # Lưu dữ liệu vào file .pkl
    x_path = os.path.join(output_dir, "X.pkl")
    y_path = os.path.join(output_dir, "y.pkl")

    with open(x_path, 'wb') as f:
        pickle.dump(X, f)
    with open(y_path, 'wb') as f:
        pickle.dump(y, f)

    print(f"Dữ liệu MNIST đã được lưu vào: {x_path} và {y_path}")

if __name__ == "__main__":
    download_and_save_mnist()