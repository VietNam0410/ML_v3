import streamlit as st
import numpy as np
import openml
from tensorflow.keras.datasets import mnist  # Backup nếu OpenML không tải được

# Sử dụng st.cache_data để cache dữ liệu, tăng tốc độ load
@st.cache_data
def load_mnist_from_openml():
    try:
        # Tải dữ liệu MNIST từ OpenML (ID dataset MNIST là 554)
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        
        # Chuyển đổi X (DataFrame) thành mảng numpy và chuẩn hóa (28x28x1)
        X = X.values.reshape(-1, 28, 28, 1) / 255.0  # Sử dụng .values để lấy mảng numpy từ DataFrame
        y = y.astype(np.int32)
        
        return X, y
    except Exception as e:
        st.error(f"Không thể tải dữ liệu từ OpenML. Sử dụng dữ liệu từ TensorFlow: {str(e)}")
        # Fallback: Tải từ TensorFlow nếu OpenML thất bại
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X = np.concatenate([X_train, X_test], axis=0) / 255.0
        y = np.concatenate([y_train, y_test], axis=0)
        return X.reshape(-1, 28, 28, 1), y

def preprocess_mnist():
    st.header("Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️")

    # Cho người dùng đặt tên Experiment (chỉ để hiển thị)
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Preprocessing")

    # Tải dữ liệu MNIST từ OpenML (sử dụng cache)
    if 'X_full' not in st.session_state or 'y_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_from_openml()
        st.success("Dữ liệu MNIST đã được tải và chuẩn hóa thành công! ✅")

    # Kiểm tra dữ liệu
    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']
    total_samples = len(X_full)

    # Hiển thị thông tin dữ liệu đầy đủ và hình ảnh
    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ 🔍")
    st.write(f"Tổng số lượng mẫu: {total_samples}")
    st.write("Hình ảnh mẫu (5 mẫu đầu tiên):")
    for i in range(min(5, total_samples)):  # Hiển thị tối đa 5 mẫu đầu tiên
        st.image(X_full[i].reshape(28, 28), caption=f"Chữ số: {y_full[i]}", width=100)

if __name__ == "__main__":
    preprocess_mnist()