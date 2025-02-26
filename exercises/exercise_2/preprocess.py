import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist  # Backup nếu OpenML không tải được
import pandas as pd

# Sử dụng st.cache_data để cache dữ liệu, tăng tốc độ load
@st.cache_data
def load_mnist_from_openml():
    try:
        # Tải dữ liệu MNIST từ OpenML (ID dataset MNIST là 554)
        dataset = openml.datasets.get_dataset(554)
        X, y, _, _ = dataset.get_data(target='class')
        
        # Chuyển đổi X thành mảng numpy và chuẩn hóa (28x28x1)
        X = X.reshape(-1, 28, 28, 1) / 255.0
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

    # Cho người dùng đặt tên Experiment (giờ chỉ để hiển thị, không dùng MLflow)
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="MNIST_Preprocessing")
    
    # Khởi tạo session_state để lưu dữ liệu
    if 'mnist_data' not in st.session_state:
        st.session_state['mnist_data'] = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = {}

    # Tải dữ liệu MNIST từ OpenML (sử dụng cache)
    if st.button("Tải dữ liệu MNIST từ OpenML"):
        # Sửa đổi bởi Grok 3: Tải trực tiếp từ OpenML, sử dụng st.cache_data
        X_full, y_full = load_mnist_from_openml()
        
        # Lưu dữ liệu đầy đủ vào session
        st.session_state['mnist_data'] = {
            'X_full': X_full,
            'y_full': y_full
        }
        st.session_state['preprocessing_steps'] = {"loaded": True}
        st.success("Dữ liệu MNIST đã được tải từ OpenML và chuẩn hóa thành công! ✅")

    if 'mnist_data' not in st.session_state or st.session_state['mnist_data'] is None:
        st.warning("Vui lòng tải dữ liệu MNIST để tiếp tục. ⚠️")
        return

    # Hiển thị thông tin dữ liệu đầy đủ
    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ 🔍")
    mnist_data = st.session_state['mnist_data']
    st.write(f"Tổng số lượng mẫu: {len(mnist_data['X_full'])}")
    st.write("Hình ảnh mẫu (đầu tiên):")
    st.image(mnist_data['X_full'][0].reshape(28, 28), caption=f"Chữ số: {mnist_data['y_full'][0]}", width=100)

    # Chia tách dữ liệu theo lựa chọn của người dùng
    # Sửa đổi bởi Grok 3: Cho phép người dùng chọn kích thước tập dữ liệu
    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    train_size = st.slider("Chọn kích thước tập huấn luyện (%)", min_value=10, max_value=90, value=70, step=5) / 100
    val_size = st.slider("Chọn kích thước tập validation (%)", min_value=0, max_value=30, value=15, step=5) / 100
    test_size = 1 - train_size - val_size  # Tính kích thước tập kiểm tra

    if test_size < 0:
        st.error("Tổng kích thước tập huấn luyện và validation không được vượt quá 100%. Vui lòng điều chỉnh lại.")
    else:
        if st.button("Chia dữ liệu"):
            X_full = mnist_data['X_full']
            y_full = mnist_data['y_full']

            # Chia dữ liệu thành tập huấn luyện, validation, và kiểm tra
            X_temp, X_test, y_temp, y_test = train_test_split(X_full, y_full, test_size=test_size, random_state=42)
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=val_size/(train_size+val_size), random_state=42)

            # Lưu vào session_state
            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }
            st.session_state['preprocessing_steps']['split'] = {
                "train_size": train_size,
                "val_size": val_size,
                "test_size": test_size
            }
            st.success(f"Đã chia dữ liệu với kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}% và lưu trong session! ✅")

            st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
            st.write(f"Tập validation: {len(X_valid)} mẫu")
            st.write(f"Tập kiểm tra: {len(X_test)} mẫu")
            st.write("Hình ảnh mẫu từ tập huấn luyện:", X_train[0].reshape(28, 28))
            st.write(f"Chữ số thực tế: {y_train[0]}")

    # Lưu dữ liệu trong session (không lưu file)
    st.write("### Lưu dữ liệu đã tiền xử lý trong session 💾")
    if st.button("Lưu dữ liệu đã xử lý trong session 📋"):
        st.session_state['processed_mnist'] = st.session_state['mnist_data'].copy()
        st.success("Dữ liệu đã được lưu trong session! ✅")

        st.subheader("Xem trước dữ liệu đã xử lý trong session 🔚")
        st.write(st.session_state['processed_mnist'])

if __name__ == "__main__":
    preprocess_mnist()