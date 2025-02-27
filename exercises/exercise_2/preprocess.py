import streamlit as st
import numpy as np
import openml
from sklearn.model_selection import train_test_split
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

    # Hiển thị thông tin dữ liệu đầy đủ (không hiển thị hình ảnh)
    st.subheader("Thông tin Dữ liệu MNIST Đầy đủ 🔍")
    st.write(f"Tổng số lượng mẫu: {total_samples}")

    # Cho phép người dùng chọn số lượng mẫu và tỷ lệ chia
    st.subheader("Chia tách Dữ liệu (Tùy chọn) 🔀")
    max_samples = st.slider("Chọn số lượng mẫu tối đa (0 để dùng toàn bộ)", 0, total_samples, total_samples, step=100)
    
    if max_samples == 0:
        max_samples = total_samples
    elif max_samples > total_samples:
        st.error(f"Số lượng mẫu ({max_samples}) vượt quá tổng số mẫu có sẵn ({total_samples}). Đặt lại về {total_samples}.")
        max_samples = total_samples

    train_size = st.slider("Chọn kích thước tập huấn luyện (%)", min_value=10, max_value=90, value=70, step=5) / 100
    val_size = st.slider("Chọn kích thước tập validation (%)", min_value=0, max_value=30, value=15, step=5) / 100
    test_size = 1 - train_size - val_size  # Tính kích thước tập kiểm tra

    if test_size < 0:
        st.error("Tổng kích thước tập huấn luyện và validation không được vượt quá 100%. Vui lòng điều chỉnh lại.")
    else:
        if st.button("Chia dữ liệu"):
            # Lấy mẫu ngẫu nhiên nếu max_samples < total_samples
            if max_samples < total_samples:
                indices = np.random.choice(total_samples, max_samples, replace=False)
                X_subset = X_full[indices]
                y_subset = y_full[indices]
            else:
                X_subset = X_full
                y_subset = y_full

            # Chia dữ liệu thành tập huấn luyện, validation, và kiểm tra
            X_temp, X_test, y_temp, y_test = train_test_split(X_subset, y_subset, test_size=test_size, random_state=42)
            X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=val_size/(train_size+val_size), random_state=42)

            # Hiển thị kết quả chia dữ liệu
            st.success(f"Đã chia dữ liệu với số lượng mẫu: {max_samples}. Kích thước: Huấn luyện {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Kiểm tra {test_size*100:.1f}%! ✅")

            st.write(f"Tập huấn luyện: {len(X_train)} mẫu")
            st.write(f"Tập validation: {len(X_valid)} mẫu")
            st.write(f"Tập kiểm tra: {len(X_test)} mẫu")

            # Lưu dữ liệu vào session_state để sử dụng trong train.py
            st.session_state['mnist_data'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_valid': X_valid,
                'y_valid': y_valid,
                'X_test': X_test,
                'y_test': y_test
            }

if __name__ == "__main__":
    preprocess_mnist()