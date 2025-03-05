import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional

# Tối ưu cache dữ liệu với TTL (time-to-live) để làm mới dữ liệu sau 24 giờ
@st.cache_data(ttl=86400)  # Dữ liệu sẽ được làm mới sau 24 giờ
def load_mnist_from_pkl(data_dir: str = "./exercises/exercise_3/data/") -> Tuple[np.ndarray, np.ndarray]:
    """
    Tải dữ liệu MNIST từ các file .pkl trong thư mục exercises/exercise_3/data/.
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
            max_samples = st.session_state.get("max_samples", 70000)
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

# Hàm trực quan hóa MNIST với tùy chọn tương tác
def visualize_mnist(X: np.ndarray, y: np.ndarray, num_examples: int = 10) -> None:
    if X is None or y is None:
        st.error("Không có dữ liệu để trực quan hóa. Vui lòng kiểm tra dữ liệu.")
        return

    st.subheader("🌟 Ví dụ các chữ số trong MNIST")
    unique_labels = np.unique(y)
    images = []

    # Lấy một ảnh cho mỗi nhãn từ 0 đến 9 (hoặc ít hơn nếu num_examples nhỏ)
    for label in unique_labels[:num_examples]:
        idx = np.nonzero(y == label)[0][0]  # Lấy index đầu tiên của label
        images.append((X[idx].reshape(28, 28), label))

    # Tạo layout trực quan hơn với grid động
    cols = st.columns(min(num_examples, 5))  # Hiển thị tối đa 5 cột trên mỗi hàng
    for i, (image, label) in enumerate(images):
        with cols[i % len(cols)]:
            st.image(image, caption=f"Chữ số: {label}", use_container_width=True, clamp=True)

# Hàm giới thiệu tập dữ liệu MNIST
def introduce_mnist():
    # Thanh tiến trình để theo dõi tải dữ liệu
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Tải dữ liệu với thông báo trạng thái
    status_text.write("Bắt đầu tải dữ liệu MNIST từ file .pkl...")
    X, y = load_mnist()
    if X is None or y is None:
        st.error("Không thể tải dữ liệu MNIST. Ứng dụng sẽ dừng lại.")
        st.stop()

    progress_bar.progress(100)
    status_text.write("Dữ liệu MNIST đã sẵn sàng! ✅")

    # Chia layout thành 2 cột lớn để hiển thị trực quan
    col1, col2 = st.columns([1, 1])

    with col1:
        # Tương tác: Cho phép người dùng chọn số lượng ví dụ muốn xem
        num_examples = st.slider(
            "Chọn số lượng ví dụ chữ số để xem (tối đa 10)",
            min_value=1, max_value=10, value=5, key="num_examples_slider"
        )
        visualize_mnist(X, y, num_examples)

        # Thông tin cơ bản về dữ liệu
        st.subheader("📊 Thông tin cơ bản về MNIST")
        st.write(f"**Tổng số mẫu**: {X.shape[0]}")
        st.write(f"**Kích thước mỗi ảnh**: {X.shape[1]}x{X.shape[2]} pixel (28x28 pixel, grayscale)")
        st.write(f"**Số lớp (nhãn)**: 10 (chữ số từ 0 đến 9)")

    with col2:
        st.markdown("""
            ### 📚 Giới thiệu về MNIST
            **MNIST (Modified National Institute of Standards and Technology)** là một trong những tập dữ liệu nổi tiếng nhất trong lĩnh vực nhận dạng chữ số viết tay. Đây là tập dữ liệu tiêu chuẩn để huấn luyện và đánh giá các mô hình machine learning (ML) và deep learning (DL), đặc biệt là nhận dạng hình ảnh.

            - **Cấu trúc dữ liệu**:
                - 60.000 ảnh dùng để huấn luyện (training set)
                - 10.000 ảnh dùng để đánh giá (test set)
                - Mỗi ảnh là ảnh grayscale (đen trắng, 1 kênh màu) với kích thước 28x28 pixel.

            - **Chuẩn hóa dữ liệu**:
                Giá trị pixel ban đầu nằm trong khoảng **[0, 255]**. Chúng tôi đã chuẩn hóa dữ liệu, chia cho 255.0 để đưa về khoảng **[0, 1]** để phù hợp với các mô hình học máy.

            - **Ứng dụng**:
                MNIST thường được sử dụng để thử nghiệm các thuật toán nhận dạng chữ số, từ các mô hình đơn giản như SVM, Decision Tree đến các mô hình phức tạp như Convolutional Neural Networks (CNN).
        """)

    # Trực quan hóa phân phối nhãn với biểu đồ đẹp hơn
    st.header("📈 Phân phối các nhãn trong MNIST")
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar(unique, counts, tick_label=[str(i) for i in unique], color='skyblue', edgecolor='black')
    ax.set_title("Phân phối các chữ số trong tập dữ liệu MNIST", fontsize=12, pad=15)
    ax.set_xlabel("Chữ số", fontsize=10)
    ax.set_ylabel("Số lượng", fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig, use_container_width=True)

    # Thêm nút "Tải lại dữ liệu" để làm mới nếu cần
    if st.button("🔄 Tải lại dữ liệu", key="refresh_data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    introduce_mnist()