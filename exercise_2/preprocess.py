import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from common.common import load_mnist  # Import từ common.py

# Tối ưu cache dữ liệu với TTL để làm mới dữ liệu sau 24 giờ
@st.cache_data(ttl=86400)  # Dữ liệu sẽ được làm mới sau 24 giờ
def load_mnist_data(max_samples: int = 70000) -> Tuple[np.ndarray, np.ndarray]:
    'Tải dữ liệu MNIST từ common/common.py để tránh tải lại.'
    return load_mnist(max_samples=max_samples)

# Hàm trực quan hóa MNIST với tùy chọn tương tác
def visualize_mnist(X: np.ndarray, y: np.ndarray, num_examples: int = 10) -> None:
    if X is None or y is None:
        st.error('Không có dữ liệu để trực quan hóa. Vui lòng kiểm tra dữ liệu.')
        return

    st.subheader('🌟 Ví dụ các chữ số trong MNIST')
    unique_labels = np.unique(y)
    images = []

    # Lấy một ảnh cho mỗi nhãn từ 0 đến 9 (hoặc ít hơn nếu num_examples nhỏ)
    for label in unique_labels[:num_examples]:
        try:
            idx = np.nonzero(y == label)[0][0]  # Lấy index đầu tiên của label
            images.append((X[idx].reshape(28, 28), label))
        except IndexError:
            st.error(f'Không tìm thấy dữ liệu cho nhãn {label}. Bỏ qua nhãn này.')
            continue

    # Tạo layout trực quan hơn với grid động
    if not images:
        st.error('Không có hình ảnh nào để hiển thị. Vui lòng kiểm tra dữ liệu.')
        return

    cols = st.columns(min(num_examples, 5))  # Hiển thị tối đa 5 cột trên mỗi hàng
    for i, (image, label) in enumerate(images):
        with cols[i % len(cols)]:
            st.image(image, caption=f'Chữ số: {label}', use_container_width=True, clamp=True)

# Hàm giới thiệu tập dữ liệu MNIST
def preprocess_mnist():
    st.header('Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️')

    # Tải dữ liệu từ bộ nhớ đệm nếu chưa có trong session_state
    if 'mnist_data' not in st.session_state or 'X_full' not in st.session_state:
        st.session_state['X_full'], st.session_state['y_full'] = load_mnist_data(max_samples=70000)
        st.success('Dữ liệu MNIST đã được tải và chuẩn hóa thành công! ✅')

    X_full = st.session_state['X_full']
    y_full = st.session_state['y_full']
    total_samples = len(X_full)

    st.subheader('Thông tin Dữ liệu MNIST Đầy đủ 🔍')
    st.write(f'Tổng số lượng mẫu: {total_samples}')

    # Trực quan hóa một số ví dụ từ dữ liệu đầy đủ
    num_examples = st.slider(
        'Chọn số lượng ví dụ chữ số để xem (tối đa 10)',
        min_value=1, max_value=10, value=5, key='num_examples_slider'
    )
    visualize_mnist(X_full, y_full, num_examples)

    # Lưu dữ liệu đầy đủ vào session_state để sử dụng trong train
    st.session_state['mnist_data'] = {
        'X_full': X_full,
        'y_full': y_full
    }

    with st.expander('Thông tin chi tiết về MNIST'):
        st.write('### Giới thiệu về MNIST\n' +
                 'MNIST (Modified National Institute of Standards and Technology) là một trong những tập dữ liệu nổi tiếng nhất trong lĩnh vực nhận dạng chữ số viết tay. Đây là tập dữ liệu tiêu chuẩn để huấn luyện và đánh giá các mô hình machine learning (ML) và deep learning (DL), đặc biệt là nhận dạng hình ảnh.\n\n' +
                 '- Cấu trúc dữ liệu:\n' +
                 '  - 60.000 ảnh dùng để huấn luyện (training set)\n' +
                 '  - 10.000 ảnh dùng để đánh giá (test set)\n' +
                 '  - Mỗi ảnh là ảnh grayscale (đen trắng, 1 kênh màu) với kích thước 28x28 pixel.\n\n' +
                 '- Chuẩn hóa dữ liệu:\n' +
                 '  Giá trị pixel ban đầu nằm trong khoảng [0, 255]. Chúng tôi đã chuẩn hóa dữ liệu, chia cho 255.0 để đưa về khoảng [0, 1] để phù hợp với các mô hình học máy.\n\n' +
                 '- Ứng dụng:\n' +
                 '  MNIST thường được sử dụng để thử nghiệm các thuật toán nhận dạng chữ số, từ các mô hình đơn giản như SVM, Decision Tree đến các mô hình phức tạp như Convolutional Neural Networks (CNN).')

    # Trực quan hóa phân phối nhãn với biểu đồ đẹp hơn
    st.header('📈 Phân phối các nhãn trong MNIST')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(y_full, return_counts=True)
    ax.bar(unique, counts, tick_label=[str(i) for i in unique], color='skyblue', edgecolor='black')
    ax.set_title('Phân phối các chữ số trong tập dữ liệu MNIST', fontsize=12, pad=15)
    ax.set_xlabel('Chữ số', fontsize=10)
    ax.set_ylabel('Số lượng', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig, use_container_width=True)

if __name__ == '__main__':
    preprocess_mnist()