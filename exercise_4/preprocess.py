import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from typing import Tuple, Optional
from common.common import load_mnist, handle_error  # Import load_mnist từ common.py

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
    X, y = load_mnist(data_dir="./exercises/exercise_4/data/")  # Sử dụng load_mnist từ common.py với đường dẫn phù hợp
    if X is None or y is None:
        handle_error("Không thể tải dữ liệu MNIST. Ứng dụng sẽ dừng lại.", "dừng")

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