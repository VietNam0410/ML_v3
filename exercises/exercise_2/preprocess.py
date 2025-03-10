import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Hàm trực quan hóa MNIST với tùy chọn tương tác
def visualize_mnist(X: np.ndarray, y: np.ndarray, num_examples: int = 10) -> None:
    if X is None or y is None:
        st.error('Không có dữ liệu để trực quan hóa.')
        return

    st.subheader('🌟 Ví dụ các chữ số trong MNIST')
    unique_labels = np.unique(y)
    images = []
    for label in unique_labels[:num_examples]:
        try:
            idx = np.nonzero(y == label)[0][0]
            images.append((X[idx], label))
        except IndexError:
            st.error(f'Không tìm thấy dữ liệu cho nhãn {label}.')
            continue

    if not images:
        st.error('Không có hình ảnh nào để hiển thị.')
        return

    cols = st.columns(min(num_examples, 5))
    for i, (image, label) in enumerate(images):
        with cols[i % len(cols)]:
            st.image(image, caption=f'Chữ số: {label}', use_container_width=True, clamp=True)

def preprocess_mnist(X_full, y_full):
    st.header('Tiền xử lý Dữ liệu MNIST Chữ số Viết Tay 🖌️')
    total_samples = len(X_full)

    st.subheader('Thông tin Dữ liệu MNIST Đầy đủ 🔍')
    st.write(f'Tổng số lượng mẫu: {total_samples}')

    num_examples = st.slider(
        'Chọn số lượng ví dụ chữ số để xem (tối đa 10)',
        min_value=1, max_value=10, value=5, key='num_examples_slider'
    )
    visualize_mnist(X_full, y_full, num_examples)

    with st.expander('Thông tin chi tiết về MNIST'):
        st.write('### Giới thiệu về MNIST\n' +
                 'MNIST là tập dữ liệu tiêu chuẩn cho nhận dạng chữ số viết tay.\n\n' +
                 '- Cấu trúc dữ liệu:\n' +
                 '  - 60.000 ảnh huấn luyện\n' +
                 '  - 10.000 ảnh kiểm tra\n' +
                 '  - Ảnh grayscale 28x28 pixel.\n\n' +
                 '- Chuẩn hóa: Pixel từ [0, 255] về [0, 1].')

    st.header('📈 Phân phối các nhãn trong MNIST')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(y_full, return_counts=True)
    ax.bar(unique, counts, tick_label=[str(i) for i in unique], color='skyblue', edgecolor='black')
    ax.set_title('Phân phối các chữ số', fontsize=12, pad=15)
    ax.set_xlabel('Chữ số', fontsize=10)
    ax.set_ylabel('Số lượng', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig, use_container_width=True)