import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

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
            images.append((X[idx].reshape(28, 28), label))
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

def introduce_mnist(X, y):
    st.subheader('📊 Thông tin cơ bản về MNIST')
    st.write(f'**Tổng số mẫu**: {X.shape[0]}')
    st.write(f'**Kích thước mỗi ảnh**: 28x28 pixel (grayscale)')
    st.write(f'**Số lớp**: 10 (0-9)')

    num_examples = st.slider('Chọn số lượng ví dụ', 1, 10, 5)
    visualize_mnist(X, y, num_examples)

    st.header('📈 Phân phối nhãn')
    fig, ax = plt.subplots(figsize=(8, 4))
    unique, counts = np.unique(y, return_counts=True)
    ax.bar(unique, counts, tick_label=[str(i) for i in unique], color='skyblue', edgecolor='black')
    ax.set_title('Phân phối các chữ số', fontsize=12)
    ax.set_xlabel('Chữ số')
    ax.set_ylabel('Số lượng')
    ax.grid(True, linestyle='--', alpha=0.7)
    st.pyplot(fig)