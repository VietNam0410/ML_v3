import streamlit as st
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

@st.cache_data
def load_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    return X_train, y_train, X_test, y_test

def preprocess_mnist_3():
    st.header("MNIST Data Preprocessing")
    
    # Load data
    X_train, y_train, X_test, y_test = load_mnist_data()
    
    # Chọn số lượng mẫu
    n_samples = st.slider("Số lượng mẫu", 1000, len(X_train), 10000)
    
    # Lấy mẫu ngẫu nhiên
    idx = np.random.choice(len(X_train), n_samples, replace=False)
    X_selected = X_train[idx].reshape(n_samples, -1)  # Flatten images
    y_selected = y_train[idx]
    
    st.write(f"Đã chọn {n_samples} mẫu từ tập MNIST")
    st.write(f"Kích thước dữ liệu: {X_selected.shape}")
    
    # Hiển thị vài mẫu ví dụ
    n_examples = 5
    fig, axes = plt.subplots(1, n_examples)
    for i in range(n_examples):
        axes[i].imshow(X_train[i], cmap='gray')
        axes[i].axis('off')
    st.pyplot(fig)
    
    return X_selected, y_selected, X_test.reshape(len(X_test), -1), y_test

if __name__ == "__main__":
    preprocess_mnist()