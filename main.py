import streamlit as st
from exercises.exercise_1.preprocess import preprocess_data
from exercises.exercise_1.train import train_model
from exercises.exercise_1.demo import show_demo
from exercises.exercise_2.preprocess import preprocess_mnist
from exercises.exercise_2.train import train_mnist
from exercises.exercise_2.demo import show_mnist_demo
from exercises.exercise_3.preprocess import introduce_mnist
from exercises.exercise_3.train import train_clustering
from exercises.exercise_3.demo import view_clustering_logs
from exercises.exercise_4.preprocess import introduce_mnist
from exercises.exercise_4.train import dimensionality_reduction_app
from exercises.exercise_4.demo import view_logs_app
from tensorflow.keras.datasets import mnist
import numpy as np

# Cấu hình trang
st.set_page_config(page_title='Machine Learning Exercises', layout='wide', initial_sidebar_state='collapsed')

# Hàm tải dữ liệu MNIST chung, chỉ chạy một lần
@st.cache_data(ttl=86400)  # Cache 24 giờ
def load_mnist_shared():
    with st.spinner('Đang tải dữ liệu MNIST từ Keras...'):
        try:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_full = np.concatenate([X_train, X_test], axis=0).astype('float32') / 255.0  # Chuẩn hóa [0, 1]
            y_full = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
            st.success('Đã tải dữ liệu MNIST thành công! ✅')
            return X_full, y_full
        except Exception as e:
            st.error(f'Lỗi tải dữ liệu MNIST: {str(e)}')
            return None, None

# Khởi tạo session_state cho dữ liệu và tab
if 'mnist_data' not in st.session_state:
    st.session_state['mnist_data'] = load_mnist_shared()  # Tải toàn bộ 70,000 mẫu
if 'selected_exercise' not in st.session_state:
    st.session_state['selected_exercise'] = 'Exercise 2: MNIST Handwritten Digit Recognition'

st.title('Machine Learning Exercises')

# Sidebar chọn bài tập
exercise_options = [
    'Exercise 1: Titanic Survival Prediction',
    'Exercise 2: MNIST Handwritten Digit Recognition',
    'Exercise 3: Clustering Algorithms (K-Means & DBSCAN)',
    'Exercise 4: Dimensionality Reduction (PCA & t-SNE)'
]
exercise = st.sidebar.selectbox('Chọn một Bài tập', exercise_options, key='exercise_selector',
                                index=exercise_options.index(st.session_state['selected_exercise']),
                                on_change=lambda: st.session_state.update(selected_exercise=st.session_state.exercise_selector))

# Hàm hiển thị cho từng bài
def display_exercise_1():
    tab1, tab2, tab3 = st.tabs(['Preprocess Data', 'Train Model', 'Demo'])
    with tab1:
        preprocess_data()
    with tab2:
        train_model()
    with tab3:
        show_demo()

def display_exercise_2(X_full, y_full):
    tab1, tab2, tab3 = st.tabs(['Preprocess Data', 'Train Model', 'Demo'])
    with tab1:
        preprocess_mnist(X_full, y_full)
    with tab2:
        train_mnist(X_full, y_full)
    with tab3:
        show_mnist_demo()

def display_exercise_3(X_full, y_full):
    tab1, tab2, tab3 = st.tabs(['Preprocess Data', 'Train Model', 'View Logs'])
    with tab1:
        introduce_mnist(X_full, y_full)
    with tab2:
        train_clustering(X_full, y_full)
    with tab3:
        view_clustering_logs()

def display_exercise_4(X_full, y_full):
    tab1, tab2, tab3 = st.tabs(['Preprocess Data', 'Train Model', 'View Logs'])
    with tab1:
        introduce_mnist(X_full, y_full)
    with tab2:
        dimensionality_reduction_app(X_full, y_full)
    with tab3:
        view_logs_app()

# Logic hiển thị chính
def display_exercise():
    X_full, y_full = st.session_state['mnist_data']
    if X_full is None or y_full is None:
        st.error('Dữ liệu MNIST không khả dụng. Vui lòng kiểm tra kết nối hoặc thử lại.')
        return

    if exercise == 'Exercise 1: Titanic Survival Prediction':
        display_exercise_1()
    elif exercise == 'Exercise 2: MNIST Handwritten Digit Recognition':
        display_exercise_2(X_full, y_full)
    elif exercise == 'Exercise 3: Clustering Algorithms (K-Means & DBSCAN)':
        display_exercise_3(X_full, y_full)
    elif exercise == 'Exercise 4: Dimensionality Reduction (PCA & t-SNE)':
        display_exercise_4(X_full, y_full)

if __name__ == '__main__':
    display_exercise()