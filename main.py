import streamlit as st
import numpy as np

# Bài 1
from exercises.exercise_1.preprocess import preprocess_data
from exercises.exercise_1.train import train_model
from exercises.exercise_1.demo import show_demo
# Bài 2
from exercises.exercise_2.introduce import preprocess_mnist as preprocess_mnist_2
from exercises.exercise_2.train import train_mnist as train_mnist_2
from exercises.exercise_2.demo import mnist_demo as show_mnist_demo_2
from exercises.exercise_2.theory import display_algorithm_info_2
from exercises.exercise_2.view import view_logs 
# Bài 3
from exercises.exercise_3.theory import display_algorithm_info_3
from exercises.exercise_3.introduce import introduce_mnist as introduce_mnist_3
from exercises.exercise_3.train import train_clustering
from exercises.exercise_3.demo import view_clustering_logs
# Bài 4
from exercises.exercise_4.theory import display_algorithm_info_4
from exercises.exercise_4.introduce import introduce_mnist as introduce_mnist_4
from exercises.exercise_4.train import dimensionality_reduction_app
from exercises.exercise_4.demo import view_logs_app
# Bài 5
from exercises.exercise_5.introduce import introduce_mnist as introduce_mnist_5
# from exercises.exercise_5.theory import display_algorithm_info as display_algorithm_info_5
from exercises.exercise_5.train import train_mnist as train_mnist_5
from exercises.exercise_5.demo import demo_mnist_5
from exercises.exercise_5.view import view_log_5
from tensorflow.keras.datasets import mnist

# Map bài tập sang tiêu đề
exercise_titles = {
    'Exercise 1: Titanic Survival Prediction': "Titanic Survival Prediction",
    'Exercise 2: MNIST Handwritten Digit Recognition': "MNIST Handwritten Digit Recognition",
    'Exercise 3: Clustering Algorithms (K-Means & DBSCAN)': "Clustering Algorithms (K-Means & DBSCAN)",
    'Exercise 4: Dimensionality Reduction (PCA & t-SNE)': "Dimensionality Reduction (PCA & t-SNE)",
    'Exercise 5: Neural Network Optimization': "Neural Network Optimization"
}

# Đặt tiêu đề trang ngay từ đầu (Bài 1 mặc định)
default_exercise = 'Exercise 1: Titanic Survival Prediction'
st.set_page_config(page_title=exercise_titles[default_exercise], layout='wide', initial_sidebar_state='collapsed')

# Khởi tạo session_state nếu chưa có
if 'selected_exercise' not in st.session_state:
    st.session_state['selected_exercise'] = default_exercise

# Hàm tải dữ liệu MNIST với lazy loading
@st.cache_data(ttl=86400)  # Cache 24 giờ
def load_mnist_data():
    with st.spinner('Đang tải dữ liệu MNIST...'):
        try:
            (X_train, y_train), (X_test, y_test) = mnist.load_data()
            X_full = np.concatenate([X_train, X_test], axis=0).astype('float32') / 255.0
            y_full = np.concatenate([y_train, y_test], axis=0).astype(np.int32)
            return X_full, y_full
        except Exception as e:
            st.error(f'Lỗi tải dữ liệu MNIST: {str(e)}')
            return None, None

# Lấy tiêu đề hiện tại từ session_state
selected_exercise = st.session_state['selected_exercise']
st.title(exercise_titles[selected_exercise])

# Sidebar chọn bài tập
exercise_options = list(exercise_titles.keys())
exercise = st.sidebar.selectbox('Chọn một Bài tập', exercise_options, key='exercise_selector',
                                index=exercise_options.index(selected_exercise))

# Cập nhật session_state khi người dùng chọn bài tập mới
if exercise != selected_exercise:
    st.session_state['selected_exercise'] = exercise
    st.rerun()  # Sử dụng st.rerun() thay vì experimental_rerun

# Hàm hiển thị cho từng bài với lazy loading
def display_exercise_1():
    tab1, tab2, tab3 = st.tabs(['Tiền Xử Lý Dữ Liệu', 'Huấn Luyện Mô Hình', 'Thử Nghiệm'])
    with tab1:
        preprocess_data()
    with tab2:
        train_model()
    with tab3:
        show_demo()

def display_exercise_2():
    with st.spinner('Đang chuẩn bị dữ liệu cho Bài 2...'):
        X_full, y_full = load_mnist_data()
    if X_full is None or y_full is None:
        st.error('Dữ liệu MNIST không khả dụng. Vui lòng kiểm tra kết nối.')
        return
    tab1, tab2, tab3, tab4 ,tab5= st.tabs(['Giới Thiệu Dữ Liệu', 'Thông Tin', 'Huấn Luyện Mô Hình', 'Thử Nghiệm', 'Xem Logs'])
    with tab1:
        preprocess_mnist_2(X_full, y_full)
    with tab2:
        display_algorithm_info_2()
    with tab3:
        train_mnist_2(X_full, y_full)
    with tab4:
        show_mnist_demo_2()
    with tab5:
        view_logs()

def display_exercise_3():
    with st.spinner('Đang chuẩn bị dữ liệu cho Bài 3...'):
        X_full, y_full = load_mnist_data()
    if X_full is None or y_full is None:
        st.error('Dữ liệu MNIST không khả dụng. Vui lòng kiểm tra kết nối.')
        return
    tab1, tab2, tab3, tab4 = st.tabs(['Giới Thiệu Dữ Liệu', 'Thông Tin', 'Huấn Luyện Mô Hình', 'Xem Logs'])
    with tab1:
        introduce_mnist_3(X_full, y_full)
    with tab2:
        display_algorithm_info_3()
    with tab3:
        train_clustering(X_full, y_full)
    with tab4:
        view_clustering_logs()

def display_exercise_4():
    with st.spinner('Đang chuẩn bị dữ liệu cho Bài 4...'):
        X_full, y_full = load_mnist_data()
    if X_full is None or y_full is None:
        st.error('Dữ liệu MNIST không khả dụng. Vui lòng kiểm tra kết nối.')
        return
    tab1, tab2, tab3, tab4 = st.tabs(['Giới Thiệu Dữ Liệu', "Thông Tin", 'Huấn Luyện Mô Hình', 'Xem Logs'])
    with tab1:
        introduce_mnist_4(X_full, y_full)
    with tab2:
        display_algorithm_info_4()
    with tab3:
        dimensionality_reduction_app(X_full, y_full)
    with tab4:
        view_logs_app()

def display_exercise_5():
    with st.spinner('Đang chuẩn bị dữ liệu cho Bài 5...'):
        X_full, y_full = load_mnist_data()
    if X_full is None or y_full is None:
        st.error('Dữ liệu MNIST không khả dụng. Vui lòng kiểm tra kết nối.')
        return
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['Giới Thiệu Dữ Liệu', 'Thông Tin', 'Huấn Luyện Mô Hình','Thử Nghiệm', 'Xem Logs'])
    with tab1:
        introduce_mnist_5(X_full, y_full)
    # with tab2:
    #     display_algorithm_info_5()
    with tab3:
        train_mnist_5(X_full, y_full)
    with tab4:
        demo_mnist_5()
    with tab5:
        view_log_5()

# Logic hiển thị chính
def display_exercise():
    if selected_exercise == 'Exercise 1: Titanic Survival Prediction':
        display_exercise_1()
    elif selected_exercise == 'Exercise 2: MNIST Handwritten Digit Recognition':
        display_exercise_2()
    elif selected_exercise == 'Exercise 3: Clustering Algorithms (K-Means & DBSCAN)':
        display_exercise_3()
    elif selected_exercise == 'Exercise 4: Dimensionality Reduction (PCA & t-SNE)':
        display_exercise_4()
    elif selected_exercise == 'Exercise 5: Neural Network Optimization':
        display_exercise_5()

if __name__ == '__main__':
    display_exercise()