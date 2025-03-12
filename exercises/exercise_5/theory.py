import streamlit as st

def display_algorithm_info():
    st.title("🧠 Neural Network: Học Máy Đơn Giản Như Chơi!")
    st.markdown("""
    Neural Network (Mạng Nơ-ron) là cách máy tính học giống như bộ não chúng ta.  
    Nó đơn giản nhưng mạnh mẽ, và hôm nay bạn sẽ hiểu cách nó hoạt động để tự tay huấn luyện một mô hình!
    """)

    # Phần 1: Neural Network là gì?
    st.header("1. Neural Network Là Gì?")
    st.markdown("""
    Hãy nghĩ Neural Network như một "cỗ máy đoán số". Nó có 3 phần chính:
    - **Lớp đầu vào**: Nhận dữ liệu (như ảnh số viết tay).
    - **Lớp ẩn**: Tìm đặc điểm quan trọng (như nét cong, nét thẳng).
    - **Lớp đầu ra**: Đưa ra kết quả (số đó là 0 hay 9?).

    Dữ liệu chạy qua các lớp này, được xử lý bởi các "nút" gọi là nơ-ron.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png", 
             caption="Neural Network: dữ liệu vào, xử lý, rồi ra kết quả.", width=300)

    # Phần 2: Công thức cơ bản
    st.header("2. Công Thức Cơ Bản: Nó Làm Việc Như Thế Nào?")
    st.markdown("""
    Mỗi nơ-ron trong mạng làm 2 bước:
    1. **Tính tổng**: Kết hợp dữ liệu với "sức mạnh" của nó (trọng số) và thêm "năng lượng" (bias).
    2. **Kích hoạt**: Quyết định xem nơ-ron có bật lên hay không.

    Đây là công thức:
    """)
    st.latex(r"z = w_1 \cdot x_1 + w_2 \cdot x_2 + b")
    st.latex(r"a = \text{ReLU}(z) = \max(0, z)")
    st.markdown("""
    - \(x_1, x_2\): Dữ liệu đầu vào (ví dụ: pixel ảnh).
    - \(w_1, w_2\): Trọng số (độ quan trọng của dữ liệu).
    - \(b\): Bias (điều chỉnh nhẹ).
    - \(\text{ReLU}\): Hàm kích hoạt, giữ số dương, biến số âm thành 0.

    Ở lớp cuối, ta dùng **Softmax** để biến kết quả thành xác suất:
    """)
    st.latex(r"\hat{y}_i = \frac{e^{z_i}}{\sum e^{z_j}}")

    # Phần 3: Đo sai sót
    st.header("3. Đo Sai Sót: Hàm Mất Mát")
    st.markdown("""
    Để biết mô hình đoán đúng hay sai, ta dùng **hàm mất mát** so sánh dự đoán (\(\hat{y}\)) với đáp án thật (\(y\)):
    """)
    st.latex(r"J = -\frac{1}{m} \sum [y \log(\hat{y}) + (1 - y) \log(1 - \hat{y})]")
    st.markdown("""
    - \(m\): Số mẫu dữ liệu.
    - \(y\): Đáp án thật (0 hoặc 1).
    - \(\hat{y}\): Dự đoán (xác suất).
    - **Mục tiêu**: Giảm \(J\) xuống nhỏ nhất!
    """)

    # Phần 4: Học từ sai sót
    st.header("4. Học Từ Sai Sót: Điều Chỉnh Trọng Số")
    st.markdown("""
    Khi đoán sai, ta điều chỉnh \(w\) và \(b\) để lần sau tốt hơn, dùng công thức:
    """)
    st.latex(r"w = w - \eta \cdot \frac{\partial J}{\partial w}")
    st.latex(r"b = b - \eta \cdot \frac{\partial J}{\partial b}")
    st.markdown("""
    - \(\eta\): Tốc độ học (learning rate), bước thay đổi mỗi lần (thường nhỏ như 0.001).
    - \(\frac{\partial J}{\partial w}\): Hướng thay đổi để giảm sai sót.
    """)

    # Phần 5: Các bước huấn luyện
    st.header("5. Các Bước Huấn Luyện: Làm Từng Bước")
    st.markdown("""
    Đây là cách huấn luyện Neural Network:
    1. **Bắt đầu**: Đặt \(w\) và \(b\) ngẫu nhiên (như 0.1, 0.2).
    2. **Đoán**: Đưa dữ liệu qua mạng, tính \(\hat{y}\).
    3. **Tính sai sót**: So sánh \(\hat{y}\) với \(y\) bằng hàm mất mát \(J\).
    4. **Sửa lỗi**: Điều chỉnh \(w\) và \(b\) dựa trên \(\eta\).
    5. **Lặp lại**: Làm nhiều lần (epochs) với từng mẫu dữ liệu.

    **Tham số quan trọng khi huấn luyện**:
    - **Số lớp ẩn**: Quyết định mạng sâu bao nhiêu (thường 1-3).
    - **Số nơ-ron**: Số "nhân viên" xử lý trong mỗi lớp (64, 128...).
    - **Epochs**: Số lần lặp toàn bộ dữ liệu (5, 10...).
    - **Batch size**: Số mẫu xử lý cùng lúc (32, 64...).
    - **Learning rate (\(\eta\))**: Tốc độ thay đổi \(w\) và \(b\).
    """)

    # Phần 6: Liên kết với train
    st.header("6. Sẵn Sàng Huấn Luyện Mô Hình!")
    st.markdown("""
    Bây giờ bạn đã hiểu:
    - Dữ liệu đi qua mạng với \(w\) và \(b\).
    - Sai sót được đo bằng \(J\).
    - Học bằng cách sửa \(w\), \(b\) với \(\eta\).

    Khi huấn luyện thật (như trong file `train.py`):
    1. Chọn số mẫu (bao nhiêu ảnh MNIST?).
    2. Đặt các tham số: lớp ẩn, nơ-ron, epochs, batch size, learning rate.
    3. Chạy mô hình và xem độ chính xác tăng lên qua từng epoch!
    
    Hãy thử ngay để thấy Neural Network đoán số viết tay chính xác thế nào!
    """)

if __name__ == "__main__":
    display_algorithm_info()