import streamlit as st

def display_neural_network_theory():
    st.title("🧠 Lý Thuyết Neural Network: Hiểu và Huấn Luyện Dễ Dàng!")
    st.markdown("""
    Neural Network (Mạng Nơ-ron Nhân tạo) là một công cụ mạnh mẽ trong học máy, được lấy cảm hứng từ cách hoạt động của não bộ con người.  
    Trong phần này, bạn sẽ hiểu cách Neural Network hoạt động, các bước huấn luyện, và các tham số quan trọng để áp dụng cho bài toán phân loại chữ số viết tay (như MNIST).
    """)

    # Phần 1: Neural Network là gì?
    st.header("1. Neural Network Là Gì?")
    st.markdown("""
    Neural Network là một "cỗ máy đoán số" thông minh, bao gồm các **nơ-ron** (neurons) được tổ chức thành các **lớp** (layers) kết nối với nhau.  
    Mỗi nơ-ron nhận dữ liệu đầu vào, xử lý thông tin, và truyền kết quả đến lớp tiếp theo.  

    Trong bài toán phân loại chữ số viết tay (như tập dữ liệu **MNIST**), Neural Network giúp máy tính nhận diện các chữ số (0-9) từ hình ảnh.  
    Một Neural Network cơ bản có 3 phần chính:
    - **Lớp đầu vào (Input Layer)**: Nhận dữ liệu (ví dụ: ảnh 28x28 pixel của MNIST, làm phẳng thành vector 784 chiều).
    - **Lớp ẩn (Hidden Layers)**: Xử lý dữ liệu, tìm các đặc trưng quan trọng (như nét cong, nét thẳng của chữ số).
    - **Lớp đầu ra (Output Layer)**: Đưa ra kết quả (xác suất cho từng chữ số 0-9).
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Colored_neural_network.svg/300px-Colored_neural_network.svg.png", 
             caption="Cấu trúc Neural Network: dữ liệu vào, xử lý qua các lớp, rồi ra kết quả.", width=300)

    # Phần 2: Cấu trúc cơ bản của Neural Network
    st.header("2. Cấu Trúc Cơ Bản của Neural Network")
    st.markdown("""
    Một Neural Network bao gồm nhiều nơ-ron được kết nối với nhau. Mỗi nơ-ron thực hiện hai bước chính:  
    1. **Tính tổng trọng số (Weighted Sum)**: Kết hợp dữ liệu đầu vào với trọng số \(w\) và thêm bias \(b\):  
    """)
    st.latex(r"z = \sum_{i} (w_i \cdot x_i) + b")
    st.markdown("""
       - \(x_i\): Dữ liệu đầu vào (ví dụ: giá trị pixel của ảnh).  
       - \(w_i\): Trọng số (độ quan trọng của dữ liệu đầu vào).  
       - \(b\): Bias (điều chỉnh nhẹ để mô hình linh hoạt hơn).  

    2. **Áp dụng hàm kích hoạt (Activation Function)**: Quyết định xem nơ-ron có "bật" hay không:  
    """)
    st.latex(r"a = \text{activation}(z)")
    st.markdown("""
       - Hàm kích hoạt phổ biến:  
         - `relu`:  
    """)
    st.latex(r"\text{ReLU}(z) = \max(0, z)")
    st.markdown("""
           giữ số dương, biến số âm thành 0.  
         - `sigmoid`:  
    """)
    st.latex(r"\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}")
    st.markdown("""
           đưa đầu ra về khoảng [0, 1].  
         - `tanh`:  
    """)
    st.latex(r"\text{tanh}(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}")
    st.markdown("""
           đưa đầu ra về khoảng [-1, 1].  

    Ở lớp đầu ra, ta dùng hàm **Softmax** để tính xác suất cho từng lớp (ví dụ: xác suất là chữ số 0, 1, ..., 9):  
    """)
    st.latex(r"\hat{y}_i = \frac{e^{z_i}}{\sum_{j} e^{z_j}}")
    st.markdown("""
    - \(\hat{y}_i\): Xác suất dự đoán cho lớp \(i\).
    """)

    # Phần 3: Các bước huấn luyện Neural Network
    st.header("3. Các Bước Huấn Luyện Neural Network")
    st.markdown("""
    Huấn luyện Neural Network là quá trình điều chỉnh các trọng số \(w\) và bias \(b\) để mô hình dự đoán chính xác hơn.  
    Quá trình này được thực hiện qua 5 bước sau:
    """)

    st.subheader("Bước 1: Bắt Đầu - Đặt \(w\) và \(b\) Ngẫu Nhiên")
    st.markdown("""
    - Khi bắt đầu, các trọng số \(w\) và bias \(b\) được khởi tạo ngẫu nhiên với các giá trị nhỏ (ví dụ: 0.1, 0.2).  
    - Điều này giúp mô hình có một điểm xuất phát để học. Nếu khởi tạo tất cả bằng 0, mô hình sẽ không học được gì vì các nơ-ron sẽ hoạt động giống nhau.
    """)

    st.subheader("Bước 2: Đoán - Đưa Dữ Liệu Qua Mạng, Tính \(\hat{y}\)")
    st.markdown("""
    - Dữ liệu (ví dụ: ảnh MNIST) được đưa vào lớp đầu vào.  
    - Dữ liệu đi qua các lớp ẩn, mỗi nơ-ron tính tổng trọng số \(z = \sum (w_i \cdot x_i) + b\), áp dụng hàm kích hoạt, và truyền kết quả đến lớp tiếp theo.  
    - Ở lớp đầu ra, hàm Softmax được sử dụng để tính xác suất \(\hat{y}\) cho từng lớp (ví dụ: xác suất là chữ số 0, 1, ..., 9).
    """)

    st.subheader("Bước 3: Tính Sai Sót - So Sánh \(\hat{y}\) với \(y\) Bằng Hàm Mất Mát \(J\)")
    st.markdown("""
    - Để biết mô hình đoán đúng hay sai, ta so sánh dự đoán \(\hat{y}\) với đáp án thật \(y\) bằng **hàm mất mát** (loss function).  
    - Trong bài toán phân loại nhiều lớp (như MNIST), ta dùng hàm mất mát **categorical crossentropy**:  
    """)
    st.latex(r"J = -\frac{1}{m} \sum_{i=1}^{m} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})")
    st.markdown("""
      - \(m\): Số mẫu dữ liệu.  
      - \(C\): Số lớp (10 lớp cho MNIST, từ 0 đến 9).  
      - \(y_{i,c}\): Đáp án thật (1 nếu mẫu \(i\) thuộc lớp \(c\), 0 nếu không).  
      - \(\hat{y}_{i,c}\): Xác suất dự đoán cho lớp \(c\) của mẫu \(i\).  
      - **Mục tiêu**: Giảm \(J\) xuống nhỏ nhất, nghĩa là dự đoán càng gần đáp án thật càng tốt.
    """)

    st.subheader("Bước 4: Sửa Lỗi - Điều Chỉnh \(w\) và \(b\) Dựa trên \(\eta\)")
    st.markdown("""
    - Khi mô hình đoán sai, ta điều chỉnh \(w\) và \(b\) để giảm sai sót \(J\). Điều này được thực hiện bằng thuật toán **Gradient Descent**:  
    """)
    st.latex(r"w = w - \eta \cdot \frac{\partial J}{\partial w}")
    st.latex(r"b = b - \eta \cdot \frac{\partial J}{\partial b}")
    st.markdown("""
      - \(\eta\): Tốc độ học (learning rate), quyết định bước thay đổi mỗi lần (thường nhỏ, ví dụ: 0.001).  
      - \(\frac{\partial J}{\partial w}\), \(\frac{\partial J}{\partial b}\): Đạo hàm của hàm mất mát, chỉ ra hướng thay đổi để giảm sai sót.  
    - Quá trình này được gọi là **backpropagation** (lan truyền ngược), trong đó sai sót được truyền ngược từ lớp đầu ra về lớp đầu vào để điều chỉnh các tham số.
    """)

    st.subheader("Bước 5: Lặp Lại - Làm Nhiều Lần (Epochs) Với Từng Mẫu Dữ Liệu")
    st.markdown("""
    - Các bước trên (đoán, tính sai sót, sửa lỗi) được lặp lại nhiều lần, gọi là **epochs**.  
    - Mỗi epoch, mô hình xử lý toàn bộ dữ liệu huấn luyện, chia thành các **batch** (nhóm mẫu dữ liệu) để cập nhật \(w\) và \(b\) dần dần.  
    - Việc lặp lại nhiều lần giúp mô hình học tốt hơn, nhưng nếu lặp quá nhiều có thể dẫn đến **quá khớp** (overfitting), tức là mô hình học quá tốt trên dữ liệu huấn luyện nhưng không tốt trên dữ liệu mới.
    """)

    # Phần 4: Tham số quan trọng khi huấn luyện
    st.header("4. Tham Số Quan Trọng Khi Huấn Luyện")
    st.markdown("""
    Khi huấn luyện Neural Network, có một số tham số quan trọng ảnh hưởng đến hiệu suất và tốc độ học của mô hình:
    """)

    st.subheader("Số Lớp Ẩn")
    st.markdown("""
    - Quyết định mạng sâu bao nhiêu (thường 1-3 lớp).  
    - Nhiều lớp ẩn hơn giúp mô hình học được các đặc trưng phức tạp hơn (như nét cong phức tạp của chữ số), nhưng cũng làm tăng nguy cơ quá khớp và thời gian huấn luyện.
    """)

    st.subheader("Số Nơ-ron")
    st.markdown("""
    - Số "nhân viên" xử lý trong mỗi lớp (thường 64, 128, 256, ...).  
    - Nhiều nơ-ron hơn giúp mô hình biểu diễn được nhiều đặc trưng hơn, nhưng cũng làm tăng độ phức tạp và thời gian huấn luyện.
    """)

    st.subheader("Epochs")
    st.markdown("""
    - Số lần lặp qua toàn bộ dữ liệu huấn luyện (thường 5, 10, ...).  
    - Nhiều epochs hơn giúp mô hình học tốt hơn, nhưng nếu quá nhiều có thể dẫn đến quá khớp.
    """)

    st.subheader("Batch Size")
    st.markdown("""
    - Số mẫu xử lý cùng lúc trong mỗi lần cập nhật \(w\) và \(b\) (thường 32, 64, ...).  
    - Batch size nhỏ (như 32) giúp cập nhật trọng số thường xuyên hơn, nhưng có thể làm tăng thời gian huấn luyện.  
    - Batch size lớn (như 128) giúp huấn luyện nhanh hơn nhưng có thể làm giảm độ chính xác.
    """)

    st.subheader("Learning Rate")
    st.markdown("""
    - Tốc độ thay đổi \(w\) và \(b\), ký hiệu là \(\eta\) (thường nhỏ, ví dụ: 0.001).  
    - \(\eta\) nhỏ giúp mô hình học ổn định hơn, nhưng có thể chậm.  
    - \(\eta\) lớn giúp học nhanh hơn nhưng có thể bỏ qua điểm tối ưu.
    """)

    # Phần 5: Ứng dụng
    st.header("5. Ứng Dụng: Phân Loại Chữ Số Viết Tay (MNIST)")
    st.markdown("""
    Neural Network được sử dụng trong bài toán này để phân loại chữ số viết tay từ tập dữ liệu **MNIST**:  
    - **Dữ liệu**: MNIST chứa 60,000 ảnh huấn luyện và 10,000 ảnh kiểm tra, mỗi ảnh là một chữ số viết tay (0-9) kích thước 28x28 pixel.  
    - **Mục tiêu**: Dự đoán chính xác chữ số trong ảnh (0, 1, ..., 9).  
    - **Ứng dụng thực tế**:  
      - Nhận diện chữ số trong các hệ thống như đọc mã bưu điện, xử lý hóa đơn, hoặc nhận diện chữ viết tay.  
      - Là nền tảng để học các mô hình phức tạp hơn như Convolutional Neural Networks (CNN) cho xử lý ảnh.
    """)

    # Phần 6: Một số lưu ý
    st.header("6. Một Số Lưu Ý")
    st.markdown("""
    - **Quá Khớp (Overfitting)**:  
      - Nếu mô hình học quá tốt trên dữ liệu huấn luyện nhưng không tốt trên dữ liệu mới, đó là hiện tượng quá khớp.  
      - **Cách khắc phục**: Giảm số lớp ẩn, số nơ-ron, hoặc giảm số epochs.  

    - **Thời Gian Huấn Luyện**:  
      - Số lớp ẩn, số nơ-ron, số epochs, và batch size ảnh hưởng đến thời gian huấn luyện.  
      - **Gợi ý**: Bắt đầu với số lớp ẩn ít (1-2), số nơ-ron vừa phải (64-128), và epochs ít (5-10).  

    - **Tốc Độ Học**:  
      - Ký hiệu là \(\eta\).  
      - Nếu \(\eta\) quá nhỏ, mô hình học chậm. Nếu \(\eta\) quá lớn, mô hình có thể không hội tụ (không tìm được điểm tối ưu).  
      - Thường chọn \(\eta\) trong khoảng 0.0001 đến 0.01.
    """)

    # Phần 7: Kết luận
    st.header("7. Kết Luận")
    st.markdown("""
    Neural Network là một công cụ mạnh mẽ để giải quyết các bài toán học máy như phân loại chữ số viết tay.  
    Quá trình huấn luyện bao gồm 5 bước chính:  
    1. Khởi tạo ngẫu nhiên \(w\) và \(b\).  
    2. Dự đoán \(\hat{y}\).  
    3. Tính sai sót \(J\).  
    4. Sửa lỗi bằng cách điều chỉnh \(w\) và \(b\).  
    5. Lặp lại qua nhiều epochs.  

    Các tham số như số lớp ẩn, số nơ-ron, epochs, batch size, và learning rate (ký hiệu \(\eta\)) đóng vai trò quan trọng trong việc điều chỉnh hiệu suất của mô hình.  
    Bằng cách hiểu và điều chỉnh các tham số này, bạn có thể huấn luyện một Neural Network hiệu quả để giải quyết bài toán của mình.  
    Hãy thử nghiệm với các giá trị khác nhau để thấy cách chúng ảnh hưởng đến kết quả!
    """)
