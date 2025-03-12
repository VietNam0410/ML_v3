import streamlit as st

def display_algorithm_info_4():
    st.subheader('Thông tin về Thuật toán Giảm Chiều')
    model_choice_info = st.selectbox('Chọn thuật toán để xem thông tin', ['PCA', 't-SNE'], key='info_model_choice_ex4')

    if model_choice_info == 'PCA':
        st.write("### 1. Thông tin mô hình")
        st.write("Principal Component Analysis (PCA) là một thuật toán giảm chiều tuyến tính, sử dụng để chuyển đổi dữ liệu sang một không gian có số chiều thấp hơn bằng cách giữ lại các thành phần chính (principal components) giải thích phần lớn phương sai trong dữ liệu.")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **n_components**: Số thành phần chính cần giữ lại.")
        st.write("  - Giá trị nhỏ: Giảm chiều mạnh hơn, có thể mất thông tin.")
        st.write("  - Giá trị lớn: Giữ nhiều thông tin hơn, nhưng hiệu quả giảm chiều thấp.")
        st.write("- **whiten**: Nếu bật (`True`), chuẩn hóa các thành phần để có phương sai bằng 1.")
        st.write("- **random_state**: Đặt giá trị cố định để đảm bảo kết quả tái lập được.")

        st.write("### 3. Các bước thực hiện")
        st.write("1. **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu (ví dụ: dùng StandardScaler) để các đặc trưng có trung bình 0 và phương sai 1.")
        st.write("2. **Tính toán ma trận hiệp phương sai**: Xác định mối quan hệ giữa các đặc trưng.")
        st.write("3. **Tính giá trị riêng và vectơ riêng**: Tìm các thành phần chính dựa trên phương sai lớn nhất.")
        st.write("4. **Giảm chiều**: Chọn n_components thành phần chính và ánh xạ dữ liệu sang không gian mới.")
        st.write("5. **Đánh giá**: Kiểm tra tỷ lệ phương sai được giải thích để đánh giá chất lượng giảm chiều.")

        st.write("### 4. Ưu điểm")
        st.write("- Đơn giản và nhanh chóng với dữ liệu lớn.")
        st.write("- Giữ được phần lớn thông tin quan trọng thông qua phương sai.")
        st.write("- Hiệu quả với dữ liệu có tương quan tuyến tính.")

        st.write("### 5. Nhược điểm")
        st.write("- Chỉ hoạt động tốt với dữ liệu tuyến tính.")
        st.write("- Có thể mất thông tin phi tuyến tính quan trọng.")
        st.write("- Nhạy cảm với thang đo của dữ liệu nếu không chuẩn hóa.")
    else:  # t-SNE
        st.write("### 1. Thông tin mô hình")
        st.write("t-Distributed Stochastic Neighbor Embedding (t-SNE) là một thuật toán giảm chiều phi tuyến tính, chủ yếu được sử dụng để trực quan hóa dữ liệu cao chiều bằng cách bảo toàn mối quan hệ cục bộ giữa các điểm dữ liệu trong không gian 2D hoặc 3D.")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **n_components**: Số chiều đầu ra (thường là 2 hoặc 3).")
        st.write("- **perplexity**: Đo lường số lượng láng giềng gần nhất, ảnh hưởng đến cân bằng giữa cục bộ và toàn cục.")
        st.write("  - Giá trị nhỏ: Tập trung vào cấu trúc cục bộ.")
        st.write("  - Giá trị lớn: Tập trung vào cấu trúc toàn cục.")
        st.write("- **learning_rate**: Tốc độ học, kiểm soát bước cập nhật trong quá trình tối ưu hóa.")
        st.write("- **random_state**: Đặt giá trị cố định để đảm bảo kết quả tái lập được.")

        st.write("### 3. Các bước thực hiện")
        st.write("1. **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu để các đặc trưng có cùng thang đo.")
        st.write("2. **Tính khoảng cách**: Xác định khoảng cách giữa các điểm trong không gian cao chiều.")
        st.write("3. **Chuyển đổi xác suất**: Sử dụng phân phối t để ánh xạ khoảng cách sang xác suất tương đồng.")
        st.write("4. **Tối ưu hóa**: Điều chỉnh vị trí điểm trong không gian thấp chiều để giảm thiểu sự khác biệt giữa phân phối gốc và mới.")
        st.write("5. **Đánh giá**: Kiểm tra trực quan để đánh giá chất lượng giảm chiều.")

        st.write("### 4. Ưu điểm")
        st.write("- Hiệu quả với dữ liệu phi tuyến tính và cấu trúc cục bộ.")
        st.write("- Phù hợp để trực quan hóa dữ liệu phức tạp (như MNIST).")
        st.write("- Bảo toàn tốt các cụm trong dữ liệu.")

        st.write("### 5. Nhược điểm")
        st.write("- Tốn thời gian với dữ liệu lớn.")
        st.write("- Không bảo toàn khoảng cách toàn cục tốt.")
        st.write("- Kết quả phụ thuộc mạnh vào tham số perplexity và learning_rate.")