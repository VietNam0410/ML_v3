import streamlit as st

def display_algorithm_info_2():
    st.subheader('Thông tin về Thuật toán')
    model_choice_info = st.selectbox('Chọn thuật toán để xem thông tin', ['SVM', 'Decision Tree'], key='info_model_choice_ex2')

    if model_choice_info == 'SVM':
        st.write("### 1. Thông tin mô hình")
        st.write("Support Vector Machine (SVM) là một thuật toán học máy có giám sát, được sử dụng cho cả bài toán phân loại và hồi quy. Trong bài toán phân loại, SVM tìm kiếm một siêu phẳng (hyperplane) tốt nhất để phân tách các lớp dữ liệu với khoảng cách lớn nhất giữa các điểm dữ liệu gần nhất của các lớp (gọi là margin).")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **Kernel**: Loại hàm kernel để biến đổi dữ liệu.")
        st.write("  - `linear`: Kernel tuyến tính, phù hợp với dữ liệu có thể phân tách tuyến tính.")
        st.write("  - `rbf`: Kernel Gaussian Radial Basis Function, phù hợp với dữ liệu phi tuyến tính.")
        st.write("  - `poly`: Kernel đa thức, phù hợp với các bài toán phức tạp hơn.")
        st.write("- **Probability**: Nếu bật (`True`), mô hình sẽ tính xác suất dự đoán (dùng để lấy độ tin cậy).")
        st.write("- **Random State**: Đặt giá trị cố định để đảm bảo kết quả tái lập được.")

        st.write("### 3. Các bước huấn luyện")
        st.write("1. **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu (ví dụ: dùng StandardScaler) để đảm bảo tất cả các đặc trưng có cùng thang đo.")
        st.write("2. **Chọn kernel**: Quyết định loại kernel dựa trên đặc điểm dữ liệu (tuyến tính hay phi tuyến tính).")
        st.write("3. **Huấn luyện mô hình**: Sử dụng tập huấn luyện để tìm siêu phẳng tối ưu.")
        st.write("4. **Đánh giá**: Kiểm tra độ chính xác trên tập validation và tập kiểm tra.")
        st.write("5. **Tối ưu hóa**: Điều chỉnh tham số (như kernel hoặc C nếu cần) để cải thiện hiệu suất.")

        st.write("### 4. Ưu điểm")
        st.write("- Hiệu quả với dữ liệu có số chiều lớn (như MNIST).")
        st.write("- Có thể xử lý dữ liệu phi tuyến tính thông qua kernel.")
        st.write("- Ít bị ảnh hưởng bởi nhiễu nếu được điều chỉnh đúng.")

        st.write("### 5. Nhược điểm")
        st.write("- Tốn thời gian huấn luyện với dữ liệu lớn (như 70,000 mẫu).")
        st.write("- Nhạy cảm với tham số (cần điều chỉnh kernel, C, gamma,...).")
        st.write("- Khó giải thích kết quả (so với Decision Tree).")
    else:  # Decision Tree
        st.write("### 1. Thông tin mô hình")
        st.write("Decision Tree (Cây quyết định) là một thuật toán học máy có giám sát, hoạt động bằng cách chia không gian đặc trưng thành các vùng dựa trên giá trị của các đặc trưng. Mỗi nút trong cây đại diện cho một quyết định dựa trên một đặc trưng, và lá cuối cùng đại diện cho kết quả dự đoán.")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **Max Depth**: Độ sâu tối đa của cây, kiểm soát độ phức tạp của mô hình.")
        st.write("  - Giá trị nhỏ: Ngăn overfitting nhưng có thể underfit.")
        st.write("  - Giá trị lớn: Mô hình phức tạp hơn, dễ overfitting.")
        st.write("- **Random State**: Đặt giá trị cố định để đảm bảo kết quả tái lập được.")

        st.write("### 3. Các bước huấn luyện")
        st.write("1. **Tiền xử lý dữ liệu**: Không cần chuẩn hóa (Decision Tree không nhạy cảm với thang đo của đặc trưng).")
        st.write("2. **Chọn độ sâu tối đa**: Quyết định độ sâu dựa trên độ phức tạp của dữ liệu.")
        st.write("3. **Huấn luyện mô hình**: Xây dựng cây quyết định dựa trên tập huấn luyện.")
        st.write("4. **Đánh giá**: Kiểm tra độ chính xác trên tập validation và tập kiểm tra.")
        st.write("5. **Tối ưu hóa**: Điều chỉnh max_depth để cân bằng giữa underfitting và overfitting.")

        st.write("### 4. Ưu điểm")
        st.write("- Dễ hiểu và giải thích (cây quyết định có thể được trực quan hóa).")
        st.write("- Không cần chuẩn hóa dữ liệu.")
        st.write("- Nhanh hơn SVM với dữ liệu lớn.")

        st.write("### 5. Nhược điểm")
        st.write("- Dễ bị overfitting nếu độ sâu quá lớn.")
        st.write("- Không hiệu quả bằng SVM với dữ liệu phi tuyến tính hoặc phức tạp.")
        st.write("- Nhạy cảm với nhiễu trong dữ liệu.")