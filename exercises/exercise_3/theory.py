import streamlit as st

def display_algorithm_info_3():
    st.subheader('Thông tin về Thuật toán Phân Cụm')
    model_choice_info = st.selectbox('Chọn thuật toán để xem thông tin', ['K-means', 'DBSCAN'], key='info_model_choice_ex3')

    if model_choice_info == 'K-means':
        st.write("### 1. Thông tin mô hình")
        st.write("K-means là một thuật toán phân cụm không giám sát, sử dụng để chia tập dữ liệu thành K cụm dựa trên khoảng cách Euclidean. Mỗi điểm dữ liệu được gán vào cụm gần nhất với tâm cụm (centroid), và các tâm cụm được cập nhật lặp lại cho đến khi hội tụ.")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **n_clusters**: Số lượng cụm (K) cần tạo.")
        st.write("  - Giá trị nhỏ: Ít cụm hơn, có thể mất thông tin chi tiết.")
        st.write("  - Giá trị lớn: Nhiều cụm hơn, có thể dẫn đến overfitting.")
        st.write("- **init**: Phương pháp khởi tạo tâm cụm.")
        st.write("  - `k-means++`: Phân bố ban đầu thông minh để tăng tốc độ hội tụ.")
        st.write("  - `random`: Chọn ngẫu nhiên, có thể chậm hội tụ hơn.")
        st.write("- **max_iter**: Số lần lặp tối đa, kiểm soát số lần cập nhật tâm cụm.")
        st.write("- **random_state**: Đặt giá trị cố định để đảm bảo kết quả tái lập được.")

        st.write("### 3. Các bước huấn luyện")
        st.write("1. **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu (ví dụ: dùng StandardScaler) để các đặc trưng có cùng thang đo.")
        st.write("2. **Chọn số cụm (K)**: Xác định K dựa trên phương pháp Elbow hoặc Silhouette Score.")
        st.write("3. **Khởi tạo tâm cụm**: Chọn ngẫu nhiên hoặc dùng k-means++ để khởi tạo tâm cụm ban đầu.")
        st.write("4. **Phân cụm**: Gán mỗi điểm dữ liệu vào cụm gần nhất và cập nhật tâm cụm.")
        st.write("5. **Lặp lại**: Tiếp tục cho đến khi tâm cụm không đổi hoặc đạt max_iter.")
        st.write("6. **Đánh giá**: Sử dụng các chỉ số như Inertia hoặc Silhouette để đánh giá chất lượng phân cụm.")

        st.write("### 4. Ưu điểm")
        st.write("- Đơn giản và dễ triển khai.")
        st.write("- Hiệu quả với dữ liệu có hình dạng cụm tròn hoặc đều.")
        st.write("- Tốc độ nhanh với dữ liệu vừa và nhỏ.")

        st.write("### 5. Nhược điểm")
        st.write("- Yêu cầu biết trước số cụm (K).")
        st.write("- Không hiệu quả với dữ liệu có hình dạng cụm phức tạp hoặc không đồng nhất.")
        st.write("- Nhạy cảm với giá trị khởi tạo ban đầu.")
    else:  # DBSCAN
        st.write("### 1. Thông tin mô hình")
        st.write("DBSCAN (Density-Based Spatial Clustering of Applications with Noise) là một thuật toán phân cụm dựa trên mật độ, không yêu cầu số cụm cố định. Nó nhóm các điểm dựa trên mật độ dữ liệu và xác định các điểm nhiễu (noise) nếu không thuộc cụm nào.")

        st.write("### 2. Tham số và ý nghĩa")
        st.write("- **eps**: Khoảng cách tối đa giữa hai điểm để coi là thuộc cùng cụm.")
        st.write("  - Giá trị nhỏ: Tạo nhiều cụm nhỏ hơn, dễ bỏ sót điểm.")
        st.write("  - Giá trị lớn: Gộp các cụm, có thể mất chi tiết.")
        st.write("- **min_samples**: Số điểm tối thiểu trong khu vực eps để tạo thành một cụm.")
        st.write("  - Giá trị nhỏ: Dễ tạo cụm, có thể chứa nhiễu.")
        st.write("  - Giá trị lớn: Cụm chặt chẽ hơn, có thể bỏ sót cụm nhỏ.")

        st.write("### 3. Các bước huấn luyện")
        st.write("1. **Tiền xử lý dữ liệu**: Chuẩn hóa dữ liệu để đảm bảo các đặc trưng có cùng thang đo.")
        st.write("2. **Chọn tham số**: Xác định eps và min_samples dựa trên đặc điểm dữ liệu.")
        st.write("3. **Xác định điểm lõi**: Tìm các điểm có ít nhất min_samples điểm trong bán kính eps.")
        st.write("4. **Mở rộng cụm**: Kết nối các điểm lõi và các điểm biên để tạo cụm.")
        st.write("5. **Xác định nhiễu**: Các điểm không thuộc cụm nào được đánh dấu là nhiễu.")
        st.write("6. **Đánh giá**: Kiểm tra chất lượng phân cụm bằng các chỉ số như Silhouette Score.")

        st.write("### 4. Ưu điểm")
        st.write("- Không cần biết trước số cụm.")
        st.write("- Hiệu quả với dữ liệu có hình dạng cụm bất kỳ (không cần hình tròn).")
        st.write("- Có thể phát hiện nhiễu tự nhiên.")

        st.write("### 5. Nhược điểm")
        st.write("- Nhạy cảm với tham số eps và min_samples.")
        st.write("- Khó hoạt động với dữ liệu có mật độ khác nhau.")
        st.write("- Tốn tài nguyên với dữ liệu lớn.")