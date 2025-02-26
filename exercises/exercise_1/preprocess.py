import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from common.utils import load_data, save_data
from common.mlflow_helper import log_preprocessing_params
import mlflow
import os 
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def preprocess_data():
    st.header("Tiền xử lý dữ liệu Titanic 🛳️")

    # Khởi tạo session_state để lưu dữ liệu
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = {}

    # Bắt buộc người dùng upload file trước
    uploaded_file = st.file_uploader("Tải lên file CSV Titanic 📂", type=["csv"])
    if uploaded_file and st.session_state['data'] is None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.session_state['preprocessing_steps'] = {}  # Reset steps khi upload file mới
        st.success("File đã được tải lên thành công! ✅")

    if st.session_state['data'] is None:
        st.warning("Vui lòng tải lên file CSV để tiếp tục. ⚠️")
        return

    # Hiển thị dữ liệu gốc hoặc dữ liệu đang xử lý
    st.subheader("Xem trước dữ liệu hiện tại 🔍")
    st.write("Đây là dữ liệu khi bạn tiến hành xử lý từng bước:")
    st.write(st.session_state['data'].head())

    # Hiển thị thông tin còn thiếu
    st.subheader("Dữ liệu thiếu (Trạng thái hiện tại) ⚠️")
    missing_info = st.session_state['data'].isnull().sum()
    st.write(missing_info)

    # 1. Loại bỏ cột
    st.write("### Bước 1: Loại bỏ cột 🗑️")
    columns_to_drop = st.multiselect(
        "Chọn cột cần loại bỏ",
        options=st.session_state['data'].columns.tolist(),
        help="Gợi ý: Loại bỏ 'Cabin' (nhiều dữ liệu thiếu), 'Name', 'Ticket', 'PassengerId' (không hữu ích)."
    )
    if st.button("Loại bỏ các cột đã chọn 🗑️"):
        if columns_to_drop:
            st.session_state['data'] = st.session_state['data'].drop(columns=columns_to_drop)
            st.session_state['preprocessing_steps']["dropped_columns"] = columns_to_drop
            st.success(f"Đã loại bỏ các cột: {', '.join(columns_to_drop)}")
            st.write("Xem trước dữ liệu đã cập nhật (Sau khi loại bỏ):", st.session_state['data'].head())

    # 2. Điền giá trị thiếu
    st.write("### Bước 2: Điền giá trị thiếu ✏️")
    missing_columns = [col for col in st.session_state['data'].columns if st.session_state['data'][col].isnull().sum() > 0]
    if missing_columns:
        for col in missing_columns:
            st.write(f"#### Xử lý dữ liệu thiếu ở cột '{col}'")
            st.write(f"Dữ liệu thiếu: {st.session_state['data'][col].isnull().sum()} trên tổng số {len(st.session_state['data'])} dòng")

            if st.session_state['data'][col].dtype in ['int64', 'float64']:
                st.info(f"Gợi ý: Dùng 'median' hoặc 'mean' cho dữ liệu số. Median bền vững với ngoại lệ.")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Mean", "Median", "Giá trị tùy chỉnh"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Giá trị tùy chỉnh":
                    custom_value = st.number_input(f"Nhập giá trị tùy chỉnh cho '{col}'", key=f"custom_{col}")
            else:
                st.info(f"Gợi ý: Dùng 'mode' cho dữ liệu phân loại.")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Mode", "Giá trị tùy chỉnh"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Giá trị tùy chỉnh":
                    custom_value = st.text_input(f"Nhập giá trị tùy chỉnh cho '{col}'", key=f"custom_{col}")

            if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                if fill_method == "Mean":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mean())
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "mean"
                elif fill_method == "Median":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].median())
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "median"
                elif fill_method == "Mode":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mode()[0])
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "mode"
                elif fill_method == "Giá trị tùy chỉnh":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value)
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"

                st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng phương pháp {fill_method.lower()}.")
                st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'].head())
    else:
        st.success("Không phát hiện dữ liệu thiếu trong tập dữ liệu hiện tại. ✅")

    # 3. Chuyển đổi dữ liệu phân loại
    st.write("### Bước 3: Chuyển đổi cột phân loại 🔠")
    categorical_cols = [col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype == 'object']
    if categorical_cols:
        for col in categorical_cols:
            st.write(f"#### Chuyển đổi '{col}'")
            st.info(f"Gợi ý: 'Label Encoding' cho dữ liệu có thứ tự/có ít giá trị; 'One-Hot Encoding' cho dữ liệu không thứ tự/có ít giá trị.")
            encoding_method = st.selectbox(
                f"Chọn phương pháp mã hóa cho '{col}'",
                ["Label Encoding", "One-Hot Encoding"],
                key=f"encode_{col}"
            )
            if st.button(f"Áp dụng mã hóa cho '{col}' 🔠", key=f"apply_{col}"):
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    st.session_state['data'][col] = le.fit_transform(st.session_state['data'][col])
                    st.session_state['preprocessing_steps'][f"{col}_encoded"] = "label"
                    st.success(f"Đã áp dụng Label Encoding cho '{col}'.")
                elif encoding_method == "One-Hot Encoding":
                    st.session_state['data'] = pd.get_dummies(st.session_state['data'], columns=[col], prefix=col)
                    st.session_state['preprocessing_steps'][f"{col}_encoded"] = "one-hot"
                    st.success(f"Đã áp dụng One-Hot Encoding cho '{col}'.")
                st.write("Xem trước dữ liệu đã cập nhật (Sau khi mã hóa):", st.session_state['data'].head())
    else:
        st.success("Không có cột phân loại nào để mã hóa.")

    # 4. Chuẩn hóa dữ liệu
    st.write("### Bước 4: Chuẩn hóa/Dữ liệu quy mô 🔢")
    numerical_cols = [col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype in ['int64', 'float64']]
    if numerical_cols:
        st.info("Gợi ý: 'Min-Max Scaling' (0-1) cho phạm vi giới hạn; 'Standard Scaling' (mean=0, std=1) cho dữ liệu chuẩn.")
        scaling_method = st.selectbox(
            "Chọn phương pháp chuẩn hóa",
            ["Min-Max Scaling", "Standard Scaling"]
        )
        cols_to_scale = st.multiselect(
            "Chọn các cột số cần chuẩn hóa",
            options=numerical_cols,
            default=numerical_cols
        )
        if st.button("Áp dụng chuẩn hóa 📏"):
            if cols_to_scale:
                if scaling_method == "Min-Max Scaling":
                    scaler = MinMaxScaler()
                    st.session_state['data'][cols_to_scale] = scaler.fit_transform(st.session_state['data'][cols_to_scale])
                    st.session_state['preprocessing_steps']["scaling"] = "min-max"
                elif scaling_method == "Standard Scaling":
                    scaler = StandardScaler()
                    st.session_state['data'][cols_to_scale] = scaler.fit_transform(st.session_state['data'][cols_to_scale])
                    st.session_state['preprocessing_steps']["scaling"] = "standard"
                st.session_state['preprocessing_steps']["scaled_columns"] = cols_to_scale
                st.success(f"Đã áp dụng {scaling_method} cho các cột: {', '.join(cols_to_scale)}")
                st.write("Xem trước dữ liệu đã cập nhật (Sau khi chuẩn hóa):", st.session_state['data'].head())
            else:
                st.warning("Không có cột nào được chọn để chuẩn hóa.")
    else:
        st.success("Không có cột số nào để chuẩn hóa.")

    # 5. Lưu và log dữ liệu đã tiền xử lý
    st.write("### Bước 5: Lưu dữ liệu đã tiền xử lý 💾")
    if st.button("Lưu dữ liệu đã xử lý và log vào MLflow 📋"):
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        save_data(st.session_state['data'], processed_file)
        st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

        # Hiển thị dữ liệu cuối cùng trước khi lưu
        st.subheader("Xem trước dữ liệu đã xử lý cuối cùng (Trước khi lưu) 🔚")
        st.write(st.session_state['data'].head())

        # Log các tham số tiền xử lý vào MLflow
        log_preprocessing_params(st.session_state['preprocessing_steps'])
        st.success("Các bước tiền xử lý đã được log vào MLflow! 📊")

        # Xác nhận dữ liệu đã lưu đúng
        saved_data = load_data(processed_file)
        st.write("Xác nhận: Dữ liệu tải lại từ file đã lưu trùng khớp với các lựa chọn tiền xử lý của bạn:")
        st.write(saved_data.head())

if __name__ == "__main__":
    preprocess_data()
