import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from common.utils import load_data, save_data
from common.mlflow_helper import log_preprocessing_params
import mlflow
import os
import random
import string

# Thiết lập tracking URI cục bộ
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def preprocess_data():
    st.header("Tiền xử lý dữ liệu Titanic 🛳️")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="Titanic_Preprocessing")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

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

        # Kiểm tra kiểu dữ liệu của Name và PassengerId
        if 'Name' in st.session_state['data'].columns:
            if st.session_state['data']['Name'].dtype != 'object':
                st.warning("Cột 'Name' không phải kiểu chuỗi (object). Đảm bảo dữ liệu Name là chuỗi trước khi tiếp tục.")
        if 'PassengerId' in st.session_state['data'].columns:
            if st.session_state['data']['PassengerId'].dtype not in ['int64', 'object']:
                st.warning("Cột 'PassengerId' không phải kiểu số nguyên (int64) hoặc chuỗi (object). Đảm bảo dữ liệu PassengerId là số nguyên hoặc chuỗi trước khi tiếp tục.")

    if st.session_state['data'] is None:
        st.warning("Vui lòng tải lên file CSV để tiếp tục. ⚠️")
        return

    # Hiển thị dữ liệu gốc hoặc dữ liệu đang xử lý (bao gồm Name và PassengerId)
    st.subheader("Xem trước dữ liệu hiện tại 🔍")
    st.write("Đây là dữ liệu khi bạn tiến hành xử lý từng bước:")
    st.write(st.session_state['data'])

    # Hiển thị thông tin còn thiếu
    st.subheader("Dữ liệu thiếu (Trạng thái hiện tại) ⚠️")
    missing_info = st.session_state['data'].isnull().sum()
    st.write(missing_info)

    # 1. Loại bỏ cột
    st.write("### Bước 1: Loại bỏ cột 🗑️")
    columns_to_drop = st.multiselect(
        "Chọn cột cần loại bỏ",
        options=st.session_state['data'].columns.tolist(),
        help="Gợi ý: Loại bỏ 'Ticket', 'Cabin', 'Name', 'PassengerId' nếu không cần thiết."
    )
    if st.button("Loại bỏ các cột đã chọn 🗑️"):
        if columns_to_drop:
            st.session_state['data'] = st.session_state['data'].drop(columns=columns_to_drop)
            st.session_state['preprocessing_steps']["dropped_columns"] = columns_to_drop
            st.success(f"Đã loại bỏ các cột: {', '.join(columns_to_drop)}")
            st.write("Xem trước dữ liệu đã cập nhật (Sau khi loại bỏ):", st.session_state['data'])

    # 2. Điền giá trị thiếu
    st.write("### Bước 2: Điền giá trị thiếu ✏️")
    missing_columns = [col for col in st.session_state['data'].columns if col not in ['Name', 'PassengerId'] and st.session_state['data'][col].isnull().sum() > 0]
    if missing_columns:
        for col in missing_columns:
            st.write(f"#### Xử lý dữ liệu thiếu ở cột '{col}'")
            st.write(f"Dữ liệu thiếu: {st.session_state['data'][col].isnull().sum()} trên tổng số {len(st.session_state['data'])} dòng")

            if col == 'Cabin':
                st.info("Cột 'Cabin' sẽ được điền theo định dạng 'Chữ + Số' (ví dụ: C123).")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Mode (định dạng chuẩn)", "Giá trị ngẫu nhiên theo định dạng Cabin", "Giá trị phổ biến nhất theo chữ cái"],
                    key=f"fill_method_{col}"
                )

                # Hàm chuẩn hóa định dạng Cabin (1 chữ cái + số)
                def normalize_cabin(cabin):
                    if pd.isna(cabin):
                        return None
                    # Lấy phần đầu tiên (chữ cái + số) nếu có nhiều giá trị
                    if isinstance(cabin, str) and ' ' in cabin:
                        parts = cabin.split()
                        if parts:
                            cabin = parts[0]
                    # Đảm bảo định dạng đúng (1 chữ cái + số)
                    if isinstance(cabin, str) and cabin:
                        match = ''.join(filter(str.isalnum, cabin))  # Lọc chỉ chữ cái và số
                        if match and len(match) > 1 and match[0].isalpha() and match[1:].isdigit():
                            return match
                    return None

                # Chuẩn hóa dữ liệu Cabin trước khi xử lý
                st.session_state['data'][col] = st.session_state['data'][col].apply(normalize_cabin)

                if fill_method == "Mode (định dạng chuẩn)":
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        # Tìm mode của cột Cabin đã chuẩn hóa
                        mode_value = st.session_state['data'][col].mode()[0] if not st.session_state['data'][col].mode().empty else None
                        if mode_value and isinstance(mode_value, str) and len(mode_value) > 1 and mode_value[0].isalpha() and mode_value[1:].isdigit():
                            st.session_state['data'][col] = st.session_state['data'][col].fillna(mode_value)
                            st.session_state['preprocessing_steps'][f"{col}_filled"] = f"mode_{mode_value}"
                            st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng mode: {mode_value}.")
                        else:
                            st.error("Không tìm thấy mode phù hợp định dạng 'Chữ + Số'. Vui lòng thử phương pháp khác.")
                        st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])

                elif fill_method == "Giá trị ngẫu nhiên theo định dạng Cabin":
                    def generate_cabin():
                        letter = random.choice(string.ascii_uppercase)
                        number = random.randint(1, 999)
                        return f"{letter}{number}"
                    
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        st.session_state['data'][col] = st.session_state['data'][col].apply(
                            lambda x: x if pd.notnull(x) else generate_cabin()
                        )
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = "random_cabin_format"
                        st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng giá trị ngẫu nhiên theo định dạng Cabin.")
                        st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])

                elif fill_method == "Giá trị phổ biến nhất theo chữ cái":
                    # Phân tích giá trị phổ biến nhất dựa trên chữ cái đầu tiên
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        # Lấy tất cả giá trị Cabin không thiếu
                        valid_cabins = st.session_state['data'][col].dropna().apply(normalize_cabin).dropna()
                        if not valid_cabins.empty:
                            # Đếm số lần xuất hiện của chữ cái đầu tiên
                            first_letters = valid_cabins.str[0].value_counts()
                            if not first_letters.empty:
                                most_common_letter = first_letters.idxmax()
                                # Tạo giá trị ngẫu nhiên với chữ cái phổ biến nhất
                                number = random.randint(1, 999)
                                fill_value = f"{most_common_letter}{number}"
                                st.session_state['data'][col] = st.session_state['data'][col].fillna(fill_value)
                                st.session_state['preprocessing_steps'][f"{col}_filled"] = f"most_common_letter_{fill_value}"
                                st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng giá trị phổ biến nhất theo chữ cái: {fill_value}.")
                            else:
                                st.error("Không thể xác định chữ cái phổ biến nhất. Vui lòng thử phương pháp khác.")
                        else:
                            st.error("Không có giá trị Cabin hợp lệ để phân tích. Vui lòng thử phương pháp khác.")
                        st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])

            elif col == 'Age':
                st.info(f"Gợi ý: Dùng 'median' hoặc 'mode' cho cột 'Age' để giữ kiểu số nguyên. Median bền vững với ngoại lệ.")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Median", "Mode", "Giá trị tùy chỉnh (số nguyên)"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Giá trị tùy chỉnh (số nguyên)":
                    custom_value = st.number_input(f"Nhập giá trị tùy chỉnh cho '{col}' (số nguyên)", min_value=0, max_value=150, value=30, step=1, key=f"custom_{col}")
                if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                    if fill_method == "Median":
                        median_value = int(st.session_state['data'][col].median()) if not pd.isna(st.session_state['data'][col].median()) else 0
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(median_value).astype(int)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"median_{median_value}"
                    elif fill_method == "Mode":
                        mode_value = int(st.session_state['data'][col].mode()[0]) if not st.session_state['data'][col].mode().empty else 0
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(mode_value).astype(int)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"mode_{mode_value}"
                    elif fill_method == "Giá trị tùy chỉnh (số nguyên)":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value).astype(int)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng phương pháp {fill_method.lower()} và chuyển thành số nguyên.")
                    st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])

            elif st.session_state['data'][col].dtype in ['int64', 'float64']:
                st.info(f"Gợi ý: Dùng 'median' hoặc 'mean' cho dữ liệu số. Median bền vững với ngoại lệ.")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Mean", "Median", "Giá trị tùy chỉnh"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Giá trị tùy chỉnh":
                    custom_value = st.number_input(f"Nhập giá trị tùy chỉnh cho '{col}'", key=f"custom_{col}")
                if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                    if fill_method == "Mean":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mean())
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = "mean"
                    elif fill_method == "Median":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].median())
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = "median"
                    elif fill_method == "Giá trị tùy chỉnh":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng phương pháp {fill_method.lower()}.")
                    st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])
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
                    if fill_method == "Mode":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mode()[0])
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = "mode"
                    elif fill_method == "Giá trị tùy chỉnh":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng phương pháp {fill_method.lower()}.")
                    st.write("Xem trước dữ liệu đã cập nhật (Sau khi điền):", st.session_state['data'])
    else:
        st.success("Không phát hiện dữ liệu thiếu trong tập dữ liệu hiện tại (ngoại trừ 'Name' và 'PassengerId'). ✅")

    # 3. Chuyển đổi dữ liệu phân loại
    st.write("### Bước 3: Chuyển đổi cột phân loại 🔠")
    categorical_cols = [col for col in st.session_state['data'].columns if col not in ['Name', 'PassengerId'] and st.session_state['data'][col].dtype == 'object']
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
                    st.success(f"Đã áp dụng Label Encoding cho '{col}'. Cột 'Name' và 'PassengerId' không bị ảnh hưởng.")
                elif encoding_method == "One-Hot Encoding":
                    st.session_state['data'] = pd.get_dummies(st.session_state['data'], columns=[col], prefix=col)
                    st.session_state['preprocessing_steps'][f"{col}_encoded"] = "one-hot"
                    st.success(f"Đã áp dụng One-Hot Encoding cho '{col}'. Cột 'Name' và 'PassengerId' không bị ảnh hưởng.")
                st.write("Xem trước dữ liệu đã cập nhật (Sau khi mã hóa):", st.session_state['data'])
    else:
        st.success("Không có cột phân loại nào (ngoại trừ 'Name' và 'PassengerId') để mã hóa.")

    # 4. Chuẩn hóa dữ liệu
    st.write("### Bước 4: Chuẩn hóa/Dữ liệu quy mô 🔢")
    numerical_cols = [col for col in st.session_state['data'].columns if col not in ['Name', 'PassengerId'] and st.session_state['data'][col].dtype in ['int64', 'float64']]
    if numerical_cols:
        st.info("Gợi ý: 'Min-Max Scaling' (0-1) cho phạm vi giới hạn; 'Standard Scaling' (mean=0, std=1) cho dữ liệu chuẩn. Cột 'Name' và 'PassengerId' sẽ không được chuẩn hóa.")
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
                st.success(f"Đã áp dụng {scaling_method} cho các cột: {', '.join(cols_to_scale)}. Cột 'Name' và 'PassengerId' không bị ảnh hưởng.")
                st.write("Xem trước dữ liệu đã cập nhật (Sau khi chuẩn hóa):", st.session_state['data'])
            else:
                st.warning("Không có cột nào được chọn để chuẩn hóa.")
    else:
        st.success("Không có cột số nào (ngoại trừ 'Name' và 'PassengerId') để chuẩn hóa.")

    # 5. Lưu và log dữ liệu đã tiền xử lý
    st.write("### Bước 5: Lưu dữ liệu đã tiền xử lý 💾")
    if st.button("Lưu dữ liệu đã xử lý và log vào MLflow 📋"):
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        save_data(st.session_state['data'], processed_file)
        st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

        # Hiển thị dữ liệu cuối cùng trước khi lưu (bao gồm Name và PassengerId)
        st.subheader("Xem trước dữ liệu đã xử lý cuối cùng (Trước khi lưu) 🔚")
        st.write(st.session_state['data'])

        # Log các tham số tiền xử lý vào MLflow
        log_preprocessing_params(st.session_state['preprocessing_steps'])
        st.success("Các bước tiền xử lý đã được log vào MLflow! 📊")

        # Xác nhận dữ liệu đã lưu đúng
        saved_data = load_data(processed_file)
        st.write("Xác nhận: Dữ liệu tải lại từ file đã lưu trùng khớp với các lựa chọn tiền xử lý của bạn:")
        st.write(saved_data)

if __name__ == "__main__":
    preprocess_data()
