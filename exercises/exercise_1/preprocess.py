import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from common.utils import load_data, save_data
from common.mlflow_helper import log_preprocessing_params
import mlflow
import os
import random
import string
import dagshub
import datetime

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/vn0410.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "22fd02345f8ff45482a20960058627630acaf190"  # Thay bằng token cá nhân của bạn
    DAGSHUB_REPO = "vn0410"
    return DAGSHUB_REPO

def preprocess_data():
    st.header("Tiền xử lý dữ liệu Titanic 🛳️")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow
    DAGSHUB_REPO = mlflow_input()

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Nhập tên Experiment cho tiền xử lý", value="Titanic_Preprocessing")
    with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
        client = mlflow.tracking.MlflowClient()
        try:
            # Kiểm tra xem experiment đã tồn tại và bị xóa chưa
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                st.warning(f"Experiment '{experiment_name}' đã bị xóa trước đó. Vui lòng chọn tên khác hoặc khôi phục experiment qua DagsHub UI.")
                new_experiment_name = st.text_input("Nhập tên Experiment mới", value=f"{experiment_name}_Restored_{datetime.datetime.now().strftime('%Y%m%d')}")
                if new_experiment_name:
                    mlflow.set_experiment(new_experiment_name)
                    experiment_name = new_experiment_name
                else:
                    st.error("Vui lòng nhập tên experiment mới để tiếp tục.")
                    return
            else:
                mlflow.set_experiment(experiment_name)
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
            return

    # Khởi tạo session_state để lưu dữ liệu
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = {}

    # Upload file CSV
    uploaded_file = st.file_uploader("Tải lên file CSV Titanic 📂", type=["csv"])
    if uploaded_file and st.session_state['data'] is None:
        with st.spinner("Đang tải file CSV..."):
            st.session_state['data'] = pd.read_csv(uploaded_file)
            st.session_state['preprocessing_steps'] = {}
        st.success("File đã được tải lên thành công! ✅")

        if 'Name' in st.session_state['data'].columns:
            if st.session_state['data']['Name'].dtype != 'object':
                st.warning("Cột 'Name' không phải kiểu chuỗi (object).")
        if 'PassengerId' in st.session_state['data'].columns:
            if st.session_state['data']['PassengerId'].dtype not in ['int64', 'object']:
                st.warning("Cột 'PassengerId' không phải kiểu số nguyên (int64) hoặc chuỗi (object).")

    if st.session_state['data'] is None:
        st.warning("Vui lòng tải lên file CSV để tiếp tục. ⚠️")
        return

    # Hiển thị dữ liệu hiện tại
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
        help="Gợi ý: Loại bỏ 'Ticket', 'Cabin' nếu không cần thiết."
    )
    if st.button("Loại bỏ các cột đã chọn 🗑️"):
        if columns_to_drop:
            st.session_state['data'] = st.session_state['data'].drop(columns=columns_to_drop)
            st.session_state['preprocessing_steps']["dropped_columns"] = columns_to_drop
            st.success(f"Đã loại bỏ các cột: {', '.join(columns_to_drop)}")
            st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

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

                def normalize_cabin(cabin):
                    if pd.isna(cabin):
                        return None
                    if isinstance(cabin, str) and ' ' in cabin:
                        parts = cabin.split()
                        if parts:
                            cabin = parts[0]
                    if isinstance(cabin, str) and cabin:
                        match = ''.join(filter(str.isalnum, cabin))
                        if match and len(match) > 1 and match[0].isalpha() and match[1:].isdigit():
                            return match
                    return None

                st.session_state['data'][col] = st.session_state['data'][col].apply(normalize_cabin)

                if fill_method == "Mode (định dạng chuẩn)":
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        mode_value = st.session_state['data'][col].mode()[0] if not st.session_state['data'][col].mode().empty else None
                        if mode_value and isinstance(mode_value, str) and len(mode_value) > 1 and mode_value[0].isalpha() and mode_value[1:].isdigit():
                            st.session_state['data'][col] = st.session_state['data'][col].fillna(mode_value)
                            st.session_state['preprocessing_steps'][f"{col}_filled"] = f"mode_{mode_value}"
                            st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng mode: {mode_value}.")
                        else:
                            st.error("Không tìm thấy mode phù hợp định dạng 'Chữ + Số'.")
                        st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

                elif fill_method == "Giá trị ngẫu nhiên theo định dạng Cabin":
                    def generate_cabin():
                        letter = random.choice(string.ascii_uppercase)
                        number = random.randint(1, 999)
                        return f"{letter}{number}"
                    
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        st.session_state['data'][col] = st.session_state['data'].apply(
                            lambda row: row[col] if pd.notnull(row[col]) else generate_cabin(), axis=1
                        )
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = "random_cabin_format"
                        st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng giá trị ngẫu nhiên.")
                        st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

                elif fill_method == "Giá trị phổ biến nhất theo chữ cái":
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        valid_cabins = st.session_state['data'][col].dropna().apply(normalize_cabin).dropna()
                        if not valid_cabins.empty:
                            first_letters = valid_cabins.str[0].value_counts()
                            if not first_letters.empty:
                                most_common_letter = first_letters.idxmax()
                                number = random.randint(1, 999)
                                fill_value = f"{most_common_letter}{number}"
                                st.session_state['data'][col] = st.session_state['data'][col].fillna(fill_value)
                                st.session_state['preprocessing_steps'][f"{col}_filled"] = f"most_common_letter_{fill_value}"
                                st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng: {fill_value}.")
                            else:
                                st.error("Không thể xác định chữ cái phổ biến nhất.")
                        else:
                            st.error("Không có giá trị Cabin hợp lệ để phân tích.")
                        st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

            elif col == 'Age':
                st.info("Gợi ý: Dùng 'median' hoặc 'mode' cho 'Age' để giữ kiểu số nguyên.")
                fill_method = st.selectbox(
                    f"Chọn phương pháp điền cho '{col}'",
                    ["Median", "Mode", "Giá trị tùy chỉnh (số nguyên)"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Giá trị tùy chỉnh (số nguyên)":
                    custom_value = st.number_input(f"Nhập giá trị tùy chỉnh cho '{col}'", min_value=0, max_value=150, value=30, step=1, key=f"custom_{col}")
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
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng {fill_method.lower()}.")
                    st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

            elif st.session_state['data'][col].dtype in ['int64', 'float64']:
                st.info("Gợi ý: Dùng 'median' hoặc 'mean' cho dữ liệu số.")
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
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng {fill_method.lower()}.")
                    st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])

            else:
                st.info("Gợi ý: Dùng 'mode' cho dữ liệu phân loại.")
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
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng {fill_method.lower()}.")
                    st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])
    else:
        st.success("Không phát hiện dữ liệu thiếu (ngoại trừ 'Name' và 'PassengerId'). ✅")

    # 3. Chuyển đổi dữ liệu phân loại
    st.write("### Bước 3: Chuyển đổi cột phân loại 🔠")
    categorical_cols = [col for col in st.session_state['data'].columns if col not in ['Name', 'PassengerId'] and st.session_state['data'][col].dtype == 'object']
    if categorical_cols:
        for col in categorical_cols:
            st.write(f"#### Chuyển đổi '{col}'")
            st.info("Gợi ý: 'Label Encoding' cho dữ liệu có thứ tự; 'One-Hot Encoding' cho dữ liệu không thứ tự.")
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
                st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])
    else:
        st.success("Không có cột phân loại nào (ngoại trừ 'Name' và 'PassengerId') để mã hóa.")

    # 4. Chuẩn hóa dữ liệu
    st.write("### Bước 4: Chuẩn hóa/Dữ liệu quy mô 🔢")
    numerical_cols = [col for col in st.session_state['data'].columns if col not in ['Name', 'PassengerId', 'Survived'] and st.session_state['data'][col].dtype in ['int64', 'float64']]
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
                st.write("Xem trước dữ liệu đã cập nhật:", st.session_state['data'])
            else:
                st.warning("Không có cột nào được chọn để chuẩn hóa.")
    else:
        st.success("Không có cột số nào (ngoại trừ 'Name', 'PassengerId', 'Survived') để chuẩn hóa.")

    # 5. Lưu dữ liệu
    st.write("### Bước 5: Lưu dữ liệu 💾")
    if st.button("Lưu dữ liệu đã xử lý 💾"):
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        with st.spinner("Đang lưu dữ liệu..."):
            save_data(st.session_state['data'], processed_file)
            st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

            st.subheader("Xem trước dữ liệu đã xử lý cuối cùng 🔚")
            st.write(st.session_state['data'])

            saved_data = load_data(processed_file)
            st.write("Xác nhận: Dữ liệu tải lại từ file đã lưu:", saved_data)

    # Logging và tracking vào MLflow trên DagsHub
    run_id_input = st.text_input("Nhập tên Run ID (để trống để tự động tạo)", value="", max_chars=10, help="Tên ngắn gọn, ví dụ: 'Run1'")
    if st.button("Lưu dữ liệu đã xử lý và log 📋"):
        if mlflow.active_run():
            mlflow.end_run()
            st.info("Đã đóng run MLflow đang hoạt động trước khi bắt đầu log mới.")

        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        with st.spinner("Đang lưu và log dữ liệu lên DagsHub..."):
            save_data(st.session_state['data'], processed_file)
            st.success(f"Dữ liệu đã được lưu vào {processed_file} 💾")

            st.subheader("Xem trước dữ liệu đã xử lý cuối cùng 🔚")
            st.write(st.session_state['data'])

            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = run_id_input if run_id_input else f"Run_{timestamp[-6:]}"

            try:
                with mlflow.start_run(run_name=run_name) as run:
                    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_preprocessing_params(st.session_state['preprocessing_steps'])
                    mlflow.log_artifact(processed_file, artifact_path="processed_data")
                    mlflow.log_param("num_rows", len(st.session_state['data']))
                    mlflow.log_param("num_columns", len(st.session_state['data'].columns))
                    mlflow.log_metric("missing_values_before", missing_info.sum())
                    mlflow.log_metric("missing_values_after", st.session_state['data'].isnull().sum().sum())
                    mlflow.log_metric("missing_values_handled", missing_info.sum() - st.session_state['data'].isnull().sum().sum())

                    run_id = run.info.run_id
                    mlflow_uri = st.session_state['mlflow_url']
                    st.success(f"Đã log dữ liệu thành công lúc {log_time}! 📊")
                    st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

            except Exception as e:
                st.error(f"Lỗi khi log: {str(e)}")

            saved_data = load_data(processed_file)
            st.write("Xác nhận: Dữ liệu tải lại từ file đã lưu:", saved_data)

if __name__ == "__main__":
    preprocess_data()