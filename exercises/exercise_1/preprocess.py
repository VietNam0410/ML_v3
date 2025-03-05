import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from common.utils import load_data  # Chỉ giữ load_data, không dùng save_data
from common.mlflow_helper import log_preprocessing_params
import mlflow
import os
import random
import string
import dagshub
import datetime
import sklearn

# Hàm khởi tạo MLflow với caching
@st.cache_resource
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

# Hàm tiền xử lý dữ liệu
def preprocess_data():
    st.header("Tiền xử lý dữ liệu Titanic 🛳️")

    # Khởi tạo MLflow chỉ một lần và lưu vào session_state
    if 'mlflow_url' not in st.session_state:
        with st.spinner("Đang khởi tạo MLflow..."):
            try:
                mlflow_uri = mlflow_input()
                st.session_state['mlflow_url'] = mlflow_uri
                st.success("Đã khởi tạo MLflow thành công!")
            except Exception as e:
                st.error(f"Lỗi khi thiết lập MLflow: {str(e)}")
                return

    # Thiết lập experiment "Titanic_Preprocessing" cố định
    experiment_name = "Titanic_Preprocessing"
    if 'experiment_set' not in st.session_state:
        with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
            try:
                mlflow.set_tracking_uri(st.session_state['mlflow_url'])
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    mlflow.create_experiment(experiment_name)
                elif experiment.lifecycle_stage == "deleted":
                    mlflow.tracking.MlflowClient().restore_experiment(experiment.experiment_id)
                mlflow.set_experiment(experiment_name)
                st.session_state['experiment_set'] = True
                st.success(f"Đã thiết lập Experiment '{experiment_name}' thành công!")
            except Exception as e:
                st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
                return

    # Khởi tạo session_state
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

        if 'Name' in st.session_state['data'].columns and st.session_state['data']['Name'].dtype != 'object':
            st.warning("Cột 'Name' không phải kiểu chuỗi (object).")
        if 'PassengerId' in st.session_state['data'].columns and st.session_state['data']['PassengerId'].dtype not in ['int64', 'object']:
            st.warning("Cột 'PassengerId' không phải kiểu số nguyên (int64) hoặc chuỗi (object).")

    if st.session_state['data'] is None:
        st.warning("Vui lòng tải lên file CSV để tiếp tục. ⚠️")
        return

    # Hiển thị dữ liệu
    st.subheader("Xem trước dữ liệu hiện tại 🔍")
    st.write(st.session_state['data'])

    # Hiển thị thông tin dữ liệu thiếu
    st.subheader("Dữ liệu thiếu ⚠️")
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
            st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

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

                if f"{col}_normalized" not in st.session_state['preprocessing_steps']:
                    st.session_state['data'][col] = st.session_state['data'][col].apply(normalize_cabin)
                    st.session_state['preprocessing_steps'][f"{col}_normalized"] = True

                if fill_method == "Mode (định dạng chuẩn)":
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        mode_value = st.session_state['data'][col].mode()[0] if not st.session_state['data'][col].mode().empty else None
                        if mode_value and isinstance(mode_value, str) and len(mode_value) > 1 and mode_value[0].isalpha() and mode_value[1:].isdigit():
                            st.session_state['data'][col] = st.session_state['data'][col].fillna(mode_value)
                            st.session_state['preprocessing_steps'][f"{col}_filled"] = f"mode_{mode_value}"
                            st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng mode: {mode_value}.")
                        else:
                            st.error("Không tìm thấy mode phù hợp định dạng 'Chữ + Số'.")
                        st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

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
                        st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

                elif fill_method == "Giá trị phổ biến nhất theo chữ cái":
                    if st.button(f"Điền giá trị cho '{col}' ✏️", key=f"fill_{col}"):
                        valid_cabins = st.session_state['data'][col].dropna()
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
                        st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

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
                    st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

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
                    st.write("Dữ liệu đã cập nhật:", st.session_state['data'])

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
                        mode_value = st.session_state['data'][col].mode()[0] if not st.session_state['data'][col].mode().empty else None
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(mode_value)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"mode_{mode_value}"
                    elif fill_method == "Giá trị tùy chỉnh":
                        st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value)
                        st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"
                    st.success(f"Đã điền dữ liệu thiếu ở '{col}' bằng {fill_method.lower()}.")
                    st.write("Dữ liệu đã cập nhật:", st.session_state['data'])
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
                st.write("Dữ liệu đã cập nhật:", st.session_state['data'])
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
                st.write("Dữ liệu đã cập nhật:", st.session_state['data'])
            else:
                st.warning("Không có cột nào được chọn để chuẩn hóa.")
    else:
        st.success("Không có cột số nào (ngoại trừ 'Name', 'PassengerId', 'Survived') để chuẩn hóa.")

    # 5. Lưu và log dữ liệu
    st.write("### Bước 5: Lưu và Log dữ liệu 📋")
    run_id_input = st.text_input("Nhập tên Run ID (để trống để tự động tạo)", value="", max_chars=10)
    if st.button("Lưu và log dữ liệu lên DagsHub 📤"):
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        os.makedirs(os.path.dirname(processed_file), exist_ok=True)
        
        with st.spinner("Đang xử lý và log dữ liệu..."):
            # Lưu file trực tiếp mà không dùng save_data
            try:
                st.session_state['data'].to_csv(
                    processed_file,
                    index=False,
                    compression='infer',  # Giữ nén nếu cần
                    encoding='utf-8'
                )
            except Exception as e:
                st.error(f"Lỗi khi lưu file: {str(e)}")
                return
            
            # Tạo run name
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            run_name = ''.join(c for c in run_id_input if c.isalnum() or c in ['_', '-']) if run_id_input else f"Preprocess_{timestamp[-6:]}"
            
            # Đảm bảo experiment được đặt trước khi log
            mlflow.set_experiment(experiment_name)
            
            # Bắt đầu MLflow run
            try:
                with mlflow.start_run(run_name=run_name):
                    log_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Log parameters và metrics
                    log_preprocessing_params(st.session_state['preprocessing_steps'])
                    mlflow.log_artifact(processed_file, artifact_path="processed_data")
                    mlflow.log_param("num_rows", len(st.session_state['data']))
                    mlflow.log_param("num_columns", len(st.session_state['data'].columns))
                    mlflow.log_param("pandas_version", pd.__version__)
                    mlflow.log_param("sklearn_version", sklearn.__version__)
                    mlflow.log_metric("missing_values_before", missing_info.sum())
                    mlflow.log_metric("missing_values_after", st.session_state['data'].isnull().sum().sum())
                    mlflow.log_metric("missing_values_handled", missing_info.sum() - st.session_state['data'].isnull().sum().sum())
                    
                    # Lấy run ID và URL
                    run_id = mlflow.active_run().info.run_id
                    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                    run_url = f"{st.session_state['mlflow_url']}/#/experiments/{experiment_id}/runs/{run_id}"
                    
                    st.success(f"Đã log dữ liệu thành công lúc {log_time} vào '{experiment_name}'! 📊")
                    st.markdown(f"Xem chi tiết tại: [{run_url}]({run_url})")
                    
            except Exception as e:
                st.error(f"Lỗi khi log lên MLflow: {str(e)}")
                return
                
            st.success(f"Dữ liệu đã được lưu tại: {processed_file}")
            st.write("Dữ liệu đã xử lý:", st.session_state['data'])
            saved_data = load_data(processed_file)
            st.write("Xác nhận dữ liệu từ file đã lưu:", saved_data)

if __name__ == "__main__":
    preprocess_data()