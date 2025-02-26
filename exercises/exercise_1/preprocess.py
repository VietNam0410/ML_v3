import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from common.utils import load_data, save_data
from common.mlflow_helper import log_preprocessing_params
import mlflow
import os

# Thiết lập tracking URI cục bộ
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def preprocess_data():
    st.header("Preprocess Titanic Data")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Enter Experiment Name for Preprocessing", value="Titanic_Preprocessing")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Khởi tạo session_state để lưu dữ liệu
    if 'data' not in st.session_state:
        st.session_state['data'] = None
    if 'preprocessing_steps' not in st.session_state:
        st.session_state['preprocessing_steps'] = {}

    # Bắt buộc người dùng upload file trước
    uploaded_file = st.file_uploader("Upload Titanic CSV file", type=["csv"])
    if uploaded_file and st.session_state['data'] is None:
        st.session_state['data'] = pd.read_csv(uploaded_file)
        st.session_state['preprocessing_steps'] = {}  # Reset steps khi upload file mới
        st.success("File uploaded successfully!")

    if st.session_state['data'] is None:
        st.warning("Please upload a CSV file to proceed.")
        return

    # Hiển thị dữ liệu gốc hoặc dữ liệu đang xử lý
    st.subheader("Current Data Preview")
    st.write("This reflects the data as you preprocess it step-by-step:")
    st.write(st.session_state['data'].head())

    # Hiển thị thông tin còn thiếu
    st.subheader("Missing Values (Current State)")
    missing_info = st.session_state['data'].isnull().sum()
    st.write(missing_info)

    # 1. Loại bỏ cột
    st.write("### Step 1: Remove Columns")
    columns_to_drop = st.multiselect(
        "Select columns to drop",
        options=st.session_state['data'].columns.tolist(),
        help="Suggestion: Drop 'Cabin' (many missing), 'Name', 'Ticket', 'PassengerId' (not useful)."
    )
    if st.button("Drop Selected Columns"):
        if columns_to_drop:
            st.session_state['data'] = st.session_state['data'].drop(columns=columns_to_drop)
            st.session_state['preprocessing_steps']["dropped_columns"] = columns_to_drop
            st.success(f"Dropped columns: {', '.join(columns_to_drop)}")
            st.write("Updated Data Preview (After Dropping):", st.session_state['data'].head())

    # 2. Điền giá trị thiếu
    st.write("### Step 2: Fill Missing Values")
    missing_columns = [col for col in st.session_state['data'].columns if st.session_state['data'][col].isnull().sum() > 0]
    if missing_columns:
        for col in missing_columns:
            st.write(f"#### Handling missing values in '{col}'")
            st.write(f"Missing: {st.session_state['data'][col].isnull().sum()} out of {len(st.session_state['data'])} rows")

            if st.session_state['data'][col].dtype in ['int64', 'float64']:
                st.info(f"Suggestion: Use 'median' or 'mean' for numerical data. Median is robust to outliers.")
                fill_method = st.selectbox(
                    f"Choose method for '{col}'",
                    ["Mean", "Median", "Custom Value"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Custom Value":
                    custom_value = st.number_input(f"Enter custom value for '{col}'", key=f"custom_{col}")
            else:
                st.info(f"Suggestion: Use 'mode' for categorical data.")
                fill_method = st.selectbox(
                    f"Choose method for '{col}'",
                    ["Mode", "Custom Value"],
                    key=f"fill_method_{col}"
                )
                if fill_method == "Custom Value":
                    custom_value = st.text_input(f"Enter custom value for '{col}'", key=f"custom_{col}")

            if st.button(f"Fill '{col}'", key=f"fill_{col}"):
                if fill_method == "Mean":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mean())
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "mean"
                elif fill_method == "Median":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].median())
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "median"
                elif fill_method == "Mode":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(st.session_state['data'][col].mode()[0])
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = "mode"
                elif fill_method == "Custom Value":
                    st.session_state['data'][col] = st.session_state['data'][col].fillna(custom_value)
                    st.session_state['preprocessing_steps'][f"{col}_filled"] = f"custom_{custom_value}"

                st.success(f"Filled missing values in '{col}' using {fill_method.lower()}.")
                st.write("Updated Data Preview (After Filling):", st.session_state['data'].head())
    else:
        st.success("No missing values detected in the current dataset.")

    # 3. Chuyển đổi dữ liệu phân loại
    st.write("### Step 3: Convert Categorical Columns")
    categorical_cols = [col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype == 'object']
    if categorical_cols:
        for col in categorical_cols:
            st.write(f"#### Convert '{col}'")
            st.info(f"Suggestion: 'Label Encoding' for ordinal/low cardinality; 'One-Hot Encoding' for nominal/few values.")
            encoding_method = st.selectbox(
                f"Choose encoding method for '{col}'",
                ["Label Encoding", "One-Hot Encoding"],
                key=f"encode_{col}"
            )
            if st.button(f"Apply encoding to '{col}'", key=f"apply_{col}"):
                if encoding_method == "Label Encoding":
                    le = LabelEncoder()
                    st.session_state['data'][col] = le.fit_transform(st.session_state['data'][col])
                    st.session_state['preprocessing_steps'][f"{col}_encoded"] = "label"
                    st.success(f"Applied Label Encoding to '{col}'.")
                elif encoding_method == "One-Hot Encoding":
                    st.session_state['data'] = pd.get_dummies(st.session_state['data'], columns=[col], prefix=col)
                    st.session_state['preprocessing_steps'][f"{col}_encoded"] = "one-hot"
                    st.success(f"Applied One-Hot Encoding to '{col}'.")
                st.write("Updated Data Preview (After Encoding):", st.session_state['data'].head())
    else:
        st.success("No categorical columns available for encoding.")

    # 4. Chuẩn hóa dữ liệu
    st.write("### Step 4: Normalize/Scale Numerical Columns")
    numerical_cols = [col for col in st.session_state['data'].columns if st.session_state['data'][col].dtype in ['int64', 'float64']]
    if numerical_cols:
        st.info("Suggestion: 'Min-Max Scaling' (0-1) for bounded ranges; 'Standard Scaling' (mean=0, std=1) for normal data.")
        scaling_method = st.selectbox(
            "Choose scaling method",
            ["Min-Max Scaling", "Standard Scaling"]
        )
        cols_to_scale = st.multiselect(
            "Select numerical columns to scale",
            options=numerical_cols,
            default=numerical_cols
        )
        if st.button("Apply Scaling"):
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
                st.success(f"Applied {scaling_method} to columns: {', '.join(cols_to_scale)}")
                st.write("Updated Data Preview (After Scaling):", st.session_state['data'].head())
            else:
                st.warning("No columns selected for scaling.")
    else:
        st.success("No numerical columns available for scaling.")

    # 5. Lưu và log dữ liệu đã tiền xử lý
    st.write("### Step 5: Save Processed Data")
    if st.button("Save processed data and log to MLflow"):
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        save_data(st.session_state['data'], processed_file)
        st.success(f"Processed data saved to {processed_file}")

        # Hiển thị dữ liệu cuối cùng trước khi lưu
        st.subheader("Final Processed Data Preview (Before Saving)")
        st.write(st.session_state['data'].head())

        # Log các tham số tiền xử lý vào MLflow
        log_preprocessing_params(st.session_state['preprocessing_steps'])
        st.success(f"Preprocessing steps logged to MLflow under Experiment: '{experiment_name}'!")

        # Xác nhận dữ liệu đã lưu đúng
        saved_data = load_data(processed_file)
        st.write("Verification: Loaded data from saved file matches your preprocessing choices:")
        st.write(saved_data.head())

if __name__ == "__main__":
    preprocess_data()