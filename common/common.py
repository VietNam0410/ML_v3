import streamlit as st
import pandas as pd
import numpy as np
import os
import logging
import datetime
import requests
from typing import Optional, Any

# Thiết lập logging để debug nếu cần
logging.getLogger("streamlit").setLevel(logging.INFO)

# Hàm kiểm tra kết nối mạng
def check_network_connection() -> bool:
    """
    Kiểm tra kết nối internet bằng cách gửi yêu cầu đến Google.
    Trả về True nếu kết nối thành công, ngược lại hiển thị lỗi và trả về False.
    """
    try:
        requests.get("https://www.google.com", timeout=5)
        return True
    except requests.RequestException:
        st.error("Không thể kết nối đến mạng. Vui lòng kiểm tra kết nối internet.")
        return False

# Hàm kiểm tra kết nối MLflow
def check_mlflow_connection(client: Any) -> bool:
    """
    Kiểm tra kết nối đến MLflow.
    Trả về True nếu kết nối thành công, ngược lại hiển thị lỗi và trả về False.
    """
    try:
        client.list_experiments()
        return True
    except Exception as e:
        st.error(f"Lỗi kết nối MLflow: {str(e)}")
        return False

# Hàm hiển thị dữ liệu với xử lý lỗi và giới hạn tài nguyên
@st.cache_data
def display_data(df: Optional[pd.DataFrame], max_display: int = 50, width: int = 800) -> None:
    """
    Hiển thị DataFrame với giới hạn số hàng và xử lý lỗi.
    """
    try:
        if df is not None:
            # Giới hạn số hàng hiển thị để tránh crash với dữ liệu lớn
            df_display = df.head(max_display)
            st.dataframe(df_display, hide_index=True, width=width)
            st.write(f"Hiển thị {len(df_display)} hàng đầu tiên (tổng {len(df)} hàng).")
        else:
            st.warning("Không có dữ liệu để hiển thị.")
    except Exception as e:
        st.error(f"Lỗi khi hiển thị dữ liệu: {str(e)}")

# Hàm xóa cache
def clear_cache() -> None:
    """
    Xóa cache của Streamlit và hiển thị thông báo thành công.
    """
    st.cache_data.clear()
    st.success("Đã làm mới dữ liệu và xóa cache!")

# Hàm tạo key duy nhất cho các phần tử Streamlit
def generate_unique_key(prefix: str = "element") -> str:
    """
    Tạo một key duy nhất dựa trên thời gian vi giây.
    """
    return f"{prefix}_{datetime.datetime.now().microsecond}"

# Hàm xử lý lỗi chung
def handle_error(error_msg: str, action: str = "tiếp tục") -> None:
    """
    Hiển thị thông báo lỗi và xử lý theo hành động (dừng hoặc tiếp tục).
    """
    st.error(f"Lỗi: {error_msg}")
    if action == "dừng":
        st.stop()
    else:
        st.write("Ứng dụng sẽ tiếp tục chạy, nhưng một số chức năng có thể bị ảnh hưởng.")

# Hàm giới hạn tài nguyên (mẫu, có thể tùy chỉnh)
def load_limited_data(data_source: Any, max_items: int = 100) -> Optional[pd.DataFrame]:
    """
    Tải dữ liệu với giới hạn số lượng để tránh crash.
    Thay data_source bằng logic thực tế của bạn (ví dụ: MLflow, file, v.v.).
    """
    try:
        if isinstance(data_source, pd.DataFrame):
            return data_source.head(max_items)
        elif isinstance(data_source, list):
            return pd.DataFrame(data_source[:max_items])
        else:
            # Tạo dữ liệu mẫu nếu không có nguồn cụ thể
            data = {
                "ID": np.arange(max_items),
                "Value": np.random.rand(max_items),
                "Category": np.random.choice(["A", "B", "C"], max_items)
            }
            return pd.DataFrame(data)
    except Exception as e:
        handle_error(f"Khi tải dữ liệu: {str(e)}")
        return None

# Hàm cấu hình MLflow/DagsHub (mẫu, có thể tùy chỉnh)
def configure_mlflow() -> tuple[str, MlflowClient]:
    """
    Cấu hình MLflow và DagsHub, trả về URI và client.
    """
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

    client = MlflowClient()
    return DAGSHUB_MLFLOW_URI, client

# Hàm hiển thị thông báo trạng thái
def show_status(message: str, spinner: bool = False) -> None:
    """
    Hiển thị thông báo trạng thái với hoặc không có spinner.
    """
    if spinner:
        with st.spinner(message):
            st.write(message)
    else:
        st.write(message)