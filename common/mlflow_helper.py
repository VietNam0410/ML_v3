import mlflow
import streamlit as st
from typing import Dict, Any
import json

def log_preprocessing_params(steps: Dict[str, Any], prefix: str = "preprocessing_") -> None:
    """
    Log preprocessing steps as parameters to MLflow with optimization for Streamlit.

    Args:
        steps (Dict[str, Any]): Dictionary containing preprocessing steps to log.
        prefix (str): Prefix to add to parameter names for better organization in MLflow.
                      Defaults to "preprocessing_".

    Returns:
        None

    Notes:
        - Converts complex data types (lists, dicts) to strings efficiently.
        - Includes error handling to prevent crashes in Streamlit.
        - Uses batch logging to minimize MLflow API calls.
    """
    if not steps:
        st.warning("Không có bước tiền xử lý nào để log vào MLflow.")
        return

    try:
        # Chuẩn bị dictionary để log hàng loạt
        params_to_log = {}

        for key, value in steps.items():
            param_key = f"{prefix}{key}"
            
            # Xử lý các loại dữ liệu khác nhau
            if isinstance(value, (list, tuple)):
                # Chuyển list/tuple thành chuỗi ngắn gọn
                params_to_log[param_key] = ",".join(map(str, value))
            elif isinstance(value, dict):
                # Chuyển dict thành JSON string để lưu trữ cấu trúc
                params_to_log[param_key] = json.dumps(value, ensure_ascii=False)
            elif isinstance(value, (int, float, str, bool)) or value is None:
                # Log trực tiếp các kiểu dữ liệu cơ bản
                params_to_log[param_key] = str(value)
            else:
                # Xử lý các kiểu dữ liệu không xác định
                st.warning(f"Không thể log tham số '{key}': Kiểu dữ liệu không được hỗ trợ ({type(value)}). Chuyển thành chuỗi.")
                params_to_log[param_key] = str(value)

        # Log tất cả tham số trong một lần gọi
        if params_to_log:
            mlflow.log_params(params_to_log)
            st.success(f"Đã log {len(params_to_log)} tham số tiền xử lý vào MLflow.")

    except Exception as e:
        st.error(f"Lỗi khi log tham số tiền xử lý vào MLflow: {str(e)}")
        raise

def log_preprocessing_metrics(metrics: Dict[str, float]) -> None:
    """
    Log preprocessing metrics to MLflow.

    Args:
        metrics (Dict[str, float]): Dictionary containing metrics to log (e.g., missing values handled).

    Returns:
        None
    """
    try:
        if metrics:
            mlflow.log_metrics(metrics)
            st.success(f"Đã log {len(metrics)} metric vào MLflow.")
    except Exception as e:
        st.error(f"Lỗi khi log metric vào MLflow: {str(e)}")
        raise

def log_model(model: Any, artifact_path: str = "model", input_example: Any = None) -> None:
    """
    Log a trained model to MLflow with an input example to infer signature.

    Args:
        model (Any): Trained model object (e.g., sklearn model).
        artifact_path (str): Path in MLflow artifact store. Defaults to "model".
        input_example (Any): Example input to infer model signature. Defaults to None.

    Returns:
        None
    """
    try:
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            input_example=input_example
        )
        model_type = type(model).__name__
        st.success(f"Đã log mô hình '{model_type}' vào MLflow tại '{artifact_path}' với input_example.")
    except Exception as e:
        st.error(f"Lỗi khi log mô hình: {str(e)}")
        raise

if __name__ == "__main__":
    # Không có mẫu ví dụ, chỉ để trống
    pass