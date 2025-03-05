import streamlit as st
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub
import datetime
from io import BytesIO
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import logging

# Tắt cảnh báo từ dagshub
logging.getLogger("dagshub.auth.tokens").setLevel(logging.ERROR)

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

# Hàm tải dữ liệu với cache
@st.cache_data
def load_cached_data(file_path):
    """Tải dữ liệu từ file CSV hoặc NPZ và lưu vào bộ nhớ đệm."""
    if file_path.endswith('.npz'):
        data = np.load(file_path)
        return data
    return load_data(file_path)

def delete_mlflow_run(run_id):
    try:
        with st.spinner(f"Đang xóa Run {run_id}..."):
            mlflow.delete_run(run_id)
        st.success(f"Đã xóa run có ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể xóa run {run_id}: {str(e)}")

def get_mlflow_experiments():
    """Lấy danh sách các experiment từ MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {exp.name: exp.experiment_id for exp in experiments if exp.lifecycle_stage == "active"}
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các experiment từ MLflow: {str(e)}")
        return {}

def get_mlflow_runs(experiment_name: str = "MNIST_Training"):
    """Lấy danh sách các run từ MLflow, lọc các run có mô hình và sắp xếp theo độ chính xác hoặc thời gian."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiment_id = client.get_experiment_by_name(experiment_name).experiment_id
        runs = client.search_runs([experiment_id], order_by=["metrics.valid_accuracy DESC"])
        runs_with_model = [run for run in runs if client.list_artifacts(run.info.run_id, "model") and client.list_artifacts(run.info.run_id, "scaler")]
        return runs_with_model
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return []

def preprocess_image(image):
    """Chuyển đổi và tối ưu hóa ảnh thành mảng 28x28 pixel (normalized 0-1)."""
    # Chuyển ảnh về kích thước 28x28, grayscale
    image = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    
    # Áp dụng threshold để làm mịn (chuyển thành đen trắng rõ ràng hơn)
    image_array = np.array(image)
    threshold = 127  # Ngưỡng để phân biệt đen và trắng
    image_array = (image_array < threshold).astype(np.float32)  # 0 cho trắng, 1 cho đen
    image_array = 1 - image_array  # Đảo ngược (đen = 1, trắng = 0) để khớp với dữ liệu huấn luyện
    return image_array.reshape(1, 28 * 28)

def show_mnist_demo():
    st.header("Dự đoán Chữ số MNIST")

    # Đóng bất kỳ run nào đang hoạt động
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Khởi tạo DagsHub/MLflow chỉ một lần
    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_REPO = mlflow_input()
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_REPO
    else:
        DAGSHUB_REPO = st.session_state['mlflow_url']

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Chọn Mô hình và Dự đoán Chữ số")
        processed_file = "exercises/exercise_mnist/data/processed/mnist_processed.npz"
        try:
            with st.spinner("Đang tải dữ liệu MNIST đã xử lý..."):
                data = load_cached_data(processed_file)
                X_train = data['X_train']
                X_valid = data['X_valid']
                X_test = data['X_test']
        except FileNotFoundError:
            st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trước.")
            return

        # Lấy danh sách run từ experiment "MNIST_Training", ưu tiên run có valid_accuracy cao nhất
        st.write("Chọn mô hình đã huấn luyện từ experiment 'MNIST_Training':")
        runs = get_mlflow_runs("MNIST_Training")
        if not runs:
            st.error("Không tìm thấy mô hình nào trong MLflow. Vui lòng huấn luyện mô hình trong 'train_mnist.py' trước.")
            return

        run_options = [f"ID Run: {run.info.run_id} - {run.data.tags.get('mlflow.runName', 'Không tên')} (Độ chính xác Valid: {run.data.metrics.get('valid_accuracy', 0):.4f}, Thời gian: {run.info.start_time.strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else 'Không rõ'})" for run in runs]
        selected_run = st.selectbox("Chọn run chứa mô hình tốt nhất", options=run_options, key="model_select")
        selected_run_id = selected_run.split("ID Run: ")[1].split(" - ")[0]

        # Tải mô hình và scaler từ MLflow
        try:
            with st.spinner("Đang tải mô hình và scaler từ DagsHub MLflow..."):
                model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
            st.session_state['mnist_model'] = model
            st.session_state['mnist_scaler'] = scaler
            st.success(f"Mô hình và scaler đã được tải từ MLflow (Run ID: {selected_run_id})")
        except Exception as e:
            st.error(f"Không thể tải mô hình hoặc scaler từ Run ID {selected_run_id}: {str(e)}")
            return

        model = st.session_state['mnist_model']
        scaler = st.session_state['mnist_scaler']

        # Tùy chọn vẽ hoặc upload ảnh
        st.subheader("Bước 2: Vẽ hoặc Tải lên Hình Chữ số (28x28 pixel, màu đen trên nền trắng)")
        option = st.radio("Chọn phương thức nhập dữ liệu:", ("Vẽ trên canvas", "Upload ảnh"), key="input_method")

        if option == "Vẽ trên canvas":
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=5,
                stroke_color="black",
                background_color="white",
                update_streamlit=True,
                height=280,
                width=280,
                drawing_mode="freedraw",
                key="canvas"
            )

            if canvas_result.image_data is not None:
                image = Image.fromarray(canvas_result.image_data).convert('L')
                input_data = preprocess_image(image)
                st.write("Dữ liệu Hình ảnh từ Canvas (28x28 pixel, normalized):")
                st.write(pd.DataFrame(input_data))

                if st.button("Dự đoán từ Canvas", key="predict_canvas"):
                    with st.spinner("Đang thực hiện dự đoán..."):
                        try:
                            X_input_scaled = scaler.transform(input_data)
                            probabilities = model.predict_proba(X_input_scaled) if hasattr(model, 'predict_proba') else None
                            predictions = model.predict(X_input_scaled)
                            predicted_digit = predictions[0]
                            confidence = probabilities[0][predicted_digit] if probabilities is not None else 0.5

                            st.write("### Kết quả Dự đoán")
                            st.write(f"Chữ số dự đoán: **{predicted_digit}**")
                            st.write(f"Độ tin cậy: **{confidence:.2%}**")

                            # Logging dự đoán vào MLflow experiment "MNIST_Demo"
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_name = st.text_input("Nhập tên cho run dự đoán này (để trống để tự động tạo)", value="", max_chars=20, key="canvas_run_name")
                            if not run_name.strip():
                                run_name = f"MNIST_Predict_Canvas_{timestamp.replace(' ', '_').replace(':', '-')}"
                            if st.button("Log Dự đoán từ Canvas vào MLflow", key="log_canvas"):
                                with st.spinner("Đang log dữ liệu dự đoán vào DagsHub MLflow..."):
                                    experiment_name = "MNIST_Demo"
                                    client = mlflow.tracking.MlflowClient()
                                    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id if client.get_experiment_by_name(experiment_name) else client.create_experiment(experiment_name)
                                    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                                        mlflow.log_param("timestamp", timestamp)
                                        mlflow.log_param("run_id", run.info.run_id)
                                        mlflow.log_param("model_run_id", selected_run_id)
                                        mlflow.log_param("predicted_digit", predicted_digit)
                                        mlflow.log_param("confidence", confidence)
                                        mlflow.log_param("input_method", "Canvas")
                                        for i in range(28):
                                            for j in range(28):
                                                mlflow.log_param(f"pixel_{i}_{j}", input_data[0, i * 28 + j])

                                        st.write("### Thông tin Đã Log")
                                        log_info = {
                                            "Tên Run": run_name,
                                            "ID Run": run.info.run_id,
                                            "Thời gian": timestamp,
                                            "Mô hình Nguồn": selected_run_id,
                                            "Chữ số Dự đoán": predicted_digit,
                                            "Độ Tin Cậy": confidence,
                                            "Phương thức nhập": "Canvas",
                                            "Dữ liệu Hình ảnh": {f"pixel_{i}_{j}": input_data[0, i * 28 + j] for i in range(28) for j in range(28)}
                                        }
                                        st.write(log_info)

                                        run_id = run.info.run_id
                                        mlflow_uri = st.session_state['mlflow_url']
                                        st.success(f"Dự đoán đã được log thành công vào experiment 'MNIST_Demo'!\n- Tên Run: '{run_name}'\n- ID Run: {run_id}\n- Thời gian: {timestamp}")
                                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

                        except ValueError as e:
                            st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo ảnh hợp lệ (28x28 pixel, màu đen trên nền trắng).")

        elif option == "Upload ảnh":
            uploaded_file = st.file_uploader("Tải lên ảnh chữ số (PNG/JPG, 28x28 pixel, màu đen trên nền trắng)", type=["png", "jpg", "jpeg"], key="upload")
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                input_data = preprocess_image(image)
                st.write("Dữ liệu Hình ảnh từ Upload (28x28 pixel, normalized):")
                st.write(pd.DataFrame(input_data))

                if st.button("Dự đoán từ Ảnh Upload", key="predict_upload"):
                    with st.spinner("Đang thực hiện dự đoán..."):
                        try:
                            X_input_scaled = scaler.transform(input_data)
                            probabilities = model.predict_proba(X_input_scaled) if hasattr(model, 'predict_proba') else None
                            predictions = model.predict(X_input_scaled)
                            predicted_digit = predictions[0]
                            confidence = probabilities[0][predicted_digit] if probabilities is not None else 0.5

                            st.write("### Kết quả Dự đoán")
                            st.write(f"Chữ số dự đoán: **{predicted_digit}**")
                            st.write(f"Độ tin cậy: **{confidence:.2%}**")

                            # Logging dự đoán vào MLflow experiment "MNIST_Demo"
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_name = st.text_input("Nhập tên cho run dự đoán này (để trống để tự động tạo)", value="", max_chars=20, key="upload_run_name")
                            if not run_name.strip():
                                run_name = f"MNIST_Predict_Upload_{timestamp.replace(' ', '_').replace(':', '-')}"
                            if st.button("Log Dự đoán từ Ảnh Upload vào MLflow", key="log_upload"):
                                with st.spinner("Đang log dữ liệu dự đoán vào DagsHub MLflow..."):
                                    experiment_name = "MNIST_Demo"
                                    client = mlflow.tracking.MlflowClient()
                                    experiment_id = client.get_experiment_by_name(experiment_name).experiment_id if client.get_experiment_by_name(experiment_name) else client.create_experiment(experiment_name)
                                    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                                        mlflow.log_param("timestamp", timestamp)
                                        mlflow.log_param("run_id", run.info.run_id)
                                        mlflow.log_param("model_run_id", selected_run_id)
                                        mlflow.log_param("predicted_digit", predicted_digit)
                                        mlflow.log_param("confidence", confidence)
                                        mlflow.log_param("input_method", "Upload")
                                        for i in range(28):
                                            for j in range(28):
                                                mlflow.log_param(f"pixel_{i}_{j}", input_data[0, i * 28 + j])

                                        st.write("### Thông tin Đã Log")
                                        log_info = {
                                            "Tên Run": run_name,
                                            "ID Run": run.info.run_id,
                                            "Thời gian": timestamp,
                                            "Mô hình Nguồn": selected_run_id,
                                            "Chữ số Dự đoán": predicted_digit,
                                            "Độ Tin Cậy": confidence,
                                            "Phương thức nhập": "Upload",
                                            "Dữ liệu Hình ảnh": {f"pixel_{i}_{j}": input_data[0, i * 28 + j] for i in range(28) for j in range(28)}
                                        }
                                        st.write(log_info)

                                        run_id = run.info.run_id
                                        mlflow_uri = st.session_state['mlflow_url']
                                        st.success(f"Dự đoán đã được log thành công vào experiment 'MNIST_Demo'!\n- Tên Run: '{run_name}'\n- ID Run: {run_id}\n- Thời gian: {timestamp}")
                                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

                        except ValueError as e:
                            st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo ảnh hợp lệ (28x28 pixel, màu đen trên nền trắng).")

    with tab2:
        st.subheader("Xem Kết quả Đã Log từ Experiment 'MNIST_Demo'")
        with st.spinner("Đang tải danh sách runs từ experiment 'MNIST_Demo' trên DagsHub MLflow..."):
            experiments = get_mlflow_experiments()
            if "MNIST_Demo" not in experiments:
                st.warning("Experiment 'MNIST_Demo' chưa được tạo. Vui lòng thực hiện dự đoán và log để tạo experiment.")
                return

            experiment_id = experiments["MNIST_Demo"]
            runs = mlflow.search_runs([experiment_id], order_by=["start_time DESC"])
        if runs.empty:
            st.write("Chưa có run dự đoán nào được log trong experiment 'MNIST_Demo'.")
        else:
            st.write("Danh sách các run đã log trong Experiment 'MNIST_Demo':")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
                columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
            )
            display_runs['Thời gian Bắt đầu'] = display_runs['Thời gian Bắt đầu'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else 'Không rõ')
            st.write(display_runs)

            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0] if not runs.empty else None)
            if default_run and default_run in runs['run_id'].tolist():
                default_index = runs['run_id'].tolist().index(default_run)
            else:
                default_index = 0 if not runs.empty else None

            if not runs.empty:
                selected_run_id = st.selectbox(
                    "Chọn một run để xem chi tiết",
                    options=runs['run_id'].tolist(),
                    index=default_index,
                    format_func=lambda x: f"ID Run: {x} - {runs[runs['run_id'] == x]['tags.mlflow.runName'].iloc[0] if not runs[runs['run_id'] == x]['tags.mlflow.runName'].empty else 'Không tên'}"
                )
                if selected_run_id:
                    with st.spinner("Đang tải chi tiết run từ DagsHub MLflow..."):
                        run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                        st.write("### Chi tiết Run:")
                        st.write(f"ID Run: {run_details['run_id']}")
                        st.write(f"Tên Run: {run_details.get('tags.mlflow.runName', 'Không tên')}")
                        st.write(f"Thời gian Bắt đầu: {run_details['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run_details['start_time'] else 'Không rõ'}")

                        st.write("### Thông số Đã Log:")
                        params = mlflow.get_run(selected_run_id).data.params
                        st.write(params)

                        mlflow_uri = st.session_state['mlflow_url']
                        st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({mlflow_uri})")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết từ Experiment 'MNIST_Demo'")
        with st.spinner("Đang tải danh sách runs từ experiment 'MNIST_Demo' trên DagsHub MLflow..."):
            experiments = get_mlflow_experiments()
            if "MNIST_Demo" not in experiments:
                st.warning("Experiment 'MNIST_Demo' chưa được tạo. Không có run nào để xóa.")
                return

            experiment_id = experiments["MNIST_Demo"]
            runs = mlflow.search_runs([experiment_id])
        if runs.empty:
            st.write("Không có run nào để xóa trong experiment 'MNIST_Demo'.")
        else:
            st.write("Chọn các run để xóa trong Experiment 'MNIST_Demo':")
            run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Thời gian: {run['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run['start_time'] else 'Không rõ'})" 
                          for _, run in runs.iterrows()]
            default_delete = [f"ID Run: {st.session_state['selected_run_id']} - {runs[runs['run_id'] == st.session_state['selected_run_id']]['tags.mlflow.runName'].iloc[0]} (Thời gian: {runs[runs['run_id'] == st.session_state['selected_run_id']]['start_time'].strftime('%Y-%m-%d %H:%M:%S') if runs[runs['run_id'] == st.session_state['selected_run_id']]['start_time'].iloc[0] else 'Không rõ'})" 
                             if 'selected_run_id' in st.session_state and st.session_state['selected_run_id'] in runs['run_id'].tolist() else None]
            runs_to_delete = st.multiselect(
                "Chọn các run",
                options=run_options,
                default=[d for d in default_delete if d],
                key="delete_runs"
            )
            if st.button("Xóa Các Run Đã Chọn"):
                for run_str in runs_to_delete:
                    run_id = run_str.split("ID Run: ")[1].split(" - ")[0]
                    try:
                        with st.spinner(f"Đang xóa Run {run_id}..."):
                            mlflow.delete_run(run_id)
                        st.success(f"Đã xóa run có ID: {run_id}")
                    except mlflow.exceptions.MlflowException as e:
                        st.error(f"Không thể xóa run {run_id}: {str(e)}")
                st.success("Các run đã chọn đã được xóa. Làm mới trang để cập nhật danh sách.")

if __name__ == "__main__":
    show_mnist_demo()