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

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/vn0410.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "22fd02345f8ff45482a20960058627630acaf190"  # Thay bằng token cá nhân của bạn
    DAGSHUB_REPO = "vn0410"
    return DAGSHUB_REPO

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

def get_mlflow_runs():
    """Lấy danh sách tất cả các run từ MLflow tại DAGSHUB_MLFLOW_URI, lọc các run có mô hình."""
    try:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs()
        # Lọc các run có mô hình (giả sử mô hình được log dưới artifact "model" và "scaler")
        runs_with_model = [run for run in runs if client.list_artifacts(run.info.run_id, "model") and client.list_artifacts(run.info.run_id, "scaler")]
        return runs_with_model
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return []

def preprocess_image(image):
    """Chuyển đổi ảnh thành mảng 28x28 pixel (normalized 0-1)."""
    # Chuyển ảnh về kích thước 28x28, grayscale, và normalize
    image = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    image_array = np.array(image) / 255.0  # Normalize về [0, 1]
    return image_array.reshape(1, 28 * 28)

def show_mnist_demo():
    st.header("Dự đoán Chữ số MNIST")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow tại DAGSHUB_MLFLOW_URI
    DAGSHUB_REPO = mlflow_input()

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán (Vẽ hoặc Upload Ảnh)")
        processed_file = "exercises/exercise_mnist/data/processed/mnist_processed.npz"
        try:
            with st.spinner("Đang tải dữ liệu MNIST đã xử lý..."):
                data = load_cached_data(processed_file)
                X_train = data['X_train']
                X_valid = data['X_valid']
                X_test = data['X_test']
        except FileNotFoundError:
            st.error("Dữ liệu MNIST đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý Dữ liệu MNIST' trước.")
            return

        # Lấy danh sách tất cả các run có mô hình từ MLflow tại DAGSHUB_MLFLOW_URI
        st.write("Chọn mô hình đã huấn luyện để dự đoán (từ DagsHub MLflow):")
        runs = get_mlflow_runs()
        if not runs:
            st.error("Không tìm thấy mô hình nào trong MLflow. Vui lòng huấn luyện mô hình trong 'train_mnist.py' trước.")
            return

        run_options = [f"ID Run: {run.info.run_id} - {run.data.tags.get('mlflow.runName', 'Không tên')} (Experiment: {run.info.experiment_id}, Thời gian: {run.info.start_time.strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else 'Không rõ'})" for run in runs]
        selected_run = st.selectbox("Chọn run chứa mô hình", options=run_options)
        selected_run_id = selected_run.split("ID Run: ")[1].split(" - ")[0]

        # Tải mô hình và scaler từ MLflow tại DAGSHUB_MLFLOW_URI
        try:
            with st.spinner("Đang tải mô hình và scaler từ DagsHub MLflow..."):
                model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
            # Khởi tạo hoặc cập nhật trong session_state
            st.session_state['mnist_model'] = model
            if 'mnist_scaler' not in st.session_state:
                st.session_state['mnist_scaler'] = scaler  # Khởi tạo nếu chưa có
            else:
                st.session_state['mnist_scaler'] = scaler  # Cập nhật nếu đã có
            st.write(f"Mô hình và scaler đã được tải từ MLflow (Run ID: {selected_run_id}, Thời gian: {next(run.info.start_time.strftime('%Y-%m-%d %H:%M:%S') for run in runs if run.info.run_id == selected_run_id) if any(run.info.run_id == selected_run_id for run in runs) else 'Không rõ'})")
        except Exception as e:
            st.error(f"Không thể tải mô hình hoặc scaler từ Run ID {selected_run_id} tại DagsHub MLflow: {str(e)}")
            return

        # Kiểm tra và lấy mô hình, scaler từ session_state
        if 'mnist_model' not in st.session_state or 'mnist_scaler' not in st.session_state:
            st.error("Mô hình hoặc scaler không tồn tại trong session. Vui lòng kiểm tra lại huấn luyện mô hình.")
            return

        model = st.session_state['mnist_model']
        scaler = st.session_state['mnist_scaler']

        # Tùy chọn vẽ hoặc upload ảnh
        st.write("Vẽ hoặc upload ảnh chữ số (28x28 pixel, màu đen trên nền trắng):")
        option = st.radio("Chọn phương thức nhập dữ liệu:", ("Vẽ trên canvas", "Upload ảnh"))

        if option == "Vẽ trên canvas":
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=5,
                stroke_color="black",
                background_color="white",
                update_streamlit=True,
                height=280,  # 28x10 cho dễ vẽ
                width=280,   # 28x10 cho dễ vẽ
                drawing_mode="freedraw",
                key="canvas"
            )

            if canvas_result.image_data is not None:
                image = Image.fromarray(canvas_result.image_data).convert('L')
                input_data = preprocess_image(image)
                st.write("Dữ liệu Hình ảnh từ Canvas (28x28 pixel, normalized):")
                st.write(pd.DataFrame(input_data))

                if st.button("Dự đoán từ Canvas"):
                    with st.spinner("Đang thực hiện dự đoán..."):
                        try:
                            X_input_scaled = scaler.transform(input_data)
                            predictions = model.predict(X_input_scaled)
                            st.write("Kết quả Dự đoán:")
                            st.write(f"Chữ số dự đoán: {predictions[0]}")

                            # Logging dự đoán vào MLflow tại DAGSHUB_MLFLOW_URI
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_name = st.text_input("Nhập tên cho run dự đoán này (để trống để tự động tạo)", value="", max_chars=20, key="canvas_run_name_input")
                            if run_name.strip() == "":
                                run_name = f"MNIST_Predict_Canvas_{timestamp.replace(' ', '_').replace(':', '-')}"
                            if st.button("Log Dự đoán từ Canvas vào MLflow"):
                                with st.spinner("Đang log dữ liệu dự đoán vào DagsHub MLflow..."):
                                    with mlflow.start_run(run_name=run_name) as run:
                                        mlflow.log_param("timestamp", timestamp)
                                        mlflow.log_param("run_id", run.info.run_id)
                                        for i in range(28):
                                            for j in range(28):
                                                mlflow.log_param(f"pixel_{i}_{j}", input_data[0, i * 28 + j])
                                        mlflow.log_param("predicted_digit", predictions[0])
                                        mlflow.log_param("demo_name", "MNIST_Demo")
                                        mlflow.log_param("input_method", "Canvas")

                                        st.write("Thông tin Đã Log:")
                                        log_info = {
                                            "Tên Run": run_name,
                                            "ID Run": run.info.run_id,
                                            "Thời gian": timestamp,
                                            "Dữ liệu Hình ảnh": {f"pixel_{i}_{j}": input_data[0, i * 28 + j] for i in range(28) for j in range(28)},
                                            "Chữ số Dự đoán": predictions[0],
                                            "Tên Demo": "MNIST_Demo",
                                            "Phương thức nhập": "Canvas"
                                        }
                                        st.write(log_info)

                                        run_id = run.info.run_id
                                        mlflow_uri = st.session_state['mlflow_url']
                                        st.success(f"Dự đoán đã được log thành công!\n- Experiment: 'MNIST_Demo'\n- Tên Run: '{run_name}'\n- ID Run: {run_id}\n- Thời gian: {timestamp}")
                                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

                        except ValueError as e:
                            st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo ảnh hợp lệ (28x28 pixel, màu đen trên nền trắng).")

        elif option == "Upload ảnh":
            uploaded_file = st.file_uploader("Tải lên ảnh chữ số (PNG/JPG, 28x28 pixel, màu đen trên nền trắng)", type=["png", "jpg", "jpeg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                input_data = preprocess_image(image)
                st.write("Dữ liệu Hình ảnh từ Upload (28x28 pixel, normalized):")
                st.write(pd.DataFrame(input_data))

                if st.button("Dự đoán từ Ảnh Upload"):
                    with st.spinner("Đang thực hiện dự đoán..."):
                        try:
                            X_input_scaled = scaler.transform(input_data)
                            predictions = model.predict(X_input_scaled)
                            st.write("Kết quả Dự đoán:")
                            st.write(f"Chữ số dự đoán: {predictions[0]}")

                            # Logging dự đoán vào MLflow tại DAGSHUB_MLFLOW_URI
                            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            run_name = st.text_input("Nhập tên cho run dự đoán này (để trống để tự động tạo)", value="", max_chars=20, key="upload_run_name_input")
                            if run_name.strip() == "":
                                run_name = f"MNIST_Predict_Upload_{timestamp.replace(' ', '_').replace(':', '-')}"
                            if st.button("Log Dự đoán từ Ảnh Upload vào MLflow"):
                                with st.spinner("Đang log dữ liệu dự đoán vào DagsHub MLflow..."):
                                    with mlflow.start_run(run_name=run_name) as run:
                                        mlflow.log_param("timestamp", timestamp)
                                        mlflow.log_param("run_id", run.info.run_id)
                                        for i in range(28):
                                            for j in range(28):
                                                mlflow.log_param(f"pixel_{i}_{j}", input_data[0, i * 28 + j])
                                        mlflow.log_param("predicted_digit", predictions[0])
                                        mlflow.log_param("demo_name", "MNIST_Demo")
                                        mlflow.log_param("input_method", "Upload")

                                        st.write("Thông tin Đã Log:")
                                        log_info = {
                                            "Tên Run": run_name,
                                            "ID Run": run.info.run_id,
                                            "Thời gian": timestamp,
                                            "Dữ liệu Hình ảnh": {f"pixel_{i}_{j}": input_data[0, i * 28 + j] for i in range(28) for j in range(28)},
                                            "Chữ số Dự đoán": predictions[0],
                                            "Tên Demo": "MNIST_Demo",
                                            "Phương thức nhập": "Upload"
                                        }
                                        st.write(log_info)

                                        run_id = run.info.run_id
                                        mlflow_uri = st.session_state['mlflow_url']
                                        st.success(f"Dự đoán đã được log thành công!\n- Experiment: 'MNIST_Demo'\n- Tên Run: '{run_name}'\n- ID Run: {run_id}\n- Thời gian: {timestamp}")
                                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

                        except ValueError as e:
                            st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo ảnh hợp lệ (28x28 pixel, màu đen trên nền trắng).")

    with tab2:
        st.subheader("Xem Kết quả Đã Log")
        with st.spinner("Đang tải danh sách runs từ DagsHub MLflow..."):
            runs = mlflow.search_runs()
        if runs.empty:
            st.write("Chưa có run dự đoán nào được log.")
        else:
            st.write("Danh sách các run đã log:")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
                columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
            )
            display_runs['Thời gian Bắt đầu'] = display_runs['Thời gian Bắt đầu'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S') if x else 'Không rõ')
            st.write(display_runs)

            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0])
            selected_run_id = st.selectbox(
                "Chọn một run để xem chi tiết",
                options=runs['run_id'].tolist(),
                index=runs['run_id'].tolist().index(default_run) if default_run in runs['run_id'].tolist() else 0
            )
            if selected_run_id:
                with st.spinner("Đang tải chi tiết run từ DagsHub MLflow..."):
                    run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                    st.write("Chi tiết Run:")
                    st.write(f"ID Run: {run_details['run_id']}")
                    st.write(f"Tên Run: {run_details.get('tags.mlflow.runName', 'Không tên')}")
                    st.write(f"ID Experiment: {run_details['experiment_id']}")
                    st.write(f"Thời gian Bắt đầu: {run_details['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run_details['start_time'] else 'Không rõ'}")

                    st.write("Thông số Đã Log:")
                    params = mlflow.get_run(selected_run_id).data.params
                    st.write(params)

                    mlflow_uri = st.session_state['mlflow_url']
                    st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({mlflow_uri})")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        with st.spinner("Đang tải danh sách runs từ DagsHub MLflow..."):
            runs = mlflow.search_runs()
        if runs.empty:
            st.write("Không có run nào để xóa.")
        else:
            st.write("Chọn các run để xóa:")
            run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']}, Thời gian: {run['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run['start_time'] else 'Không rõ'})" 
                          for _, run in runs.iterrows()]
            default_delete = [f"ID Run: {st.session_state['selected_run_id']} - {runs[runs['run_id'] == st.session_state['selected_run_id']]['tags.mlflow.runName'].iloc[0]} (Exp: {runs[runs['run_id'] == st.session_state['selected_run_id']]['experiment_id'].iloc[0]}, Thời gian: {runs[runs['run_id'] == st.session_state['selected_run_id']]['start_time'].strftime('%Y-%m-%d %H:%M:%S') if runs[runs['run_id'] == st.session_state['selected_run_id']]['start_time'].iloc[0] else 'Không rõ'})" 
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