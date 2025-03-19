import streamlit as st
import numpy as np
import mlflow
import mlflow.sklearn
import os
import datetime
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import logging

# Tắt log không cần thiết
logging.getLogger("mlflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Tắt log TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Chạy trên CPU

def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        if experiments:
            st.success('Kết nối MLflow với DagsHub thành công! ✅')
        else:
            st.warning('Không tìm thấy experiment nào, nhưng kết nối MLflow vẫn hoạt động.')
        return DAGSHUB_MLFLOW_URI
    except mlflow.exceptions.MlflowException as e:
        st.error(f'Lỗi xác thực MLflow: {str(e)}. Vui lòng kiểm tra token tại https://dagshub.com/user/settings/tokens.')
        return None

def get_mlflow_runs(experiment_name: str = "MNIST_Training"):
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name(experiment_name)
        if not experiment:
            st.error(f"Không tìm thấy experiment '{experiment_name}' trong MLflow. Vui lòng kiểm tra lại.")
            return []
        experiment_id = experiment.experiment_id
        
        # Lấy tất cả các run
        runs = client.search_runs([experiment_id], order_by=["start_time DESC"])
        
        # Lọc các run có chứa model
        valid_runs = [
            run for run in runs 
            if client.list_artifacts(run.info.run_id, "model")
        ]
        
        if not valid_runs:
            st.error("Không tìm thấy run nào trong experiment 'MNIST_Training' chứa 'model'. Vui lòng kiểm tra lại MLflow.")
        return valid_runs
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return []

def load_model_from_mlflow(run_id: str):
    try:
        model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
        return model
    except Exception as e:
        st.error(f"Không thể tải mô hình từ Run ID {run_id}: {str(e)}.")
        return None

def preprocess_image(image):
    image = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    image_array = np.array(image)
    threshold = 127
    image_array = (image_array < threshold).astype(np.float32)
    image_array = 1 - image_array  # Đảo ngược để khớp dữ liệu huấn luyện
    return image_array.reshape(1, 28 * 28)

def mnist_demo():
    st.title("Dự đoán Chữ số MNIST 🎨")

    # Kiểm tra kết nối MLflow
    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_URI = mlflow_input()
        if DAGSHUB_URI is None:
            st.error("Không thể kết nối MLflow. Ứng dụng sẽ dừng.")
            return
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_URI

    # Đảm bảo không có active run trước khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()

    st.header("Bước 1: Chọn Mô hình")
    runs = get_mlflow_runs("MNIST_Training")
    if not runs:
        return

    # Hiển thị danh sách run: sử dụng run_name và thời gian
    run_options = [
        f"{run.data.tags.get('mlflow.runName', 'Không tên')} (Thời gian: {datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else 'Không rõ'})"
        for run in runs
    ]
    run_id_map = {option: run.info.run_id for option, run in zip(run_options, runs)}
    
    selected_run = st.selectbox("Chọn mô hình đã huấn luyện", options=run_options, key="model_select")
    selected_run_id = run_id_map[selected_run]

    # Tải mô hình
    model = None
    with st.spinner(f"Đang tải mô hình từ MLflow (Run: {selected_run})..."):
        model = load_model_from_mlflow(selected_run_id)
        if model is None:
            return
        st.success(f"Đã tải mô hình thành công từ Run: {selected_run}")

    st.header("Bước 2: Nhập Chữ số")
    input_method = st.radio("Chọn cách nhập dữ liệu:", ["Vẽ trên canvas", "Tải lên ảnh"], key="input_method")

    input_data = None
    image = None
    if input_method == "Vẽ trên canvas":
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="black",
            background_color="white",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            # Không có tùy chọn xóa bảng
            update_streamlit=True
        )
        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data)
            input_data = preprocess_image(image)
            st.image(image, caption="Hình ảnh bạn vẽ", width=280)

    else:
        uploaded_file = st.file_uploader("Tải lên ảnh (PNG/JPG, 28x28, đen trên nền trắng)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            input_data = preprocess_image(image)
            st.image(image, caption="Hình ảnh đã tải lên", width=280)

    if input_data is not None and st.button("Dự đoán", key="predict_button"):
        with st.spinner("Đang dự đoán..."):
            try:
                # Bỏ bước sử dụng scaler, dự đoán trực tiếp với mô hình
                prediction = model.predict(input_data)[0]
                confidence = model.predict_proba(input_data)[0][prediction] if hasattr(model, "predict_proba") else 0.5

                st.header("Bước 3: Kết quả")
                st.write(f"**Chữ số dự đoán**: {prediction}")
                st.write(f"**Độ tin cậy**: {confidence:.2%}")
                st.image(image, caption=f"Dự đoán: {prediction} (Độ tin cậy: {confidence:.2%})", width=280)

                # Lưu kết quả vào MLflow
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                run_name = f"Prediction_{input_method.replace(' ', '_')}_{timestamp}"
                with st.spinner("Đang lưu kết quả vào MLflow..."):
                    client = mlflow.tracking.MlflowClient()
                    experiment = client.get_experiment_by_name("MNIST_Demo")
                    experiment_id = experiment.experiment_id if experiment else client.create_experiment("MNIST_Demo")
                    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                        # Lưu các tham số
                        mlflow.log_param("model_run_id", selected_run_id)
                        mlflow.log_param("predicted_digit", prediction)
                        mlflow.log_param("confidence", confidence)
                        mlflow.log_param("input_method", input_method)

                        # Lưu hình ảnh đầu vào dưới dạng artifact
                        image_path = f"input_image_{timestamp}.png"
                        image.save(image_path)
                        mlflow.log_artifact(image_path, "input_images")
                        os.remove(image_path)  # Xóa file tạm sau khi lưu vào MLflow

                        run_id = run.info.run_id

                st.success(f"Kết quả đã được lưu vào MLflow (Run ID: {run_id})")
                st.markdown(f"Xem chi tiết tại: [{st.session_state['mlflow_url']}]({st.session_state['mlflow_url']})")

            except Exception as e:
                st.error(f"Lỗi trong quá trình dự đoán: {str(e)}")
                return
