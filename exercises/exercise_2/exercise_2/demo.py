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
            st.error(f"Không tìm thấy experiment '{experiment_name}' trong MLflow.")
            return []
        experiment_id = experiment.experiment_id
        runs = client.search_runs([experiment_id], order_by=["metrics.valid_accuracy DESC"])
        runs_with_model = [
            run for run in runs 
            if client.list_artifacts(run.info.run_id, "model") and client.list_artifacts(run.info.run_id, "scaler")
        ]
        return runs_with_model
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return []

def preprocess_image(image):
    image = image.resize((28, 28), Image.Resampling.LANCZOS).convert('L')
    image_array = np.array(image)
    threshold = 127
    image_array = (image_array < threshold).astype(np.float32)
    image_array = 1 - image_array  # Đảo ngược để khớp dữ liệu huấn luyện
    return image_array.reshape(1, 28 * 28)

def mnist_demo():
    st.title("Dự đoán Chữ số MNIST 🎨")

    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_URI = mlflow_input()
        if DAGSHUB_URI is None:
            st.error("Không thể kết nối MLflow. Ứng dụng sẽ dừng.")
            return
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_URI

    if mlflow.active_run():
        mlflow.end_run()

    st.header("Bước 1: Chọn Mô hình")
    runs = get_mlflow_runs("MNIST_Training")
    if not runs:
        st.error("Không tìm thấy mô hình nào trong 'MNIST_Training'. Vui lòng huấn luyện mô hình trước.")
        st.info("Chạy file 'train_mnist.py' để tạo mô hình.")
        return

    run_options = [
        f"{run.data.tags.get('mlflow.runName', 'Không tên')} (Độ chính xác: {run.data.metrics.get('valid_accuracy', 0):.4f}, Thời gian: {datetime.datetime.fromtimestamp(run.info.start_time / 1000).strftime('%Y-%m-%d %H:%M:%S') if run.info.start_time else 'Không rõ'}) - ID: {run.info.run_id}"
        for run in runs
    ]
    selected_run = st.selectbox("Chọn mô hình đã huấn luyện", options=run_options, key="model_select")
    selected_run_id = selected_run.split(" - ID: ")[-1]

    model = None
    scaler = None
    with st.spinner(f"Đang tải mô hình và scaler từ MLflow (Run ID: {selected_run_id})..."):
        try:
            model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
            scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
            st.success(f"Đã tải mô hình thành công từ Run ID: {selected_run_id}")
        except Exception as e:
            st.error(f"Không thể tải mô hình hoặc scaler: {str(e)}. Đảm bảo run chứa 'model' và 'scaler'.")
            return

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
            key="canvas"
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
            X_input_scaled = scaler.transform(input_data)
            prediction = model.predict(X_input_scaled)[0]
            confidence = model.predict_proba(X_input_scaled)[0][prediction] if hasattr(model, "predict_proba") else 0.5

            st.header("Bước 3: Kết quả")
            st.write(f"**Chữ số dự đoán**: {prediction}")
            st.write(f"**Độ tin cậy**: {confidence:.2%}")
            st.image(image, caption=f"Dự đoán: {prediction} (Độ tin cậy: {confidence:.2%})", width=280)

            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            run_name = f"Prediction_{input_method.replace(' ', '_')}_{timestamp}"
            with st.spinner("Đang lưu kết quả vào MLflow..."):
                client = mlflow.tracking.MlflowClient()
                experiment = client.get_experiment_by_name("MNIST_Demo")
                experiment_id = experiment.experiment_id if experiment else client.create_experiment("MNIST_Demo")
                with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                    mlflow.log_param("model_run_id", selected_run_id)
                    mlflow.log_param("predicted_digit", prediction)
                    mlflow.log_param("confidence", confidence)
                    mlflow.log_param("input_method", input_method)
                    run_id = run.info.run_id

            st.success(f"Kết quả đã được lưu vào MLflow (Run ID: {run_id})")
            st.markdown(f"Xem chi tiết tại: [{st.session_state['mlflow_url']}]({st.session_state['mlflow_url']})")

if __name__ == "__main__":
    mnist_demo()