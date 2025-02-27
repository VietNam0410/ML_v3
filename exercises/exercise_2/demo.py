import streamlit as st
import numpy as np
from PIL import Image
import mlflow
import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from streamlit_drawable_canvas import st_canvas

# Thiết lập MLflow Tracking URI cục bộ
mlruns_dir = os.path.abspath('mlruns')
if not os.path.exists(mlruns_dir):
    os.makedirs(mlruns_dir)
mlflow.set_tracking_uri(f"file://{mlruns_dir}")

def show_mnist_demo():
    st.header("Demo Nhận diện Chữ số MNIST 🖌️")
    experiment_name = "MNIST_Training"

    # Kiểm tra dữ liệu và mô hình từ session hoặc MLflow
    if 'mnist_model' not in st.session_state or st.session_state['mnist_model'] is None:
        # Nếu không có trong session, kiểm tra trong MLflow
        runs = mlflow.search_runs(experiment_names=[experiment_name])
        if runs.empty:
            st.error("Không tìm thấy mô hình nào trong MLflow. Vui lòng chạy 'train.py' trước.")
            return

        run_options = {f"{run['tags.mlflow.runName']} (ID: {run['run_id'][:8]})": run['run_id'] for _, run in runs.iterrows()}
        selected_run_name = st.selectbox("Chọn mô hình từ MLflow", list(run_options.keys()))
        selected_run_id = run_options[selected_run_name]

        try:
            model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
            scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
            model_type = runs[runs['run_id'] == selected_run_id]['params.model_type'].iloc[0]
            st.write(f"Mô hình được chọn: {model_type}")
        except Exception as e:
            st.error(f"Không thể tải mô hình/scaler từ MLflow: {str(e)}. Run ID: {selected_run_id}")
            return
    else:
        # Sử dụng mô hình và scaler từ session (từ train_mnist.py)
        model = st.session_state['mnist_model']
        scaler = st.session_state['scaler']
        model_type = "SVM (Support Vector Machine)" if isinstance(model, SVC) else "Decision Tree"
        st.write(f"Mô hình được chọn: {model_type} (từ session)")

    # Tùy chọn vẽ hoặc upload ảnh
    st.subheader("Vẽ hoặc Tải ảnh để nhận diện 🖋️")
    input_type = st.radio("Chọn phương thức nhập:", ["Vẽ chữ số", "Tải ảnh"])

    if input_type == "Vẽ chữ số":
        # Vẽ chữ số trên canvas
        canvas_result = st_canvas(
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas"
        )

        if st.button("Xóa canvas"):
            st.rerun()

        if canvas_result.image_data is not None:
            image = Image.fromarray(canvas_result.image_data).convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0
            st.image(image_array, caption="Hình ảnh đã vẽ", width=100)

            # Đặt tên cho lần thử nghiệm
            run_name = st.text_input("Nhập tên cho lần thử nghiệm này", value=f"Prediction_Draw_{st.session_state.get('prediction_count', 0) + 1}")

            if st.button("Dự đoán"):
                input_data = image_array.reshape(1, 28 * 28)
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)[0]

                # Log vào MLflow
                if mlflow.active_run():
                    mlflow.end_run()
                with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
                    mlflow.log_param("model_run_id", selected_run_id if 'selected_run_id' in locals() else "From_Session")
                    mlflow.log_param("predicted_digit", prediction)
                    st.success(f"Dự đoán: {prediction} (Run ID: {mlflow.active_run().info.run_id}, Tên Run: {run_name})")

                # Cập nhật số lần thử nghiệm
                if 'prediction_count' not in st.session_state:
                    st.session_state['prediction_count'] = 0
                st.session_state['prediction_count'] += 1

    else:  # Tải ảnh
        uploaded_file = st.file_uploader("Tải lên ảnh chữ số (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0
            st.image(image_array, caption="Hình ảnh đã tải", width=100)

            # Đặt tên cho lần thử nghiệm
            run_name = st.text_input("Nhập tên cho lần thử nghiệm này", value=f"Prediction_Upload_{st.session_state.get('prediction_count', 0) + 1}")

            if st.button("Dự đoán từ ảnh"):
                input_data = image_array.reshape(1, 28 * 28)
                input_data_scaled = scaler.transform(input_data)
                prediction = model.predict(input_data_scaled)[0]

                # Log vào MLflow
                if mlflow.active_run():
                    mlflow.end_run()
                with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
                    mlflow.log_param("model_run_id", selected_run_id if 'selected_run_id' in locals() else "From_Session")
                    mlflow.log_param("predicted_digit", prediction)
                    st.success(f"Dự đoán: {prediction} (Run ID: {mlflow.active_run().info.run_id}, Tên Run: {run_name})")

                # Cập nhật số lần thử nghiệm
                if 'prediction_count' not in st.session_state:
                    st.session_state['prediction_count'] = 0
                st.session_state['prediction_count'] += 1

    # Lịch sử dự đoán
    st.subheader("Lịch sử dự đoán")
    pred_runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.mlflow.runName = 'Prediction'")
    if not pred_runs.empty:
        for _, run in pred_runs.iterrows():
            # Lấy thông tin run chi tiết từ mlflow.get_run
            run_id = run['run_id']
            run_details = mlflow.get_run(run_id)
            params = run_details.data.params if run_details.data.params else {}
            digit = params.get("predicted_digit", "N/A")
            run_name = run_details.data.tags.get("mlflow.runName", "N/A")
            try:
                image_path = mlflow.artifacts.download_artifacts(run_id=run_id, path="input_image.npy")
                image_data = np.load(image_path)
                st.image(image_data, caption=f"Dự đoán: {digit} (Tên Run: {run_name})", width=100)
            except Exception as e:
                st.write(f"Không thể tải hình ảnh cho run {run_id[:8]} (Tên Run: {run_name}): {str(e)}")
    else:
        st.write("Chưa có dự đoán nào được log.")

if __name__ == "__main__":
    show_mnist_demo()