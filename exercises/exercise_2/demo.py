import streamlit as st
import numpy as np
from PIL import Image
import mlflow
import os
from streamlit_drawable_canvas import st_canvas

# Thiết lập MLflow Tracking URI cục bộ
mlruns_dir = os.path.abspath('mlruns')
if not os.path.exists(mlruns_dir):
    os.makedirs(mlruns_dir)
mlflow.set_tracking_uri(f"file://{mlruns_dir}")

def show_mnist_demo():
    st.header("Demo Nhận diện Chữ số MNIST 🖌️")
    experiment_name = "MNIST_Training"

    # Kiểm tra mô hình từ MLflow
    runs = mlflow.search_runs(experiment_names=[experiment_name])
    if runs.empty:
        st.error("Không tìm thấy mô hình nào trong MLflow. Chạy 'train.py' trước.")
        return

    run_options = {f"{run['tags.mlflow.runName']} (ID: {run['run_id'][:8]})": run['run_id'] for _, run in runs.iterrows()}
    selected_run_name = st.selectbox("Chọn mô hình", list(run_options.keys()))
    selected_run_id = run_options[selected_run_name]

    try:
        model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
        scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
        model_type = runs[runs['run_id'] == selected_run_id]['params.model_type'].iloc[0]
        st.write(f"Mô hình được chọn: {model_type}")
    except Exception as e:
        st.error(f"Không thể tải mô hình/scaler từ MLflow: {str(e)}. Run ID: {selected_run_id}")
        return

    # Vẽ chữ số
    st.subheader("Vẽ chữ số để nhận diện 🖋️")
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

        if st.button("Dự đoán"):
            input_data = image_array.reshape(1, 28 * 28)
            input_data_scaled = scaler.transform(input_data)
            prediction = model.predict(input_data_scaled)[0]

            # Log vào MLflow
            if mlflow.active_run():
                mlflow.end_run()
            with mlflow.start_run(run_name="Prediction", experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
                mlflow.log_param("model_run_id", selected_run_id)
                mlflow.log_param("predicted_digit", prediction)
                with tempfile.NamedTemporaryFile(delete=False, suffix='.npy') as tmp:
                    np.save(tmp.name, image_array)
                    mlflow.log_artifact(tmp.name, "input_image.npy")
                    os.unlink(tmp.name)

                st.success(f"Dự đoán: {prediction} (Run ID: {mlflow.active_run().info.run_id})")

    # Lịch sử dự đoán
    st.subheader("Lịch sử dự đoán")
    pred_runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.mlflow.runName = 'Prediction'")
    if not pred_runs.empty:
        for _, run in pred_runs.iterrows():
            digit = run.data.params.get("predicted_digit", "N/A")
            try:
                image_path = mlflow.artifacts.download_artifacts(run_id=run.run_id, path="input_image.npy")
                image_data = np.load(image_path)
                st.image(image_data, caption=f"Dự đoán: {digit}", width=100)
            except Exception as e:
                st.write(f"Không thể tải hình ảnh cho run {run['run_id'][:8]}: {str(e)}")
    else:
        st.write("Chưa có dự đoán nào được log.")

if __name__ == "__main__":
    show_mnist_demo()