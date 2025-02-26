import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import mlflow
import os
from streamlit_drawable_canvas import st_canvas
from sklearn.preprocessing import StandardScaler

# Thiết lập MLflow Tracking URI cục bộ
if not os.path.exists('mlruns'):
    os.makedirs('mlruns')
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def show_mnist_demo():
    st.header("Demo Nhận diện Chữ số Viết Tay MNIST 🖌️")

    # Kiểm tra mô hình đã huấn luyện trong MLflow
    experiment_name = "MNIST_Training"  # Giả định tên experiment từ train.py
    client = mlflow.tracking.MlflowClient()
    runs = mlflow.search_runs(experiment_names=[experiment_name])

    if runs.empty:
        st.error("Không tìm thấy mô hình đã huấn luyện trong MLflow. Vui lòng huấn luyện mô hình trong 'Huấn luyện Mô hình Nhận diện Chữ số MNIST' trước.")
        return

    # Cho người dùng chọn run (mô hình) để dự đoán
    # Sửa đổi bởi Grok 3: Loại bỏ session_state, chỉ dùng MLflow
    run_options = {f"Run {run['run_id']} - {run.get('tags.mlflow.runName', 'No Name')}": run['run_id'] for _, run in runs.iterrows()}
    selected_run_name = st.selectbox("Chọn run (mô hình) để dự đoán", options=list(run_options.keys()))
    selected_run_id = run_options[selected_run_name]

    try:
        model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
        # Load scaler từ MLflow (nếu có)
        try:
            scaler = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/scaler")
        except Exception:
            scaler = StandardScaler()  # Fallback nếu không có scaler trong MLflow
        # Load model_choice từ params trong MLflow
        run = client.get_run(selected_run_id)
        model_choice = run.data.params.get("model_type", "SVM (Support Vector Machine)")  # Mặc định là SVM nếu không tìm thấy
    except Exception as e:
        st.error(f"Không thể tải mô hình hoặc thông tin từ MLflow. Lỗi: {str(e)}. Vui lòng kiểm tra MLflow hoặc huấn luyện lại mô hình.")
        return

    # Hiển thị mô hình được chọn để dự đoán
    # Sửa đổi bởi Grok 3: Hiển thị và log model_choice
    st.write(f"Mô hình được sử dụng để dự đoán: {model_choice}")

    # Tạo giao diện cho người dùng vẽ chữ số bằng streamlit-drawable-canvas
    st.subheader("Vẽ một chữ số để nhận diện 🖋️")
    drawing_mode = st.checkbox("Bật chế độ vẽ", value=True)
    if drawing_mode:
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0.0)",  # Màu nền trong suốt
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            width=280,
            height=280,
            drawing_mode="freedraw" if drawing_mode else None,
            key="canvas"
        )

        if canvas_result.image_data is not None:
            # Chuyển đổi hình ảnh từ canvas thành mảng numpy bằng PIL
            image = Image.fromarray(canvas_result.image_data).convert('L')  # Chuyển thành grayscale
            image = image.resize((28, 28))  # Thay đổi kích thước về 28x28
            image = np.array(image) / 255.0  # Chuẩn hóa [0, 1]

            # Hiển thị hình ảnh đã vẽ
            st.image(image, caption="Hình ảnh đã vẽ (28x28)", width=100)

            # Dự đoán
            if st.button("Nhận diện chữ số"):
                # Chuẩn bị dữ liệu cho mô hình (Flatten thành vector 784 chiều)
                input_data = image.reshape(1, 28 * 28)
                input_data_scaled = scaler.fit_transform(input_data) if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else input_data

                prediction = model.predict(input_data_scaled)
                predicted_digit = np.argmax(prediction) if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None
                confidence = np.max(prediction) * 100 if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None

                if predicted_digit is not None and confidence is not None:
                    st.write(f"Dự đoán: {predicted_digit}")
                    st.write(f"Độ tin cậy: {confidence:.2f}%")
                else:
                    st.write("Mô hình này là phân cụm (K-means/DBSCAN), không dự đoán nhãn. Vui lòng sử dụng SVM hoặc Decision Tree.")

                # Kiểm tra và kết thúc run hiện tại nếu có
                # Sửa đổi bởi Grok 3: Thêm log dự đoán và model_choice vào MLflow
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.end_run()

                with mlflow.start_run(run_name=f"MNIST_Prediction_{experiment_name}"):
                    # Log dữ liệu đầu vào, kết quả dự đoán, và model_choice
                    mlflow.log_param("input_image_shape", image.shape)
                    mlflow.log_param("predicted_digit", predicted_digit if predicted_digit else "N/A")
                    mlflow.log_param("confidence", confidence if confidence else "N/A")
                    mlflow.log_param("model_run_id", selected_run_id)
                    mlflow.log_param("model_used", model_choice)  # Log model_choice
                    mlflow.log_text(str(image.flatten()), "input_image.txt")  # Log dưới dạng chuỗi để tránh lỗi bytes

                    st.success(f"Kết quả dự đoán đã được log vào MLflow thành công!\n- Experiment: '{experiment_name}'\n- Run ID: {mlflow.active_run().info.run_id}\n- Liên kết: [Xem trong MLflow UI](http://127.0.0.1:5000/#/experiments/{mlflow.get_experiment_by_name(experiment_name).experiment_id}/runs/{mlflow.active_run().info.run_id})")

    # Hiển thị lịch sử dự đoán từ MLflow (không dùng session)
    # Sửa đổi bởi Grok 3: Loại bỏ session, hiển thị từ MLflow
    st.subheader("Lịch sử Dự đoán Đã Lưu từ MLflow")
    prediction_runs = mlflow.search_runs(experiment_names=[experiment_name], filter_string="tags.mlflow.runName LIKE 'MNIST_Prediction%'")
    if not prediction_runs.empty:
        st.write("Danh sách các dự đoán gần đây:")
        for _, run in prediction_runs.iterrows():
            pred_digit = run.data.params.get("predicted_digit", "N/A")
            confidence = run.data.params.get("confidence", "N/A")
            timestamp = run.info.start_time
            st.write(f"--- Dự đoán ---")
            st.write(f"Chữ số dự đoán: {pred_digit}")
            st.write(f"Độ tin cậy: {confidence}%")
            st.write(f"Thời gian: {pd.Timestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')}")
            # Tải và hiển thị hình ảnh từ MLflow
            image_path = mlflow.artifacts.download_artifacts(run_id=run.run_id, path="input_image.txt")
            with open(image_path, 'r') as f:
                image_data = np.array([float(x) for x in f.read().split()]).reshape(28, 28)
            st.image(image_data, caption=f"Hình ảnh đã vẽ cho dự đoán {pred_digit}", width=100)
    else:
        st.write("Chưa có dự đoán nào được log vào MLflow. Vui lòng thực hiện dự đoán để xem lịch sử.")

if __name__ == "__main__":
    show_mnist_demo()