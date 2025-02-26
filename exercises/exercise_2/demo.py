import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import mlflow
import os

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

    # Lấy run ID gần nhất (hoặc cho người dùng chọn run)
    latest_run_id = runs['run_id'].iloc[0]
    try:
        model = mlflow.sklearn.load_model(f"runs:/{latest_run_id}/model")
    except Exception as e:
        st.error(f"Không thể tải mô hình từ MLflow. Lỗi: {str(e)}. Vui lòng kiểm tra MLflow hoặc huấn luyện lại mô hình.")
        return

    # Tạo giao diện cho người dùng vẽ chữ số
    st.subheader("Vẽ một chữ số để nhận diện 🖋️")
    drawing_mode = st.checkbox("Bật chế độ vẽ", value=True)
    if drawing_mode:
        canvas_result = st.canvas(
            width=280,
            height=280,
            drawing_mode="freedraw",
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
                scaler = StandardScaler()  # Giả định mô hình đã được chuẩn hóa trong train.py
                input_data_scaled = scaler.fit_transform(input_data)

                prediction = model.predict(input_data_scaled)
                predicted_digit = np.argmax(prediction) if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None
                confidence = np.max(prediction) * 100 if model_choice in ["SVM (Support Vector Machine)", "Decision Tree"] else None

                if predicted_digit is not None and confidence is not None:
                    st.write(f"Dự đoán: {predicted_digit}")
                    st.write(f"Độ tin cậy: {confidence:.2f}%")
                else:
                    st.write("Mô hình này là phân cụm (K-means/DBSCAN), không dự đoán nhãn. Vui lòng sử dụng SVM hoặc Decision Tree.")

                # Kiểm tra và kết thúc run hiện tại nếu có
                # Sửa đổi bởi Grok 3: Thêm log dự đoán vào MLflow
                active_run = mlflow.active_run()
                if active_run:
                    mlflow.end_run()

                with mlflow.start_run(run_name=f"MNIST_Prediction_{experiment_name}"):
                    # Log dữ liệu đầu vào và kết quả dự đoán
                    mlflow.log_param("input_image_shape", image.shape)
                    mlflow.log_param("predicted_digit", predicted_digit if predicted_digit else "N/A")
                    mlflow.log_param("confidence", confidence if confidence else "N/A")
                    mlflow.log_param("model_run_id", latest_run_id)
                    mlflow.log_text(image.tobytes(), "input_image.npy")

                    st.success(f"Kết quả dự đoán đã được log vào MLflow thành công!\n- Experiment: '{experiment_name}'\n- Run ID: {mlflow.active_run().info.run_id}\n- Liên kết: [Xem trong MLflow UI](http://127.0.0.1:5000/#/experiments/{mlflow.get_experiment_by_name(experiment_name).experiment_id}/runs/{mlflow.active_run().info.run_id})")

                    # Lưu kết quả trong session (tùy chọn, để hiển thị lịch sử)
                    st.session_state['mnist_prediction'] = {
                        "input_image": image,
                        "predicted_digit": predicted_digit,
                        "confidence": confidence,
                        "timestamp": pd.Timestamp.now().isoformat()
                    }

    # Hiển thị lịch sử dự đoán (từ session hoặc MLflow)
    st.subheader("Lịch sử Dự đoán Đã Lưu")
    if 'mnist_prediction' in st.session_state:
        st.write("Kết quả dự đoán gần đây (từ session):")
        pred = st.session_state['mnist_prediction']
        st.write(f"Chữ số dự đoán: {pred['predicted_digit'] if pred['predicted_digit'] is not None else 'N/A'}")
        st.write(f"Độ tin cậy: {pred['confidence']:.2f}% if pred['confidence'] is not None else 'N/A'")
        st.write(f"Thời gian: {pred['timestamp']}")
        st.image(pred['input_image'], caption=f"Hình ảnh đã vẽ cho dự đoán {pred['predicted_digit'] if pred['predicted_digit'] is not None else 'N/A'}", width=100)
    else:
        st.write("Chưa có dự đoán nào được lưu trong session. Xem lịch sử trong MLflow UI.")

if __name__ == "__main__":
    show_mnist_demo()