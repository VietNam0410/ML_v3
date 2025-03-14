import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mlflow
import mlflow.keras
import plotly.graph_objects as go
from PIL import Image
import cv2
from streamlit_drawable_canvas import st_canvas

# Thiết lập MLflow
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

# Hàm xử lý ảnh (đồng bộ với train)
def preprocess_image(image):
    if isinstance(image, np.ndarray):  # Dữ liệu từ canvas hoặc train
        image = Image.fromarray(image.astype('uint8')).convert('L')
    if image.mode != 'L':
        image = image.convert('L')
    image = image.resize((28, 28))
    img_array = img_to_array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28)
    img_array = 1 - img_array  # Đảo ngược màu cho MNIST
    return img_array

# Hàm tải mô hình từ MLflow
@st.cache_resource
def load_trained_model(run_id):
    mlflow.set_experiment("MNIST_Neural_Network")
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.keras.load_model(model_uri)
    return model

# Giao diện demo
def demo_mnist_5():
    st.title("✏️ Dự Đoán Chữ Số MNIST")

    # Chọn run_id từ MLflow
    st.subheader("1. Chọn Mô Hình Đã Huấn Luyện")
    mlflow.set_experiment("MNIST_Neural_Network")
    runs = mlflow.search_runs()
    if runs.empty:
        st.error("Không tìm thấy mô hình nào trong MLflow. Vui lòng huấn luyện mô hình trước!")
        return
    
    run_options = {f"{row['run_id']} - {row['params.log_time']}" if 'params.log_time' in row else row['run_id']: row['run_id'] for _, row in runs.iterrows()}
    selected_run = st.selectbox("Chọn Run ID", list(run_options.keys()))
    run_id = run_options[selected_run]
    
    # Tải mô hình
    model = load_trained_model(run_id)
    st.success(f"Đã tải mô hình từ Run ID: {run_id}")

    # Chọn phương thức nhập liệu
    st.subheader("2. Nhập Chữ Số")
    input_method = st.radio("Chọn cách nhập:", ("Vẽ tay", "Upload ảnh"))

    # Biến lưu trữ ảnh đầu vào
    input_image = None

    # Xử lý đầu vào
    if input_method == "Vẽ tay":
        st.write("Vẽ chữ số (kích thước 280x280, sẽ được thu nhỏ về 28x28):")
        if 'reset_canvas' not in st.session_state:
            st.session_state['reset_canvas'] = False

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=not st.session_state['reset_canvas']
        )

        if st.button("Xóa bảng vẽ"):
            st.session_state['reset_canvas'] = True
            st.rerun()
        else:
            st.session_state['reset_canvas'] = False

        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.copy()  # Tạo bản sao để tránh lỗi ownership
            img = Image.fromarray(img_array.astype('uint8')).convert('L')
            input_image = preprocess_image(img)
            st.image(img.resize((28, 28)), caption="Ảnh đã xử lý (28x28)", width=100)
    else:
        uploaded_file = st.file_uploader("Tải ảnh chữ số (jpg, png)", type=["jpg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            input_image = preprocess_image(img)
            st.image(img.resize((28, 28)), caption="Ảnh đã xử lý (28x28)", width=100)

    # Nút dự đoán
    if input_image is not None and st.button("Dự đoán"):
        st.subheader("3. Kết Quả Dự Đoán")
        prediction = model.predict(input_image)
        predicted_digit = np.argmax(prediction)
        confidence = prediction[0][predicted_digit] * 100

        # Lấy hiệu suất mô hình từ MLflow
        run_data = runs[runs['run_id'] == run_id].iloc[0]
        test_acc = run_data['metrics.test_accuracy'] * 100 if 'metrics.test_accuracy' in run_data else None

        # Giới hạn độ tin cậy tối đa dựa trên test accuracy
        if test_acc and confidence > test_acc:
            confidence = min(confidence, test_acc + 5)  # Thêm 5% để cho phép một chút sai số
            st.warning(f"**Ghi chú**: Độ tin cậy đã được điều chỉnh xuống {confidence:.2f}% để phù hợp với độ chính xác thực tế của mô hình trên tập test ({test_acc:.2f}%).")

        st.write(f"**Dự đoán**: Chữ số **{predicted_digit}**")
        st.write(f"**Độ tin cậy**: {confidence:.2f}%")

        # Biểu đồ xác suất (đồng bộ với độ tin cậy đã điều chỉnh)
        adjusted_prediction = prediction.copy()
        adjusted_prediction[0][predicted_digit] = confidence / 100  # Cập nhật giá trị trên biểu đồ
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(10)),
            y=adjusted_prediction[0] * 100,
            marker_color='blue',
            text=[f"{x:.1f}%" for x in adjusted_prediction[0] * 100],
            textposition='auto'
        ))
        fig.update_layout(
            title="Xác Suất Dự Đoán Cho Từng Chữ Số",
            xaxis_title="Chữ số (0-9)",
            yaxis_title="Xác suất (%)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        # So sánh độ tin cậy với hiệu suất mô hình
        st.subheader("4. So Sánh Với Hiệu Suất Mô Hình")
        train_acc = run_data['metrics.train_accuracy'] * 100 if 'metrics.train_accuracy' in run_data else None
        val_acc = run_data['metrics.val_accuracy'] * 100 if 'metrics.val_accuracy' in run_data else None

        st.write(f"**Độ chính xác của mô hình (theo MLflow):**")
        if train_acc: st.write(f"- Train: {train_acc:.2f}%")
        if val_acc: st.write(f"- Validation: {val_acc:.2f}%")
        if test_acc: st.write(f"- Test: {test_acc:.2f}%")
        
        if test_acc:
            if confidence > test_acc + 10:
                st.warning("**Cảnh báo**: Độ tin cậy cao hơn đáng kể so với độ chính xác trên tập test. Kết quả có thể không chính xác.")
            elif confidence < test_acc - 20:
                st.info("**Ghi chú**: Độ tin cậy thấp hơn nhiều so với độ chính xác trên tập test. Ảnh đầu vào có thể không rõ ràng.")
    elif input_image is None:
        st.write("Vui lòng vẽ hoặc tải ảnh trước khi dự đoán.")

if __name__ == "__main__":
    demo_mnist_5()  