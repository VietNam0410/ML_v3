import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import mlflow
import mlflow.keras
import plotly.graph_objects as go
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from datetime import datetime

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

# Hàm tải mô hình từ MLflow hoặc cục bộ
@st.cache_resource
def load_trained_model(run_id, local_path="final_model"):
    mlflow.set_experiment("MNIST_Pseudo_Labeling_Train")
    model_uri = f"runs:/{run_id}/final_model"
    model = None

    # Thử tải từ MLflow
    try:
        model = mlflow.keras.load_model(model_uri)
        st.success("Đã tải mô hình từ MLflow.")
        return model
    except Exception as e:
        st.warning(f"Không thể tải mô hình từ MLflow. Lỗi: {str(e)}")
    
    # Nếu không tải được từ MLflow, thử tải từ file cục bộ
    if os.path.exists(local_path):
        try:
            model = load_model(local_path)
            st.success("Đã tải mô hình từ file cục bộ.")
            return model
        except Exception as e:
            st.error(f"Không thể tải mô hình từ file cục bộ. Lỗi: {str(e)}")
            return None
    else:
        st.error("Không tìm thấy mô hình cục bộ. Vui lòng huấn luyện và lưu mô hình trước!")
        return None

# Giao diện demo
def demo_mnist_6():
    st.title("✏️ Dự Đoán Chữ Số MNIST với Pseudo Labeling")

    # Khởi tạo lịch sử dự đoán trong session_state
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []

    # Chọn run_name từ MLflow
    st.subheader("1. Chọn Mô Hình Đã Huấn Luyện")
    mlflow.set_experiment("MNIST_Pseudo_Labeling_Train")
    runs = mlflow.search_runs(experiment_ids=[mlflow.get_experiment_by_name("MNIST_Pseudo_Labeling_Train").experiment_id])
    if runs.empty:
        st.error("Không tìm thấy mô hình nào trong MLflow. Vui lòng huấn luyện mô hình trước!")
        return
    
    # Tạo tùy chọn hiển thị run_name kèm thời gian log (nếu có)
    run_options = {}
    for _, row in runs.iterrows():
        run_id = row['run_id']
        run_name = row.get('params.run_name', 'Không có tên')  # Lấy run_name từ MLflow
        log_time = row.get('params.log_time', 'Không có thời gian')
        run_options[f"{run_name} - {log_time}"] = run_id  # Hiển thị run_name - log_time
    
    selected_run = st.selectbox("Chọn Run", list(run_options.keys()))
    run_id = run_options[selected_run]  # Lấy run_id tương ứng để tải mô hình
    
    # Tải mô hình
    model = load_trained_model(run_id)
    if model is None:
        return

    st.success(f"Đã tải mô hình từ Run ID: {run_id} (Run Name: {selected_run})")

    # Chọn phương thức nhập liệu
    st.subheader("2. Nhập Chữ Số")
    input_method = st.radio("Chọn cách nhập:", ("Vẽ tay", "Upload ảnh"))

    # Biến lưu trữ ảnh đầu vào và trạng thái
    if 'input_image' not in st.session_state:
        st.session_state['input_image'] = None
    if 'last_processed_image' not in st.session_state:
        st.session_state['last_processed_image'] = None
    if 'can_predict' not in st.session_state:
        st.session_state['can_predict'] = False

    # Xử lý đầu vào
    if input_method == "Vẽ tay":
        st.write("Vẽ chữ số (kích thước 400x400, sẽ được thu nhỏ về 28x28):")
        if 'reset_canvas' not in st.session_state:
            st.session_state['reset_canvas'] = False

        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=20,
            stroke_color="white",
            background_color="black",
            height=400,
            width=400,
            drawing_mode="freedraw",
            key="canvas",
            update_streamlit=not st.session_state['reset_canvas']
        )

        if canvas_result.image_data is not None:
            img_array = canvas_result.image_data.copy()
            img = Image.fromarray(img_array.astype('uint8')).convert('L')
            processed_image = preprocess_image(img)
            
            # Kiểm tra xem hình ảnh có thay đổi so với lần xử lý trước không
            if st.session_state['last_processed_image'] is None or not np.array_equal(processed_image, st.session_state['last_processed_image']):
                st.session_state['input_image'] = processed_image
                st.session_state['last_processed_image'] = processed_image
                st.session_state['can_predict'] = True
            st.image(img.resize((28, 28)), caption="Ảnh đã xử lý (28x28)", width=100)
    else:
        uploaded_file = st.file_uploader("Tải ảnh chữ số (jpg, png)", type=["jpg", "png"])
        if uploaded_file is not None:
            img = Image.open(uploaded_file)
            processed_image = preprocess_image(img)
            
            # Kiểm tra xem hình ảnh có thay đổi so với lần xử lý trước không
            if st.session_state['last_processed_image'] is None or not np.array_equal(processed_image, st.session_state['last_processed_image']):
                st.session_state['input_image'] = processed_image
                st.session_state['last_processed_image'] = processed_image
                st.session_state['can_predict'] = True
            st.image(img.resize((28, 28)), caption="Ảnh đã xử lý (28x28)", width=100)

    # Nút dự đoán
    if st.session_state['input_image'] is not None and st.session_state['can_predict'] and st.button("Dự đoán"):
        st.subheader("3. Kết Quả Dự Đoán")
        prediction = model.predict(st.session_state['input_image'])
        predicted_digit = np.argmax(prediction, axis=1)[0]  # Lấy chữ số có độ tin cậy cao nhất

        # Hiển thị chữ số dự đoán và độ tin cậy
        confidence = prediction[0][predicted_digit] * 100  # Độ tin cậy của chữ số được dự đoán
        st.write(f"**Dự đoán**: Chữ số **{predicted_digit}**")
        st.write(f"**Độ tin cậy cao nhất**: {confidence:.2f}%")

        # Hiển thị độ tin cậy cho tất cả các chữ số trên biểu đồ
        probabilities = prediction[0] * 100  # Chuyển đổi thành phần trăm

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=list(range(10)),
            y=probabilities,
            marker_color=['blue' if i != predicted_digit else 'red' for i in range(10)],
            text=[f"{x:.2f}%" for x in probabilities],
            textposition='auto',
            width=0.5
        ))
        fig.update_layout(
            title=f"Độ tin cậy dự đoán: {confidence:.2f}%",
            xaxis_title="Chữ số (0-9)",
            yaxis_title="Độ tin cậy (%)",
            height=400,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig, use_container_width=True, key="current_prediction_chart")

        # Lưu dự đoán vào lịch sử
        st.session_state['prediction_history'].append({
            'image': st.session_state['input_image'].reshape(28, 28),  # Lưu ảnh đã xử lý
            'predicted_digit': predicted_digit,
            'confidence': confidence,
            'probabilities': probabilities,
            # Tạo một biểu đồ mới cho lịch sử để tránh xung đột
            'fig': go.Figure(
                data=[
                    go.Bar(
                        x=list(range(10)),
                        y=probabilities,
                        marker_color=['blue' if i != predicted_digit else 'red' for i in range(10)],
                        text=[f"{x:.2f}%" for x in probabilities],
                        textposition='auto',
                        width=0.5
                    )
                ],
                layout={
                    'title': f"Độ tin cậy dự đoán: {confidence:.2f}%",
                    'xaxis_title': "Chữ số (0-9)",
                    'yaxis_title': "Độ tin cậy (%)",
                    'height': 400,
                    'yaxis': {'range': [0, 100]}
                }
            )
        })
        
        # Đặt lại trạng thái để tránh dự đoán lại trên cùng một hình ảnh
        st.session_state['can_predict'] = False

        # Hiển thị thông tin mô hình từ MLflow
        st.subheader("4. Thông Tin Mô Hình (Tham Khảo)")
        run_data = runs[runs['run_id'] == run_id].iloc[0]
        test_acc = run_data.get('metrics.final_test_accuracy', None)
        if test_acc is not None:
            test_acc = float(test_acc) * 100  # Chuyển đổi thành phần trăm
        iterations = run_data.get('params.labeling_iterations', 'Không có dữ liệu')
        num_samples = run_data.get('params.num_samples', 'Không có dữ liệu')
        initial_threshold = run_data.get('params.initial_threshold', 'Không có dữ liệu')

        st.write(f"**Thông tin từ MLflow:**")
        if test_acc is not None:
            st.write(f"- Độ chính xác trên tập test: {test_acc:.2f}%")
        else:
            st.write("- Độ chính xác trên tập test: Không có dữ liệu")
        st.write(f"- Số vòng lặp Pseudo Labeling: {iterations}")
        st.write(f"- Số mẫu huấn luyện: {num_samples}")
        st.write(f"- Ngưỡng độ tin cậy ban đầu: {initial_threshold}")

        # Kiểm tra độ tin cậy so với độ chính xác trên tập test
        if test_acc is not None:
            max_prob = confidence  # Độ tin cậy cao nhất của dự đoán hiện tại
            if max_prob > test_acc + 10:
                st.warning(f"**Cảnh báo**: Độ tin cậy ({max_prob:.2f}%) cao hơn đáng kể so với độ chính xác trên tập test ({test_acc:.2f}%). Kết quả có thể không đáng tin cậy.")
            elif max_prob < test_acc - 20:
                st.info(f"**Ghi chú**: Độ tin cậy ({max_prob:.2f}%) thấp hơn nhiều so với độ chính xác trên tập test ({test_acc:.2f}%). Ảnh đầu vào có thể không rõ ràng hoặc mô hình không phù hợp.")
    elif st.session_state['input_image'] is None:
        st.write("Vui lòng vẽ hoặc tải ảnh trước khi dự đoán.")
    elif not st.session_state['can_predict']:
        st.write("Vui lòng vẽ hoặc tải một hình ảnh mới để dự đoán.")

    # Hiển thị lịch sử dự đoán
    if st.session_state['prediction_history']:
        st.subheader("5. Lịch Sử Dự Đoán")
        for i, pred in enumerate(st.session_state['prediction_history']):
            st.markdown(f"### Dự Đoán {i+1}")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(pred['image'], caption=f"Ảnh dự đoán {i+1} (28x28)", width=100)
            with col2:
                st.write(f"**Dự đoán**: Chữ số **{pred['predicted_digit']}**")
                st.write(f"**Độ tin cậy cao nhất**: {pred['confidence']:.2f}%")
                # Thêm key duy nhất cho mỗi biểu đồ trong lịch sử
                st.plotly_chart(pred['fig'], use_container_width=True, key=f"history_chart_{i}")

        # Nút xóa lịch sử
        if st.button("Xóa lịch sử dự đoán"):
            st.session_state['prediction_history'] = []
            st.rerun()
