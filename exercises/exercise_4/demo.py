import streamlit as st
import numpy as np
import mlflow
import mlflow.sklearn
from PIL import Image
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from streamlit_drawable_canvas import st_canvas
import os
import dagshub

# Phần khởi tạo kết nối với DagsHub được comment để không truy cập ngay lập tức
# with st.spinner("Đang kết nối với DagsHub..."):
#     dagshub.init(repo_owner='VietNam0410', repo_name='vn0410', mlflow=True)
#     mlflow.set_tracking_uri(f"https://dagshub.com/VietNam0410/vn0410.mlflow")
# st.success("Đã kết nối với DagsHub thành công!")

def demo():
    st.header("Demo Dự đoán Số MNIST với Mô hình Phân loại 🖌️")
    experiment_name = "MNIST_Training"  # Liên kết với train_mnist.py

    # Kiểm tra dữ liệu và mô hình từ preprocess và train_mnist
    if 'mnist_data' not in st.session_state:
        st.error("Vui lòng chạy tiền xử lý dữ liệu trong 'preprocess.py' trước.")
        return
    if 'mnist_model' not in st.session_state:
        st.error("Vui lòng huấn luyện mô hình phân loại trong 'train_mnist.py' trước.")
        return

    # Kiểm tra mô hình phân loại trong session hoặc yêu cầu đường dẫn cục bộ
    if 'mnist_model' in st.session_state:
        model = st.session_state['mnist_model']
        scaler = st.session_state.get('mnist_scaler')  # Lấy scaler từ train_mnist nếu có
        model_type = "SVM (Support Vector Machine)" if isinstance(model, SVC) else "Decision Tree"
        st.write(f"Mô hình được chọn: {model_type} (từ session)")
    else:
        st.info("Vì logging vào DagsHub đã bị tắt, hãy cung cấp đường dẫn đến mô hình phân loại và scaler cục bộ.")
        model_path = st.text_input("Nhập đường dẫn đến file mô hình cục bộ (ví dụ: 'model.pkl')", value="")
        scaler_path = st.text_input("Nhập đường dẫn đến file scaler cục bộ (ví dụ: 'scaler.pkl')", value="")
        if not model_path or not scaler_path:
            st.warning("Vui lòng cung cấp cả đường dẫn mô hình và scaler để tiếp tục.")
            return
        try:
            with open(model_path, "rb") as f:
                import pickle
                model = pickle.load(f)
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
            model_type = "SVM (Support Vector Machine)" if isinstance(model, SVC) else "Decision Tree"
            st.write(f"Mô hình được chọn: {model_type} (từ file cục bộ)")
        except Exception as e:
            st.error(f"Không thể tải mô hình/scaler từ đường dẫn cục bộ: {str(e)}")
            return

    st.subheader("Vẽ hoặc Tải ảnh để dự đoán số 🖋️")
    input_type = st.radio("Chọn phương thức nhập:", ["Vẽ chữ số", "Tải ảnh"])

    if input_type == "Vẽ chữ số":
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

            if st.button("Dự đoán số"):
                input_data = image_array.reshape(1, 28 * 28)
                if scaler:
                    input_data_scaled = scaler.transform(input_data)
                else:
                    input_data_scaled = input_data  # Nếu không có scaler từ train_mnist
                prediction = model.predict(input_data_scaled)[0]

                st.success(f"Dự đoán số: {prediction}")
                st.write("Hình ảnh đầu vào:")
                st.image(image_array, caption="Hình ảnh đã xử lý", width=100)

                # Comment phần logging dự đoán
                # run_name = st.text_input("Nhập tên cho lần thử nghiệm này", value=f"Prediction_Draw_{st.session_state.get('prediction_count', 0) + 1}")
                # if st.button("Log Dự đoán vào MLflow"):
                #     with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id) as run:
                #         mlflow.log_param("model_run_id", "From_Session")
                #         mlflow.log_param("predicted_digit", prediction)
                #         np.save("input_image.npy", image_array)
                #         mlflow.log_artifact("input_image.npy", artifact_path="input_data")
                #         os.remove("input_image.npy")

                #         run_id = run.info.run_id
                #         dagshub_link = f"https://dagshub.com/VietNam0410/vn0410/experiments/#/experiment/{experiment_name}/{run_id}"
                #         st.success(f"Dự đoán: {prediction} (Run ID: {run_id}, Tên Run: {run_name})")
                #         st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")
                #
                #         if 'prediction_count' not in st.session_state:
                #             st.session_state['prediction_count'] = 0
                #         st.session_state['prediction_count'] += 1

    else:
        uploaded_file = st.file_uploader("Tải lên ảnh chữ số (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('L').resize((28, 28))
            image_array = np.array(image) / 255.0
            st.image(image_array, caption="Hình ảnh đã tải", width=100)

            if st.button("Dự đoán số từ ảnh"):
                input_data = image_array.reshape(1, 28 * 28)
                if scaler:
                    input_data_scaled = scaler.transform(input_data)
                else:
                    input_data_scaled = input_data  # Nếu không có scaler từ train_mnist
                prediction = model.predict(input_data_scaled)[0]

                st.success(f"Dự đoán số: {prediction}")
                st.write("Hình ảnh đầu vào:")
                st.image(image_array, caption="Hình ảnh đã xử lý", width=100)

                # Comment phần logging dự đoán
                # run_name = st.text_input("Nhập tên cho lần thử nghiệm này", value=f"Prediction_Upload_{st.session_state.get('prediction_count', 0) + 1}")
                # if st.button("Log Dự đoán vào MLflow"):
                #     with mlflow.start_run(run_name=run_name, experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id) as run:
                #         mlflow.log_param("model_run_id", "From_Session")
                #         mlflow.log_param("predicted_digit", prediction)
                #         np.save("input_image.npy", image_array)
                #         mlflow.log_artifact("input_image.npy", artifact_path="input_data")
                #         os.remove("input_image.npy")

                #         run_id = run.info.run_id
                #         dagshub_link = f"https://dagshub.com/VietNam0410/vn0410/experiments/#/experiment/{experiment_name}/{run_id}"
                #         st.success(f"Dự đoán: {prediction} (Run ID: {run_id}, Tên Run: {run_name})")
                #         st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")
                #
                #         if 'prediction_count' not in st.session_state:
                #             st.session_state['prediction_count'] = 0
                #         st.session_state['prediction_count'] += 1

    # Comment phần hiển thị lịch sử huấn luyện
    st.subheader("Lịch sử dự đoán")
    st.info("Chức năng xem lịch sử dự đoán tạm thời bị tắt vì logging vào DagsHub đã bị vô hiệu hóa.")

if __name__ == "__main__":
    demo()