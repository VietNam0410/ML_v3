import streamlit as st
import numpy as np
from PIL import Image
import cv2

def show_mnist_demo():
    st.header("Demo Nhận diện Chữ số Viết Tay MNIST 🖌️")

    # Kiểm tra mô hình đã huấn luyện trong session
    if 'mnist_model' not in st.session_state:
        st.error("Mô hình MNIST đã huấn luyện không tìm thấy. Vui lòng huấn luyện mô hình trong 'Huấn luyện Mô hình Nhận diện Chữ số MNIST' trước.")
        return

    model = st.session_state['mnist_model']

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
            # Chuyển đổi hình ảnh từ canvas thành mảng numpy
            image = Image.fromarray(canvas_result.image_data).convert('L')  # Chuyển thành grayscale
            image = np.array(image.resize((28, 28)))  # Thay đổi kích thước về 28x28
            image = image / 255.0  # Chuẩn hóa [0, 1]

            # Hiển thị hình ảnh đã vẽ
            st.image(image, caption="Hình ảnh đã vẽ (28x28)", width=100)

            # Dự đoán
            if st.button("Nhận diện chữ số"):
                prediction = model.predict(image.reshape(1, 28, 28, 1))
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100

                st.write(f"Dự đoán: {predicted_digit}")
                st.write(f"Độ tin cậy: {confidence:.2f}%")

                # Lưu kết quả trong session
                # Sửa đổi bởi Grok 3: Loại bỏ MLflow, lưu trong session_state
                st.session_state['mnist_prediction'] = {
                    "input_image": image,
                    "predicted_digit": predicted_digit,
                    "confidence": confidence,
                    "timestamp": pd.Timestamp.now().isoformat()
                }
                st.success("Kết quả dự đoán đã được lưu trong session! ✅")

    # Hiển thị lịch sử dự đoán
    st.subheader("Lịch sử Dự đoán Đã Lưu")
    if 'mnist_prediction' in st.session_state:
        st.write("Kết quả dự đoán gần đây:")
        pred = st.session_state['mnist_prediction']
        st.write(f"Chữ số dự đoán: {pred['predicted_digit']}")
        st.write(f"Độ tin cậy: {pred['confidence']:.2f}%")
        st.write(f"Thời gian: {pred['timestamp']}")
        st.image(pred['input_image'], caption=f"Hình ảnh đã vẽ cho dự đoán {pred['predicted_digit']}", width=100)

if __name__ == "__main__":
    show_mnist_demo()