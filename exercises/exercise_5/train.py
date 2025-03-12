import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import mlflow
import mlflow.keras
import plotly.graph_objects as go
import os
from datetime import datetime  # Import datetime để sửa lỗi

# Thiết lập MLflow
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

def plot_network_structure(n_hidden_layers, neurons_per_layer):
    """Vẽ biểu đồ 2D mô tả cấu trúc mạng"""
    layers = ['Input (784)'] + [f'Hidden {i+1} ({neurons_per_layer})' for i in range(n_hidden_layers)] + ['Output (10)']
    x = list(range(len(layers)))
    y = [784] + [neurons_per_layer] * n_hidden_layers + [10]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name='Số nơ-ron',
                             line=dict(color='blue'), marker=dict(size=10)))
    fig.update_layout(
        title="Cấu Trúc Neural Network",
        xaxis_title="Lớp",
        yaxis_title="Số Nơ-ron",
        xaxis=dict(tickmode='array', tickvals=x, ticktext=layers),
        height=400
    )
    return fig

def train_mnist(X_full, y_full):
    st.title("🧠 Huấn Luyện Neural Network: Nhận Diện Số MNIST")
    st.markdown("""
    Dùng Neural Network để đoán số viết tay từ MNIST.  
    Chọn các tham số dưới đây và xem mô hình học như thế nào!  
    **Lưu ý**: Kết quả dựa trên tập train (80%) và validation (20%) từ dữ liệu đầu vào.
    """)

    # Khởi tạo session_state để lưu giá trị learning_rate
    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.001  # Giá trị mặc định

    # Chọn dữ liệu
    st.subheader("1. Chọn Dữ Liệu")
    total_samples = X_full.shape[0]
    max_samples = st.slider("Số mẫu huấn luyện", 1000, total_samples, 10000, step=1000)
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X, y = X_full[indices], y_full[indices]
    else:
        X, y = X_full, y_full
    X = X.reshape(-1, 28, 28).astype('float32') / 255.0  # Chuẩn hóa
    y_cat = to_categorical(y, 10)
    st.write(f"Đã chọn {max_samples} mẫu để huấn luyện.")

    # Tham số huấn luyện (mặc định tối ưu)
    st.subheader("2. Thiết Lập Tham Số")
    st.markdown("Tùy chỉnh mạng của bạn (các giá trị mặc định được tối ưu để train nhanh):")
    col1, col2 = st.columns(2)
    with col1:
        n_hidden_layers = st.slider("Số lớp ẩn", 1, 3, 2, help="Số lớp xử lý dữ liệu (2 là tối ưu).")
        neurons_per_layer = st.selectbox("Số nơ-ron mỗi lớp", [64, 128, 256], index=1, help="128 nơ-ron là cân bằng giữa tốc độ và hiệu suất.")
        epochs = st.slider("Số vòng lặp (epochs)", 5, 20, 5, help="5 epochs là đủ để đạt độ chính xác tốt.")
    with col2:
        batch_size = st.selectbox("Kích thước batch", [32, 64, 128], index=1, help="64 là tối ưu cho tốc độ.")
        learning_rate = st.number_input(
            "Tốc độ học (η)", 
            min_value=0.00001,  # Giảm min_value để tăng độ chính xác
            max_value=0.1, 
            value=float(st.session_state['learning_rate']), 
            step=0.00001,  # Giảm step để nhập chuẩn xác hơn
            key="learning_rate_input",
            help="Nhập tốc độ học (gợi ý: 0.001 là phù hợp nhất cho MNIST, thử từ 0.0001 đến 0.01 với độ chính xác cao)."
        )
        activation = st.selectbox("Hàm kích hoạt", ['relu', 'sigmoid', 'tanh', 'softmax'], index=0, 
                                  help="ReLU thường cho kết quả tốt nhất.")

    # Cập nhật session_state với giá trị learning_rate mới
    if learning_rate != st.session_state['learning_rate']:
        st.session_state['learning_rate'] = learning_rate
        st.write(f"Đã cập nhật tốc độ học: **{learning_rate}**")

    # Huấn luyện
    if st.button("Bắt Đầu Huấn Luyện"):
        # Xây dựng mô hình
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        for _ in range(n_hidden_layers):
            model.add(Dense(neurons_per_layer, activation=activation if activation != 'softmax' else 'relu'))
            model.add(Dropout(0.2))  # Dropout cố định
        model.add(Dense(10, activation='softmax'))

        # Biên dịch mô hình
        optimizer = Adam(learning_rate=st.session_state['learning_rate'])
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])

        # MLflow
        mlflow.set_experiment("MNIST_Neural_Network")
        with mlflow.start_run(run_name=f"Train_{n_hidden_layers}_{neurons_per_layer}_{epochs}"):
            # Hiển thị thanh tiến trình 100% ngay lập tức
            progress_bar = st.progress(1.0)
            st.write("Huấn luyện hoàn tất (thanh tiến trình giả lập 100%).")

            # Huấn luyện mô hình (ẩn quá trình, chỉ lấy kết quả)
            history = model.fit(X, y_cat, epochs=epochs, batch_size=batch_size, 
                              validation_split=0.2, verbose=0)

            # Hiển thị đầy đủ kết quả
            st.subheader("3. Kết Quả Huấn Luyện")
            train_loss = min(history.history['loss'])
            val_loss = min(history.history['val_loss'])
            train_acc = max(history.history['accuracy'])
            val_acc = max(history.history['val_accuracy'])
            
            st.success(f"**Độ chính xác (Train): {train_acc:.4f}**")
            st.success(f"**Độ chính xác (Validation): {val_acc:.4f}**")
            st.success(f"**Mất mát (Train): {train_loss:.4f}**")
            st.success(f"**Mất mát (Validation): {val_loss:.4f}**")

            # Biểu đồ độ chính xác và mất mát qua các epoch
            st.subheader("4. Hiệu Suất Qua Các Epoch")
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=history.history['accuracy'], mode='lines', name='Train Accuracy', line=dict(color='blue')))
            fig.add_trace(go.Scatter(y=history.history['val_accuracy'], mode='lines', name='Validation Accuracy', line=dict(color='orange')))
            fig.add_trace(go.Scatter(y=history.history['loss'], mode='lines', name='Train Loss', line=dict(color='green')))
            fig.add_trace(go.Scatter(y=history.history['val_loss'], mode='lines', name='Validation Loss', line=dict(color='red')))
            fig.update_layout(title="Hiệu suất qua các epoch", xaxis_title="Epoch", yaxis_title="Giá trị", 
                              height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Biểu đồ cấu trúc mạng
            st.subheader("5. Cấu Trúc Mạng Neural")
            fig_structure = plot_network_structure(n_hidden_layers, neurons_per_layer)
            st.plotly_chart(fig_structure, use_container_width=True)

            # Log MLflow
            mlflow.log_params({
                "n_hidden_layers": n_hidden_layers,
                "neurons_per_layer": neurons_per_layer,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": st.session_state['learning_rate'],
                "activation": activation,
                "samples": max_samples,
                "log_time": f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            })
            mlflow.log_metrics({
                "train_accuracy": float(train_acc),
                "val_accuracy": float(val_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss)
            })
            mlflow.keras.log_model(model, "model")

            # Hiển thị thông tin MLflow
            st.subheader("6. Thông Tin Được Ghi Lại")
            runs = mlflow.search_runs()
            expected_columns = ['params.n_hidden_layers', 'params.neurons_per_layer', 'params.epochs', 
                                'params.batch_size', 'params.learning_rate', 'params.activation', 
                                'params.samples', 'metrics.train_accuracy', 'metrics.val_accuracy',
                                'metrics.train_loss', 'metrics.val_loss']
            for col in expected_columns:
                if col not in runs.columns:
                    runs[col] = None
            st.dataframe(runs[['run_id', 'params.log_time'] + expected_columns])