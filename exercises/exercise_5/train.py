import os
import streamlit as st
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import mlflow
import mlflow.keras
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from datetime import datetime

# Thiết lập MLflow
DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

# Callback để cập nhật thanh tiến trình
class ProgressCallback(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        progress = (epoch + 1) / self.total_epochs
        self.progress_bar.progress(progress)
        self.status_text.text(f"Epoch {epoch + 1}/{self.total_epochs} - Loss: {logs['loss']:.4f} - Accuracy: {logs['accuracy']:.4f}")

# Hàm xử lý ảnh
def preprocess_image(image):
    image = image.astype('float32') / 255.0
    image = image.reshape(-1, 28, 28)
    return image

def plot_network_structure(n_hidden_layers, neurons_per_layer):
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
    st.title("🧠 Huấn Luyện Neural Network")

    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.001

    # Chọn dữ liệu
    st.subheader("1. Chọn Dữ Liệu")
    total_samples = X_full.shape[0]
    max_samples = st.number_input("Số mẫu huấn luyện", min_value=1000, max_value=total_samples, value=10000, step=100)
    if max_samples < total_samples:
        indices = np.random.choice(total_samples, max_samples, replace=False)
        X, y = X_full[indices], y_full[indices]
    else:
        X, y = X_full, y_full
    X = preprocess_image(X)
    y_cat = to_categorical(y, 10)
    st.write(f"Đã chọn {max_samples} mẫu để huấn luyện.")

    # Chia dữ liệu
    st.subheader("2. Chia Dữ Liệu")
    test_size = st.number_input("Tỷ lệ tập test (%)", min_value=10, max_value=50, value=20, step=1) / 100
    val_size = st.number_input("Tỷ lệ tập validation (%)", min_value=10, max_value=50, value=20, step=1) / 100
    train_size = 1 - test_size - val_size
    if train_size <= 0:
        st.error("Tổng tỷ lệ train, validation và test phải nhỏ hơn 100%!")
        return

    X_temp, X_test, y_temp, y_test = train_test_split(X, y_cat, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size/(1-test_size), random_state=42)

    st.write(f"Tỷ lệ: Train {train_size*100:.1f}%, Validation {val_size*100:.1f}%, Test {test_size*100:.1f}%")
    st.write(f"Số mẫu: Train {len(X_train)}, Validation {len(X_val)}, Test {len(X_test)}")

    # Tham số huấn luyện
    st.subheader("3. Thiết Lập Tham Số")
    col1, col2 = st.columns(2)
    with col1:
        n_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=10, value=2)
        neurons_per_layer = st.number_input("Số nơ-ron mỗi lớp", min_value=16, max_value=1024, value=128, step=16)
        epochs = st.number_input("Số vòng lặp (epochs)", min_value=1, max_value=100, value=5)
    with col2:
        batch_size = st.number_input("Kích thước batch", min_value=16, max_value=512, value=64, step=16)
        learning_rate = st.number_input("Tốc độ học (η)", min_value=0.00001, max_value=0.1, value=float(st.session_state['learning_rate']), step=0.00001, key="learning_rate_input")
        activation = st.selectbox("Hàm kích hoạt", ['relu', 'sigmoid', 'tanh'], index=0)

    # Tùy chỉnh tên run
    run_name = st.text_input("Tên Run (để trống để tự động tạo)", value="")
    if not run_name:
        run_name = f"Train_{n_hidden_layers}_{neurons_per_layer}_{epochs}"

    if learning_rate != st.session_state['learning_rate']:
        st.session_state['learning_rate'] = learning_rate
        st.write(f"Đã cập nhật tốc độ học: **{learning_rate}**")

    # Huấn luyện
    if st.button("Bắt Đầu Huấn Luyện"):
        # Xây dựng mô hình
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        for _ in range(n_hidden_layers):
            model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(10, activation='softmax'))

        # Biên dịch mô hình
        optimizer = Adam(learning_rate=st.session_state['learning_rate'])
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        # Huấn luyện mô hình
        mlflow.set_experiment("MNIST_Neural_Network")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            progress_callback = ProgressCallback(epochs)
            history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                              validation_data=(X_val, y_val), callbacks=[progress_callback], verbose=0)

            # Đánh giá trên các tập
            train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
            val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
            test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

            # Kiểm tra overfitting
            if train_acc - val_acc > 0.1:
                st.warning("**Cảnh báo**: Độ chính xác trên tập train cao hơn đáng kể so với tập validation. Mô hình có thể đang overfitting.")

            # Hiển thị kết quả
            st.subheader("4. Kết Quả Huấn Luyện")
            st.success(f"**Độ chính xác (Train)**: {train_acc:.4f}")
            st.success(f"**Độ chính xác (Validation)**: {val_acc:.4f}")
            st.success(f"**Độ chính xác (Test)**: {test_acc:.4f}")
            st.success(f"**Mất mát (Train)**: {train_loss:.4f}")
            st.success(f"**Mất mát (Validation)**: {val_loss:.4f}")
            st.success(f"**Mất mát (Test)**: {test_loss:.4f}")

            # Biểu đồ lịch sử huấn luyện (giữ nguyên biểu đồ line)
            st.subheader("5. Biểu Đồ Hiệu Suất Theo Epoch")
            fig_line = go.Figure()
            fig_line.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history.history['accuracy'], mode='lines+markers', name='Train Accuracy'))
            fig_line.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history.history['val_accuracy'], mode='lines+markers', name='Validation Accuracy'))
            fig_line.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history.history['loss'], mode='lines+markers', name='Train Loss'))
            fig_line.add_trace(go.Scatter(x=list(range(1, epochs+1)), y=history.history['val_loss'], mode='lines+markers', name='Validation Loss'))
            fig_line.update_layout(title="Độ chính xác và Mất mát qua các Epoch", xaxis_title="Epoch", yaxis_title="Giá trị", height=500)
            st.plotly_chart(fig_line, use_container_width=True)

            # Biểu đồ cột cho từng epoch
            st.subheader("6. Biểu Đồ Cột Hiệu Suất Qua Từng Epoch")
            epochs_list = list(range(1, epochs + 1))
            train_acc_history = history.history['accuracy']  # List of accuracies per epoch
            val_acc_history = history.history['val_accuracy']
            train_loss_history = history.history['loss']
            val_loss_history = history.history['val_loss']

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=epochs_list, y=train_acc_history, name='Train Accuracy', marker_color='blue'))
            fig_bar.add_trace(go.Bar(x=epochs_list, y=val_acc_history, name='Validation Accuracy', marker_color='orange'))
            fig_bar.add_trace(go.Bar(x=epochs_list, y=train_loss_history, name='Train Loss', marker_color='green'))
            fig_bar.add_trace(go.Bar(x=epochs_list, y=val_loss_history, name='Validation Loss', marker_color='red'))
            fig_bar.update_layout(
                title="Hiệu Suất Qua Từng Epoch",
                xaxis_title="Epoch",
                yaxis_title="Giá trị",
                barmode='group',  # Nhóm các cột lại với nhau
                height=400
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # Biểu đồ so sánh
            st.subheader("7. So Sánh Train, Validation và Test")
            fig_compare = go.Figure()
            fig_compare.add_trace(go.Bar(x=['Train', 'Validation', 'Test'], y=[train_acc, val_acc, test_acc], name='Accuracy', marker_color=['blue', 'orange', 'purple']))
            fig_compare.add_trace(go.Bar(x=['Train', 'Validation', 'Test'], y=[train_loss, val_loss, test_loss], name='Loss', marker_color=['green', 'red', 'pink']))
            fig_compare.update_layout(title="So sánh Độ chính xác và Mất mát", xaxis_title="Tập dữ liệu", yaxis_title="Giá trị", barmode='group', height=400)
            st.plotly_chart(fig_compare, use_container_width=True)

            # Biểu đồ cấu trúc mạng
            st.subheader("8. Cấu Trúc Mạng Neural")
            fig_structure = plot_network_structure(n_hidden_layers, neurons_per_layer)
            st.plotly_chart(fig_structure, use_container_width=True)

            # Log MLflow
            mlflow.log_params({
                "run_id": run_id,
                "run_name": run_name,
                "n_hidden_layers": n_hidden_layers,
                "neurons_per_layer": neurons_per_layer,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": st.session_state['learning_rate'],
                "activation": activation,
                "samples": max_samples,
                "train_size": len(X_train),
                "val_size": len(X_val),
                "test_size": len(X_test),
                "log_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            mlflow.log_metrics({
                "train_accuracy": float(train_acc),  # Use the scalar value from model.evaluate()
                "val_accuracy": float(val_acc),
                "test_accuracy": float(test_acc),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "test_loss": float(test_loss)
            })
            # Log model
            mlflow.keras.log_model(model, "model")

            # Hiển thị thông tin MLflow
            st.subheader("9. Thông Tin Được Ghi Lại")
            runs = mlflow.search_runs()
            expected_columns = [
                'params.run_id',
                'params.run_name',
                'params.n_hidden_layers', 'params.neurons_per_layer', 'params.epochs',
                'params.batch_size', 'params.learning_rate', 'params.activation',
                'params.samples', 
                'metrics.train_accuracy', 'metrics.val_accuracy', 'metrics.test_accuracy', 
                'metrics.train_loss', 'metrics.val_loss', 'metrics.test_loss', 'params.log_time'
            ]
            for col in expected_columns:
                if col not in runs.columns:
                    runs[col] = None
            st.dataframe(runs[['run_id'] + expected_columns])
