import os
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
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

# Hàm chọn dữ liệu ban đầu dựa trên số lượng mẫu hoặc tỷ lệ
def select_initial_data(X, y, labeled_ratio=0.05):
    X_selected = []
    y_selected = []
    samples_per_digit = {}
    for digit in range(10):
        indices = np.where(y == digit)[0]
        num_samples = max(1, int(len(indices) * labeled_ratio))
        selected_indices = np.random.choice(indices, num_samples, replace=False)
        X_selected.append(X[selected_indices])
        y_selected.append(y[selected_indices])
        samples_per_digit[digit] = len(selected_indices)
    return np.concatenate(X_selected), np.concatenate(y_selected), samples_per_digit

# Hàm hiển thị ví dụ ảnh được gán nhãn giả
def display_pseudo_labeled_examples(X_unlabeled, y_unlabeled_true, pseudo_labels, confidences, high_confidence_indices, iteration):
    st.subheader(f"Ví dụ một số mẫu được gán nhãn giả trong vòng lặp {iteration + 1}")
    num_examples = min(5, len(high_confidence_indices))
    if num_examples == 0:
        st.write("Không có mẫu nào vượt ngưỡng độ tin cậy và được gán đúng trong vòng lặp này.")
        return
        
    sample_indices = np.random.choice(high_confidence_indices, num_examples, replace=False)
    
    cols = st.columns(num_examples)
    for i, idx in enumerate(sample_indices):
        if idx >= len(X_unlabeled):
            continue
            
        img = X_unlabeled[idx].reshape(28, 28)
        pred_label = pseudo_labels[idx]
        true_label = y_unlabeled_true[idx]
        conf = confidences[idx]
        
        with cols[i]:
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            ax.set_title(f"Pred: {pred_label}\nTrue: {true_label}\nConf: {conf:.2f}")
            st.pyplot(fig)

# Hàm huấn luyện với Pseudo Labeling
def train_mnist_pseudo_labeling(X_full, y_full):
    st.title("🧠 Huấn Luyện Neural Network với Pseudo Labeling")

    if 'learning_rate' not in st.session_state:
        st.session_state['learning_rate'] = 0.001

    # Bước 1: Chọn số lượng mẫu và chia tập dữ liệu
    st.subheader("1. Chọn Số Lượng Mẫu và Chia Tập Dữ Liệu")
    num_samples = st.number_input("Số mẫu (0-70000)", min_value=0, max_value=len(X_full), value=min(10000, len(X_full)), step=1000)
    if num_samples == 0:
        st.error("Số mẫu phải lớn hơn 0!")
        return

    labeled_ratio = st.number_input("Tỷ lệ dữ liệu gán nhãn ban đầu (%)", min_value=0.1, max_value=50.0, value=5.0, step=0.1)

    indices = np.random.choice(len(X_full), num_samples, replace=False)
    X_selected, y_selected = X_full[indices], y_full[indices]

    st.subheader("2. Phân Chia Tập Dữ Liệu")
    test_ratio = st.number_input("Tỷ lệ tập Test (%)", min_value=10.0, max_value=80.0, value=20.0, step=1.0)
    val_ratio = st.number_input("Tỷ lệ Validation (%)", min_value=0.0, max_value=80.0 - test_ratio, value=10.0, step=1.0)
    train_ratio = 100.0 - test_ratio - val_ratio
    if train_ratio <= 0:
        st.error("Tỷ lệ Train phải lớn hơn 0! Giảm tỷ lệ Test hoặc Validation.")
        return

    test_size = test_ratio / 100.0
    val_size = val_ratio / (train_ratio + val_ratio) if (train_ratio + val_ratio) > 0 else 0
    X_temp, X_test, y_temp, y_test = train_test_split(X_selected, y_selected, test_size=test_size, random_state=42, stratify=y_selected)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, random_state=42, stratify=y_temp)

    st.write(f"Số mẫu: Train {len(X_train)}, Validation {len(X_val)}, Test {len(X_test)}")

    X_train = preprocess_image(X_train)
    X_val = preprocess_image(X_val)
    X_test = preprocess_image(X_test)
    y_train_cat = to_categorical(y_train, 10)
    y_val_cat = to_categorical(y_val, 10)
    y_test_cat = to_categorical(y_test, 10)

    # Bước 2: Lấy dữ liệu ban đầu
    labeled_ratio = labeled_ratio / 100.0
    X_labeled, y_labeled, samples_per_digit = select_initial_data(X_train, y_train, labeled_ratio)
    y_labeled_cat = to_categorical(y_labeled, 10)
    st.write(f"Số mẫu được gán nhãn ban đầu: {len(X_labeled)}")
    st.write("Phân bố mẫu ban đầu theo chữ số:")
    st.json(samples_per_digit)

    labeled_indices = np.random.choice(len(X_train), len(X_labeled), replace=False)
    unlabeled_indices = np.setdiff1d(np.arange(len(X_train)), labeled_indices)
    X_unlabeled = X_train[unlabeled_indices]
    y_unlabeled_true = y_train[unlabeled_indices]
    st.write(f"Số mẫu chưa được gán nhãn: {len(X_unlabeled)}")

    # Bước 3: Thiết lập tham số Neural Network
    st.subheader("3. Thiết Lập Tham Số Neural Network")
    col1, col2 = st.columns(2)
    with col1:
        n_hidden_layers = st.number_input("Số lớp ẩn", min_value=1, max_value=5, value=2, step=1)
        neurons_per_layer = st.number_input("Số nơ-ron mỗi lớp", min_value=16, max_value=512, value=128, step=16)
        epochs = st.number_input("Số vòng lặp (epochs)", min_value=1, max_value=50, value=20, step=1)
    with col2:
        batch_size = st.number_input("Kích thước batch", min_value=16, max_value=256, value=64, step=16)
        learning_rate = st.number_input("Tốc độ học (η)", min_value=0.00001, max_value=0.01, value=0.0005, step=0.0001, key="learning_rate_input")
        activation = st.selectbox("Hàm kích hoạt", ['relu', 'sigmoid', 'tanh'], index=0)

    if learning_rate != st.session_state['learning_rate']:
        st.session_state['learning_rate'] = learning_rate
        st.write(f"Đã cập nhật tốc độ học: **{learning_rate}**")

    # Bước 4: Thiết lập tham số Pseudo Labeling
    st.subheader("4. Thiết Lập Tham Số Pseudo Labeling")
    max_iterations = st.number_input("Số bước lặp tối đa", min_value=1, max_value=20, value=10, step=1)
    initial_threshold = st.number_input("Ngưỡng độ tin cậy ban đầu", min_value=0.5, max_value=1.0, value=0.7, step=0.01)

    # Tùy chỉnh tên run
    st.subheader("5. Tùy Chỉnh Tên Run")
    run_name = st.text_input("Tên Run (để trống để tự động tạo)", value="")
    if not run_name:
        run_name = f"PseudoLabel_{num_samples}_{test_ratio}_{val_ratio}_{n_hidden_layers}_{max_iterations}_{initial_threshold}"
    st.write(f"Tên Run: **{run_name}**")  # Hiển thị tên run để người dùng biết

    try:
        model = Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        for _ in range(n_hidden_layers):
            model.add(Dense(neurons_per_layer, activation=activation))
        model.add(Dense(10, activation='softmax'))
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        st.error(f"Lỗi khi xây dựng mô hình: {str(e)}")
        return

    if st.button("Bắt Đầu Huấn Luyện với Pseudo Labeling"):
        mlflow.set_experiment("MNIST_Pseudo_Labeling_Train")
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id  # Lấy run_id từ MLflow run
            st.write(f"Run ID: **{run_id}**")  # Hiển thị run_id ngay khi bắt đầu

            X_current = X_labeled.copy()
            y_current = y_labeled_cat.copy()
            iteration = 0
            total_iterations = 0
            history_iterations = []
            
            labeled_counts = [len(X_labeled)]
            unlabeled_counts = [len(X_unlabeled)]
            initial_correct_count = len(X_labeled)
            cumulative_correct_count = initial_correct_count
            cumulative_correct_counts = [cumulative_correct_count]
            confidence_means = []
            correct_ratios = []
            test_accuracies = []
            val_accuracies = []
            iteration_data = []
            additional_epochs = 0
            final_val_acc = 0.0
            final_test_acc = 0.0

            # Thêm trạng thái ban đầu vào iteration_data
            _, initial_val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
            _, initial_test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
            iteration_data.append({
                "Iteration": 0,
                "Labeled Samples": len(X_current),
                "Unlabeled Samples": len(X_unlabeled),
                "Correct Pseudo Labels": 0,
                "Cumulative Correct": cumulative_correct_count,
                "Confidence Mean": 0.0,
                "Val Accuracy": initial_val_acc,
                "Test Accuracy": initial_test_acc
            })

            progress_container = st.empty()
            
            while iteration < max_iterations:
                # Dynamically adjust threshold
                threshold = min(0.95, initial_threshold + iteration * 0.05)
                
                with progress_container.container():
                    st.write(f"**Vòng lặp Gán Nhãn {iteration + 1}/{max_iterations}**")
                    st.write(f"Số mẫu huấn luyện hiện tại: {len(X_current)}")
                    st.write(f"Ngưỡng độ tin cậy hiện tại: {threshold:.4f}")

                    progress_callback = ProgressCallback(epochs)
                    history = model.fit(X_current, y_current, epochs=epochs, batch_size=batch_size, verbose=0,
                                        validation_data=(X_val, y_val_cat), callbacks=[progress_callback])
                    history_iterations.append(history.history)

                    _, val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
                    _, test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
                    val_accuracies.append(val_acc)
                    test_accuracies.append(test_acc)
                    st.write(f"Độ chính xác Validation: {val_acc:.4f}")
                    st.write(f"Độ chính xác Test: {test_acc:.4f}")

                    if len(X_unlabeled) == 0:
                        st.write("Không còn mẫu chưa gán nhãn để xử lý. Tiếp tục huấn luyện với tập hiện tại.")
                        iteration_data.append({
                            "Iteration": iteration + 1,
                            "Labeled Samples": len(X_current),
                            "Unlabeled Samples": 0,
                            "Correct Pseudo Labels": 0,
                            "Cumulative Correct": cumulative_correct_count,
                            "Confidence Mean": confidence_means[-1] if confidence_means else 0,
                            "Val Accuracy": val_acc,
                            "Test Accuracy": test_acc
                        })
                        labeled_counts.append(len(X_current))
                        unlabeled_counts.append(0)
                        cumulative_correct_counts.append(cumulative_correct_count)
                        confidence_means.append(confidence_means[-1] if confidence_means else 0)
                        correct_ratios.append(correct_ratios[-1] if correct_ratios else 0)
                        iteration += 1
                        continue

                    predictions = model.predict(X_unlabeled, verbose=0)
                    confidences = np.max(predictions, axis=1)
                    pseudo_labels = np.argmax(predictions, axis=1)
                    
                    confidence_mean = np.mean(confidences)
                    confidence_means.append(confidence_mean)
                    st.write(f"Độ tin cậy trung bình: {confidence_mean:.4f}")

                    correct_and_confident_indices = np.where((confidences >= threshold) & (pseudo_labels == y_unlabeled_true))[0]
                    correct_count = len(correct_and_confident_indices)

                    if correct_count == 0:
                        threshold = max(0.5, threshold - 0.1)
                        st.warning(f"Không có mẫu nào vượt ngưỡng. Giảm ngưỡng xuống {threshold:.4f}.")
                        correct_and_confident_indices = np.where((confidences >= threshold) & (pseudo_labels == y_unlabeled_true))[0]
                        correct_count = len(correct_and_confident_indices)

                    cumulative_correct_count += correct_count
                    cumulative_correct_counts.append(cumulative_correct_count)

                    correct_ratio = correct_count / len(pseudo_labels) if len(pseudo_labels) > 0 else 0
                    correct_ratios.append(correct_ratio)
                    st.write(f"Số lượng mẫu gán đúng trong vòng lặp này: {correct_count}")
                    st.write(f"Tổng số mẫu gán đúng tích lũy (Pseudo Labeling): {cumulative_correct_count}")
                    st.write(f"Tỷ lệ nhãn giả đúng trong vòng lặp này: {correct_ratio:.4f}")

                    previous_labeled_count = len(X_current)

                    if correct_count == 0:
                        st.warning("Không có mẫu nào vượt ngưỡng và được gán đúng trong vòng lặp này. Tiếp tục với dữ liệu hiện tại.")
                        iteration_data.append({
                            "Iteration": iteration + 1,
                            "Labeled Samples": len(X_current),
                            "Unlabeled Samples": len(X_unlabeled),
                            "Correct Pseudo Labels": correct_count,
                            "Cumulative Correct": cumulative_correct_count,
                            "Confidence Mean": confidence_mean,
                            "Val Accuracy": val_acc,
                            "Test Accuracy": test_acc
                        })
                        labeled_counts.append(len(X_current))
                        unlabeled_counts.append(len(X_unlabeled))
                        iteration += 1
                        continue

                    X_pseudo = X_unlabeled[correct_and_confident_indices]
                    y_pseudo = pseudo_labels[correct_and_confident_indices]
                    y_pseudo_cat = to_categorical(y_pseudo, 10)

                    X_current = np.concatenate([X_current, X_pseudo])
                    y_current = np.concatenate([y_current, y_pseudo_cat])

                    expected_labeled_count = previous_labeled_count + correct_count
                    if len(X_current) != expected_labeled_count:
                        st.error(f"Lỗi đồng bộ: Kích thước của X_current ({len(X_current)}) không khớp với kỳ vọng ({expected_labeled_count})")

                    remaining_indices = np.setdiff1d(np.arange(len(X_unlabeled)), correct_and_confident_indices)
                    X_unlabeled = X_unlabeled[remaining_indices] if len(remaining_indices) > 0 else np.array([])
                    y_unlabeled_true = y_unlabeled_true[remaining_indices] if len(remaining_indices) > 0 else np.array([])

                    iteration_data.append({
                        "Iteration": iteration + 1,
                        "Labeled Samples": len(X_current),
                        "Unlabeled Samples": len(X_unlabeled),
                        "Correct Pseudo Labels": correct_count,
                        "Cumulative Correct": cumulative_correct_count,
                        "Confidence Mean": confidence_mean,
                        "Val Accuracy": val_acc,
                        "Test Accuracy": test_acc
                    })

                    display_pseudo_labeled_examples(X_unlabeled, y_unlabeled_true, pseudo_labels, confidences, correct_and_confident_indices, iteration)

                    labeled_counts.append(len(X_current))
                    unlabeled_counts.append(len(X_unlabeled) if len(X_unlabeled) > 0 else 0)
                    iteration += 1

            st.subheader("Bảng Tiến Trình Pseudo Labeling")
            st.table(iteration_data)

            # Kiểm tra chính xác cuối cùng trên toàn bộ tập train
            final_predictions = model.predict(X_train, verbose=0)
            final_pseudo_labels = np.argmax(final_predictions, axis=1)
            final_correct_count = np.sum(final_pseudo_labels == y_train)
            final_accuracy = final_correct_count / len(y_train)
            st.success(f"Đã hoàn thành gán nhãn sau {iteration} vòng lặp.")
            st.write(f"Tổng số mẫu gán đúng trên toàn bộ tập train: {final_correct_count} / {len(y_train)}")
            st.write(f"Tỷ lệ gán đúng cuối cùng: {final_accuracy:.4f}")

            # Luôn lưu mô hình vào MLflow
            _, final_val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
            _, final_test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
            val_accuracies.append(final_val_acc)
            test_accuracies.append(final_test_acc)
            st.success(f"Độ chính xác cuối cùng - Validation: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
            mlflow.keras.log_model(model, "final_model")
            st.success("Đã log mô hình vào MLflow với tên 'final_model'.")

            # Huấn luyện bổ sung nếu đạt 100% chính xác
            if final_accuracy == 1.0:
                st.subheader("6. Huấn Luyện Bổ Sung với Toàn Bộ Dữ Liệu")
                additional_epochs = st.number_input("Số epochs bổ sung", min_value=1, max_value=50, value=5, step=1)
                if additional_epochs > 0:
                    progress_callback = ProgressCallback(additional_epochs)
                    history = model.fit(X_train, y_train_cat, epochs=additional_epochs, batch_size=batch_size, verbose=0,
                                        validation_data=(X_val, y_val_cat), callbacks=[progress_callback])
                    history_iterations.append(history.history)

                    _, final_val_acc = model.evaluate(X_val, y_val_cat, verbose=0)
                    _, final_test_acc = model.evaluate(X_test, y_test_cat, verbose=0)
                    val_accuracies.append(final_val_acc)
                    test_accuracies.append(final_test_acc)
                    st.success(f"Độ chính xác cuối cùng sau huấn luyện bổ sung - Validation: {final_val_acc:.4f}, Test: {final_test_acc:.4f}")
                    mlflow.keras.log_model(model, "final_model_after_additional_training")
                    st.success("Đã log mô hình sau huấn luyện bổ sung vào MLflow với tên 'final_model_after_additional_training'.")

            total_iterations = iteration + (additional_epochs > 0 and 1 or 0)
            st.write(f"Tổng số vòng lặp: {total_iterations}")

            st.subheader("7. Biểu Đồ Quá Trình Huấn Luyện Pseudo Labeling")
            fig_counts = go.Figure()
            fig_counts.add_trace(go.Scatter(x=list(range(len(labeled_counts))), y=labeled_counts, mode='lines+markers', name='Số mẫu đã gán nhãn'))
            fig_counts.add_trace(go.Scatter(x=list(range(len(unlabeled_counts))), y=unlabeled_counts, mode='lines+markers', name='Số mẫu chưa gán nhãn'))
            fig_counts.update_layout(title="Số lượng mẫu qua các vòng lặp", xaxis_title="Vòng lặp", yaxis_title="Số lượng", height=400)
            st.plotly_chart(fig_counts, use_container_width=True)

            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(x=list(range(len(val_accuracies))), y=val_accuracies, mode='lines+markers', name='Validation Accuracy'))
            fig_acc.add_trace(go.Scatter(x=list(range(len(test_accuracies))), y=test_accuracies, mode='lines+markers', name='Test Accuracy'))
            fig_acc.update_layout(title="Độ chính xác qua các vòng lặp", xaxis_title="Vòng lặp", yaxis_title="Độ chính xác", height=400)
            st.plotly_chart(fig_acc, use_container_width=True)

            fig_correct = go.Figure()
            fig_correct.add_trace(go.Scatter(x=list(range(len(cumulative_correct_counts))), y=cumulative_correct_counts, mode='lines+markers', name='Tích lũy trong Pseudo Labeling'))
            fig_correct.update_layout(
                title="Tổng số lượng mẫu gán đúng qua các vòng lặp",
                xaxis_title="Vòng lặp",
                yaxis_title="Số lượng",
                yaxis_range=[0, len(X_train)],
                height=400
            )
            st.plotly_chart(fig_correct, use_container_width=True)

            # Log các tham số và metrics
            mlflow.log_params({
                "run_id": run_id,  # Log run_id
                "run_name": run_name,  # Log run_name để đồng bộ
                "num_samples": num_samples,
                "test_ratio": test_ratio,
                "val_ratio": val_ratio,
                "train_ratio": train_ratio,
                "labeled_ratio": labeled_ratio,
                "n_hidden_layers": n_hidden_layers,
                "neurons_per_layer": neurons_per_layer,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "activation": activation,
                "max_iterations": max_iterations,
                "initial_threshold": initial_threshold,
                "labeling_iterations": iteration,
                "total_iterations": total_iterations,
                "log_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
            mlflow.log_metrics({
                "final_val_accuracy": float(final_val_acc),
                "final_test_accuracy": float(final_test_acc),
                "final_train_accuracy": float(final_accuracy),
                "total_correct_pseudo_labels": float(cumulative_correct_count),
                "final_correct_count": float(final_correct_count)
            })

            st.subheader("8. Thông Tin Được Ghi Lại")
            runs = mlflow.search_runs()
            expected_columns = [
                'params.run_id',  # Thêm run_id vào danh sách hiển thị
                'params.run_name',  # Thêm run_name vào danh sách hiển thị
                'params.num_samples', 'params.test_ratio', 'params.val_ratio', 'params.train_ratio',
                'params.labeled_ratio', 'params.n_hidden_layers', 'params.neurons_per_layer',
                'params.epochs', 'params.batch_size', 'params.learning_rate', 'params.activation',
                'params.max_iterations', 'params.initial_threshold',
                'params.labeling_iterations', 'params.total_iterations', 'params.log_time',
                'metrics.final_val_accuracy', 'metrics.final_test_accuracy', 'metrics.final_train_accuracy',
                'metrics.total_correct_pseudo_labels', 'metrics.final_correct_count'
            ]
            for col in expected_columns:
                if col not in runs.columns:
                    runs[col] = None
            st.dataframe(runs[['run_id'] + expected_columns])

            # Hiển thị thông báo Run ID cuối cùng
            st.success(f"Huấn luyện hoàn tất! Run ID: **{run_id}** (Tên Run: {run_name})")