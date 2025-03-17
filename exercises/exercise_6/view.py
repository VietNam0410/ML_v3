import streamlit as st
import mlflow
import os  # Đảm bảo import os để tránh lỗi

def view_log_5():
    st.title("📊 Xem Logs Huấn Luyện MNIST")
    st.markdown("Danh sách các lần huấn luyện (runs) được lưu trong MLflow.")

    # Thiết lập MLflow
    mlflow.set_tracking_uri("https://dagshub.com/VietNam0410/ML_v3.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

    # Lấy danh sách các run từ MLflow
    experiment = mlflow.get_experiment_by_name("MNIST_Neural_Network")
    if experiment is None:
        st.error("Không tìm thấy experiment 'MNIST_Neural_Network' trong MLflow.")
        return
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    # Kiểm tra nếu không có run nào
    if runs.empty:
        st.warning("Không có run nào được tìm thấy. Vui lòng huấn luyện mô hình trước.")
        return

    # Hiển thị danh sách các run
    st.subheader("Danh Sách Các Run Đã Lưu")
    expected_columns = ['params.n_hidden_layers', 'params.neurons_per_layer', 'params.epochs', 
                        'params.batch_size', 'params.learning_rate', 'params.activation', 
                        'params.samples', 'metrics.train_accuracy', 'metrics.test_accuracy',
                        'metrics.train_loss', 'metrics.test_loss']
    for col in expected_columns:
        if col not in runs.columns:
            runs[col] = None
    st.dataframe(runs[['run_id', 'params.log_time'] + expected_columns])

    # Chọn một run để xem chi tiết
    run_options = runs['run_id'].tolist()
    selected_run = st.selectbox("Chọn một run để xem chi tiết:", run_options, index=0)
    run_info = mlflow.get_run(selected_run)

    # Hiển thị thông tin chi tiết về run
    st.subheader("Thông Tin Run Được Chọn")
    st.write(f"**Run ID**: {selected_run}")
    if run_info.data.metrics:
        # Lấy và định dạng các metric
        train_accuracy = run_info.data.metrics.get('train_accuracy', 'N/A')
        test_accuracy = run_info.data.metrics.get('test_accuracy', 'N/A')
        train_loss = run_info.data.metrics.get('train_loss', 'N/A')
        test_loss = run_info.data.metrics.get('test_loss', 'N/A')

        # Định dạng giá trị
        formatted_train_acc = f"{train_accuracy:.4f}" if isinstance(train_accuracy, (int, float)) else train_accuracy
        formatted_test_acc = f"{test_accuracy:.4f}" if isinstance(test_accuracy, (int, float)) else test_accuracy
        formatted_train_loss = f"{train_loss:.4f}" if isinstance(train_loss, (int, float)) else train_loss
        formatted_test_loss = f"{test_loss:.4f}" if isinstance(test_loss, (int, float)) else test_loss

        # Hiển thị kết quả
        st.write(f"**Độ chính xác (Train)**: {formatted_train_acc}")
        st.write(f"**Độ chính xác (Test)**: {formatted_test_acc}")
        st.write(f"**Mất mát (Train)**: {formatted_train_loss}")
        st.write(f"**Mất mát (Test)**: {formatted_test_loss}")
    else:
        st.warning("Không có metrics nào được ghi lại cho run này.")

    if run_info.data.params:
        st.write("**Tham số Huấn Luyện**:")
        st.write(f"- Số lớp ẩn: {run_info.data.params.get('n_hidden_layers', 'N/A')}")
        st.write(f"- Số nơ-ron mỗi lớp: {run_info.data.params.get('neurons_per_layer', 'N/A')}")
        st.write(f"- Số vòng lặp (epochs): {run_info.data.params.get('epochs', 'N/A')}")
        st.write(f"- Kích thước batch: {run_info.data.params.get('batch_size', 'N/A')}")
        st.write(f"- Tốc độ học (η): {run_info.data.params.get('learning_rate', 'N/A')}")
        st.write(f"- Hàm kích hoạt: {run_info.data.params.get('activation', 'N/A')}")
        st.write(f"- Số mẫu huấn luyện: {run_info.data.params.get('samples', 'N/A')}")
        st.write(f"- Thời gian huấn luyện: {run_info.data.params.get('log_time', 'N/A')}")
    else:
        st.warning("Không có tham số nào được ghi lại cho run này.")