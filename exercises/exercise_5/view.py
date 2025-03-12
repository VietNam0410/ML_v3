import streamlit as st
import mlflow
import os  # Đảm bảo import os để tránh lỗi

def show_mnist_demo():
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
                        'params.samples', 'metrics.train_accuracy', 'metrics.val_accuracy',
                        'metrics.train_loss', 'metrics.val_loss']
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
        st.write(f"**Độ chính xác (Train)**: {run_info.data.metrics.get('train_accuracy', 'N/A') if isinstance(run_info.data.metrics.get('train_accuracy', 'N/A'), (int, float)) else run_info.data.metrics.get('train_accuracy', 'N/A'):.4f}")
        st.write(f"**Độ chính xác (Validation)**: {run_info.data.metrics.get('val_accuracy', 'N/A') if isinstance(run_info.data.metrics.get('val_accuracy', 'N/A'), (int, float)) else run_info.data.metrics.get('val_accuracy', 'N/A'):.4f}")
        st.write(f"**Mất mát (Train)**: {run_info.data.metrics.get('train_loss', 'N/A') if isinstance(run_info.data.metrics.get('train_loss', 'N/A'), (int, float)) else run_info.data.metrics.get('train_loss', 'N/A'):.4f}")
        st.write(f"**Mất mát (Validation)**: {run_info.data.metrics.get('val_loss', 'N/A') if isinstance(run_info.data.metrics.get('val_loss', 'N/A'), (int, float)) else run_info.data.metrics.get('val_loss', 'N/A'):.4f}")
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