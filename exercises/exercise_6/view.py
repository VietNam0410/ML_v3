import streamlit as st
import mlflow
import os
import pandas as pd

def view_log_6():
    st.title("📊 Xem Logs Huấn Luyện MNIST với Pseudo Labeling")
    st.markdown("Danh sách các lần huấn luyện (runs) được lưu trong MLflow.")

    # Thiết lập MLflow
    mlflow.set_tracking_uri("https://dagshub.com/VietNam0410/ML_v3.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"

    # Lấy danh sách các run từ MLflow
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("MNIST_Pseudo_Labeling_Train")
    if experiment is None:
        st.error("Không tìm thấy experiment 'MNIST_Pseudo_Labeling_Train' trong MLflow.")
        return
    
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    
    # Kiểm tra nếu runs không phải là DataFrame hoặc rỗng
    if not isinstance(runs, pd.DataFrame) or runs.empty:
        st.warning("Không có run nào được tìm thấy hoặc dữ liệu không hợp lệ. Vui lòng huấn luyện mô hình trước.")
        return

    # Định nghĩa các cột mong muốn
    expected_columns = [
        'params.run_name',  # Thêm run_name vào danh sách cột
        'params.n_hidden_layers', 'params.neurons_per_layer', 'params.epochs',
        'params.batch_size', 'params.learning_rate', 'params.activation',
        'params.num_samples', 'params.test_ratio', 'params.val_ratio', 'params.train_ratio',
        'params.labeled_ratio', 'params.max_iterations', 'params.initial_threshold',  # Sửa threshold thành initial_threshold để đồng bộ với code huấn luyện
        'params.labeling_iterations', 'params.total_iterations', 'params.log_time',
        'metrics.final_val_accuracy', 'metrics.final_test_accuracy', 'metrics.final_train_accuracy',
        'metrics.total_correct_pseudo_labels'
    ]

    # Tạo DataFrame với các cột hợp lệ, loại bỏ trùng lặp
    valid_columns = ['run_id'] + [col for col in expected_columns if col in runs.columns]
    df_runs = runs[valid_columns].copy()

    # Kiểm tra và loại bỏ cột trùng lặp
    if df_runs.columns.duplicated().any():
        st.warning("Phát hiện cột trùng lặp trong dữ liệu MLflow. Loại bỏ các cột trùng lặp...")
        df_runs = df_runs.loc[:, ~df_runs.columns.duplicated()]

    # Chuyển đổi các cột params và metrics thành số
    numeric_columns = [col for col in df_runs.columns if col.startswith('params.') or col.startswith('metrics.')]
    for col in numeric_columns:
        try:
            df_runs[col] = pd.to_numeric(df_runs[col], errors='coerce').fillna(0)
        except Exception as e:
            st.error(f"Lỗi khi chuyển đổi cột {col}: {str(e)}")
            df_runs[col] = 0  # Gán giá trị mặc định nếu lỗi

    # Thêm bộ lọc
    st.subheader("Lọc Danh Sách Run")
    col1, col2 = st.columns(2)
    with col1:
        min_val_acc = st.slider("Độ chính xác Validation tối thiểu", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with col2:
        min_train_acc = st.slider("Độ chính xác Train tối thiểu", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    # Lọc dữ liệu dựa trên bộ lọc
    filtered_runs = df_runs.loc[
        (df_runs['metrics.final_val_accuracy'].fillna(0) >= min_val_acc) &
        (df_runs['metrics.final_train_accuracy'].fillna(0) >= min_train_acc)
    ].copy()

    # Kiểm tra và loại bỏ cột trùng lặp trong filtered_runs
    if filtered_runs.columns.duplicated().any():
        st.warning("Phát hiện cột trùng lặp trong filtered_runs. Loại bỏ các cột trùng lặp...")
        filtered_runs = filtered_runs.loc[:, ~filtered_runs.columns.duplicated()]

    # Đảm bảo tất cả dữ liệu là kiểu cơ bản để tránh lỗi pyarrow
    for col in filtered_runs.columns:
        if filtered_runs[col].dtype == 'object':
            filtered_runs[col] = filtered_runs[col].astype(str)

    # Hiển thị danh sách các run đã lọc
    st.subheader("Danh Sách Các Run Đã Lưu (Đã Lọc)")
    if filtered_runs.empty:
        st.warning("Không có run nào khớp với bộ lọc. Vui lòng điều chỉnh bộ lọc.")
    else:
        st.dataframe(filtered_runs)  # Hiển thị DataFrame đã xử lý

    # Chọn một run để xem chi tiết
    run_options = filtered_runs['run_id'].tolist()
    selected_run = st.selectbox("Chọn một run để xem chi tiết:", run_options, index=0 if run_options else None)
    if selected_run and run_options:
        run_info = client.get_run(selected_run)
        run_name = run_info.data.params.get('run_name', 'Không có tên')  # Lấy run_name từ params

        # Hiển thị thông tin chi tiết về run
        st.subheader(f"Thông Tin Run Được Chọn: {run_name} (Run ID: {selected_run})")
        st.write(f"**Run ID**: {selected_run}")
        st.write(f"**Run Name**: {run_name}")

        # Hiển thị kết quả (Metrics)
        if run_info.data.metrics:
            st.markdown("### Kết Quả Huấn Luyện")
            final_val_accuracy = run_info.data.metrics.get('final_val_accuracy', 'N/A')
            final_test_accuracy = run_info.data.metrics.get('final_test_accuracy', 'N/A')
            final_train_accuracy = run_info.data.metrics.get('final_train_accuracy', 'N/A')
            total_correct_pseudo_labels = run_info.data.metrics.get('total_correct_pseudo_labels', 'N/A')

            formatted_val_acc = f"{final_val_accuracy:.4f}" if isinstance(final_val_accuracy, (int, float)) else final_val_accuracy
            formatted_test_acc = f"{final_test_accuracy:.4f}" if isinstance(final_test_accuracy, (int, float)) else final_test_accuracy
            formatted_train_acc = f"{final_train_accuracy:.4f}" if isinstance(final_train_accuracy, (int, float)) else final_train_accuracy
            formatted_correct_labels = f"{total_correct_pseudo_labels:.0f}" if isinstance(total_correct_pseudo_labels, (int, float)) else total_correct_pseudo_labels

            st.write(f"**Độ chính xác (Validation)**: {formatted_val_acc}")
            st.write(f"**Độ chính xác (Test)**: {formatted_test_acc}")
            st.write(f"**Độ chính xác (Train)**: {formatted_train_acc}")
            st.write(f"**Tổng số nhãn giả đúng**: {formatted_correct_labels}")
        else:
            st.warning("Không có metrics nào được ghi lại cho run này.")

        # Hiển thị tham số
        if run_info.data.params:
            st.markdown("### Tham Số Huấn Luyện")
            st.write("#### Tham Số Mạng Nơ-ron")
            st.write(f"- Số lớp ẩn: {run_info.data.params.get('n_hidden_layers', 'N/A')}")
            st.write(f"- Số nơ-ron mỗi lớp: {run_info.data.params.get('neurons_per_layer', 'N/A')}")
            st.write(f"- Số vòng lặp (epochs): {run_info.data.params.get('epochs', 'N/A')}")
            st.write(f"- Kích thước batch: {run_info.data.params.get('batch_size', 'N/A')}")
            st.write(f"- Tốc độ học (η): {run_info.data.params.get('learning_rate', 'N/A')}")
            st.write(f"- Hàm kích hoạt: {run_info.data.params.get('activation', 'N/A')}")

            st.write("#### Tham Số Dữ Liệu")
            st.write(f"- Số mẫu huấn luyện: {run_info.data.params.get('num_samples', 'N/A')}")
            st.write(f"- Tỷ lệ Test: {run_info.data.params.get('test_ratio', 'N/A')}%")
            st.write(f"- Tỷ lệ Validation: {run_info.data.params.get('val_ratio', 'N/A')}%")
            st.write(f"- Tỷ lệ Train: {run_info.data.params.get('train_ratio', 'N/A')}%")
            st.write(f"- Tỷ lệ gán nhãn ban đầu: {run_info.data.params.get('labeled_ratio', 'N/A')}")

            st.write("#### Tham Số Pseudo Labeling")
            st.write(f"- Số vòng lặp tối đa: {run_info.data.params.get('max_iterations', 'N/A')}")
            st.write(f"- Ngưỡng độ tin cậy ban đầu: {run_info.data.params.get('initial_threshold', 'N/A')}")  # Sửa để đồng bộ với code huấn luyện
            st.write(f"- Số vòng lặp gán nhãn: {run_info.data.params.get('labeling_iterations', 'N/A')}")
            st.write(f"- Tổng số vòng lặp: {run_info.data.params.get('total_iterations', 'N/A')}")

            st.write("#### Thông Tin Khác")
            st.write(f"- Thời gian huấn luyện: {run_info.data.params.get('log_time', 'N/A')}")
        else:
            st.warning("Không có tham số nào được ghi lại cho run này.")
