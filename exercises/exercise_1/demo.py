import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub
import datetime

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/vn0410.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    st.session_state['mlflow_url'] = DAGSHUB_MLFLOW_URI
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "22fd02345f8ff45482a20960058627630acaf190"  # Thay bằng token cá nhân của bạn
    DAGSHUB_REPO = "vn0410"
    return DAGSHUB_REPO

# Hàm tải dữ liệu với cache
@st.cache_data
def load_cached_data(file_path):
    """Tải dữ liệu từ file CSV và lưu vào bộ nhớ đệm."""
    return load_data(file_path)

def delete_mlflow_run(run_id):
    try:
        with st.spinner(f"Đang xóa Run {run_id}..."):
            mlflow.delete_run(run_id)
        st.success(f"Đã xóa run có ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể xóa run {run_id}: {str(e)}")

def get_mlflow_experiments():
    """Lấy danh sách các experiment từ MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {exp.name: exp.experiment_id for exp in experiments if exp.lifecycle_stage == "active"}
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các experiment từ MLflow: {str(e)}")
        return {}

def show_demo():
    st.header("Dự đoán Sinh tồn Titanic")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Gọi hàm mlflow_input để thiết lập MLflow
    DAGSHUB_REPO = mlflow_input()

    # Lấy tất cả các experiment từ MLflow
    experiments = get_mlflow_experiments()
    if not experiments:
        st.error("Không tìm thấy experiment nào trong DagsHub MLflow. Vui lòng kiểm tra kết nối hoặc huấn luyện mô hình trước.")
        return

    experiment_options = list(experiments.keys())
    experiment_name = st.selectbox(
        "Chọn Experiment để lấy mô hình",
        options=experiment_options,
        help="Chọn một experiment đã huấn luyện mô hình."
    )

    with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(experiment_name)
            if experiment and experiment.lifecycle_stage == "deleted":
                st.warning(f"Experiment '{experiment_name}' đã bị xóa trước đó. Vui lòng chọn tên khác hoặc khôi phục experiment qua DagsHub UI.")
                new_experiment_name = st.text_input("Nhập tên Experiment mới", value=f"{experiment_name}_Restored_{datetime.datetime.now().strftime('%Y%m%d')}")
                if new_experiment_name:
                    mlflow.set_experiment(new_experiment_name)
                    experiment_name = new_experiment_name
                else:
                    st.error("Vui lòng nhập tên experiment mới để tiếp tục.")
                    return
            else:
                mlflow.set_experiment(experiment_name)
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
            return

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán")
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            with st.spinner("Đang tải dữ liệu đã xử lý..."):
                data = load_cached_data(processed_file)
            X_full = data  # Sử dụng tất cả các cột, không loại bỏ cột nào
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
            return

        # Lấy danh sách runs từ experiment đã chọn
        st.write(f"Chọn mô hình đã huấn luyện từ Experiment '{experiment_name}':")
        with st.spinner("Đang tải danh sách runs từ DagsHub..."):
            runs = mlflow.search_runs(experiment_ids=[experiments[experiment_name]])
        if runs.empty:
            st.error(f"Không tìm thấy mô hình nào trong experiment '{experiment_name}'. Vui lòng huấn luyện mô hình trong 'train.py' trước.")
            return

        run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')}" for _, run in runs.iterrows()]
        selected_run = st.selectbox("Chọn run chứa mô hình", options=run_options)
        selected_run_id = selected_run.split("ID Run: ")[1].split(" - ")[0]

        # Tải mô hình từ MLflow
        try:
            with st.spinner("Đang tải mô hình từ DagsHub MLflow..."):
                model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
            st.session_state['model'] = model
            st.write(f"Mô hình đã được tải từ MLflow (Run ID: {selected_run_id})")
        except Exception as e:
            st.error(f"Không thể tải mô hình từ Run ID {selected_run_id}: {str(e)}")
            return

        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()

        st.write("Nhập giá trị cho tất cả các cột (dựa trên dữ liệu huấn luyện của mô hình):")
        input_data = {}
        for col in expected_columns:
            if col in X_full.columns:
                if col == 'Age':
                    input_data[col] = st.number_input(
                        f"Nhập giá trị cho '{col}' (0-150, số nguyên)",
                        min_value=0,
                        max_value=150,
                        value=int(X_full[col].mean()) if not pd.isna(X_full[col].mean()) else 0,
                        step=1,
                        key=f"input_{col}"
                    )
                elif col == 'Sex':
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}' (0=nam, 1=nữ)",
                        options=[0, 1],
                        key=f"input_{col}"
                    )
                elif col.startswith('Embarked_') or col in ['Pclass', 'SibSp', 'Parch', 'Name', 'PassengerId', 'Survived', 'Ticket', 'Cabin', 'Fare']:
                    unique_vals = X_full[col].unique().tolist()
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}'",
                        options=unique_vals,
                        key=f"input_{col}"
                    )
                else:
                    min_val = float(X_full[col].min()) if not pd.isna(X_full[col].min()) else 0
                    max_val = float(X_full[col].max()) if not pd.isna(X_full[col].max()) else 100
                    input_data[col] = st.number_input(
                        f"Nhập giá trị cho '{col}' (Khoảng: {min_val} đến {max_val})",
                        min_value=min_val,
                        max_value=max_val,
                        value=float(X_full[col].mean()) if not pd.isna(X_full[col].mean()) else (min_val + max_val) / 2,
                        key=f"input_{col}"
                    )
            else:
                input_data[col] = 0

        X_selected = pd.DataFrame([input_data])
        st.write("Dữ liệu Nhập của Bạn cho Dự đoán:")
        st.write(X_selected)

        if st.button("Thực hiện Dự đoán"):
            with st.spinner("Đang thực hiện dự đoán..."):
                try:
                    predictions = model.predict(X_selected)
                    result_df = pd.DataFrame({"Dự đoán Sinh tồn": predictions})
                    st.write("Kết quả Dự đoán:")
                    st.write(result_df)

                    st.write("Dữ liệu Nhập của Bạn (Tóm tắt):")
                    st.write(pd.DataFrame([input_data]))

                    # Logging dự đoán vào MLflow trên DagsHub với tên và ID cụ thể
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    demo_run_name = f"Demo_Predict_{timestamp}"
                    if st.button("Log Dự đoán vào MLflow"):
                        with st.spinner("Đang log dữ liệu dự đoán vào DagsHub..."):
                            with mlflow.start_run(run_name=demo_run_name) as run:
                                for col, value in input_data.items():
                                    mlflow.log_param(f"input_{col}", value)
                                mlflow.log_param("predicted_survival", predictions[0])
                                mlflow.log_param("demo_name", "Titanic_Survival_Demo")
                                mlflow.log_param("demo_run_id", run.info.run_id)

                                st.write("Thông tin Đã Log:")
                                log_info = {
                                    "Tên Run": demo_run_name,
                                    "ID Run": run.info.run_id,
                                    "Dữ liệu Nhập": input_data,
                                    "Dự đoán Sinh tồn": predictions[0],
                                    "Tên Demo": "Titanic_Survival_Demo"
                                }
                                st.write(log_info)

                                run_id = run.info.run_id
                                mlflow_uri = st.session_state['mlflow_url']
                                st.success(f"Dự đoán đã được log thành công!\n- Experiment: '{experiment_name}'\n- Tên Run: '{demo_run_name}'\n- ID Run: {run_id}")
                                st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")

                                st.write("Bạn muốn làm gì tiếp theo?")
                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button("Xem run này trong 'Xem Kết quả Đã Log'"):
                                        st.session_state['selected_run_id'] = run_id
                                        st.session_state['active_tab'] = 1
                                        st.info("Vui lòng chuyển sang tab 'Xem Kết quả Đã Log'.")
                                with col2:
                                    if st.button("Xóa run này trong 'Xóa Log'"):
                                        st.session_state['selected_run_id'] = run_id
                                        st.session_state['active_tab'] = 2
                                        st.info("Vui lòng chuyển sang tab 'Xóa Log'.")

                except ValueError as e:
                    st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo dữ liệu nhập khớp với các cột: {expected_columns}")

    with tab2:
        st.subheader("Xem Kết quả Đã Log")
        with st.spinner("Đang tải danh sách runs từ DagsHub..."):
            runs = mlflow.search_runs()
        if runs.empty:
            st.write("Chưa có run dự đoán nào được log.")
        else:
            st.write("Danh sách các run đã log:")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
                columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
            )
            st.write(display_runs)

            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0])
            selected_run_id = st.selectbox(
                "Chọn một run để xem chi tiết",
                options=runs['run_id'].tolist(),
                index=runs['run_id'].tolist().index(default_run) if default_run in runs['run_id'].tolist() else 0
            )
            if selected_run_id:
                with st.spinner("Đang tải chi tiết run từ DagsHub..."):
                    run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                    st.write("Chi tiết Run:")
                    st.write(f"ID Run: {run_details['run_id']}")
                    st.write(f"Tên Run: {run_details.get('tags.mlflow.runName', 'Không tên')}")
                    st.write(f"ID Experiment: {run_details['experiment_id']}")
                    st.write(f"Thời gian Bắt đầu: {run_details['start_time']}")

                    st.write("Thông số Đã Log:")
                    params = mlflow.get_run(selected_run_id).data.params
                    st.write(params)

                    mlflow_uri = st.session_state['mlflow_url']
                    st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({mlflow_uri})")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        with st.spinner("Đang tải danh sách runs từ DagsHub..."):
            runs = mlflow.search_runs()
        if runs.empty:
            st.write("Không có run nào để xóa.")
        else:
            st.write("Chọn các run để xóa:")
            run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']})" 
                          for _, run in runs.iterrows()]
            default_delete = [f"ID Run: {st.session_state['selected_run_id']} - {runs[runs['run_id'] == st.session_state['selected_run_id']]['tags.mlflow.runName'].iloc[0]} (Exp: {runs[runs['run_id'] == st.session_state['selected_run_id']]['experiment_id'].iloc[0]})" 
                             if 'selected_run_id' in st.session_state and st.session_state['selected_run_id'] in runs['run_id'].tolist() else None]
            runs_to_delete = st.multiselect(
                "Chọn các run",
                options=run_options,
                default=[d for d in default_delete if d],
                key="delete_runs"
            )
            if st.button("Xóa Các Run Đã Chọn"):
                for run_str in runs_to_delete:
                    run_id = run_str.split("ID Run: ")[1].split(" - ")[0]
                    try:
                        with st.spinner(f"Đang xóa Run {run_id}..."):
                            mlflow.delete_run(run_id)
                        st.success(f"Đã xóa run có ID: {run_id}")
                    except mlflow.exceptions.MlflowException as e:
                        st.error(f"Không thể xóa run {run_id}: {str(e)}")
                st.success("Các run đã chọn đã được xóa. Làm mới trang để cập nhật danh sách.")

if __name__ == "__main__":
    show_demo()