import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os

# Thiết lập tracking URI cục bộ
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def get_mlflow_experiments():
    """Lấy danh sách các Experiment từ MLflow."""
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {exp.name: exp.experiment_id for exp in experiments}
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các experiment từ MLflow: {str(e)}")
        return {}

def get_mlflow_runs():
    """Lấy danh sách các run từ MLflow cục bộ."""
    try:
        runs = mlflow.search_runs()
        return runs
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return pd.DataFrame()

def delete_mlflow_run(run_id):
    """Xóa một run từ MLflow."""
    try:
        mlflow.delete_run(run_id)
        st.success(f"Đã xóa run có ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể xóa run {run_id}: {str(e)}")

def show_demo():
    st.header("Dự đoán Sinh tồn Titanic")

    # Lấy danh sách Experiment hiện có
    experiments = get_mlflow_experiments()
    experiment_options = list(experiments.keys()) if experiments else ["Titanic_Demo"]

    # Cho người dùng chọn hoặc nhập tên Experiment
    experiment_name = st.selectbox(
        "Chọn hoặc nhập tên Experiment cho Demo",
        options=experiment_options,
        index=experiment_options.index("Titanic_Demo") if "Titanic_Demo" in experiment_options else 0,
        help="Chọn một experiment hiện có hoặc nhập tên mới."
    )
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Tạo các tab
    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    # Tab 1: Dự đoán
    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán")

        # Load dữ liệu đã tiền xử lý để lấy thông tin cột
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            data = load_data(processed_file)
            X_full = data.drop(columns=['Survived', 'Name'] if 'Name' in data.columns else ['Survived'])
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
        else:
            # Lấy danh sách mô hình từ MLflow
            runs = get_mlflow_runs()
            if runs.empty:
                st.error("Không tìm thấy mô hình đã huấn luyện trong MLflow. Vui lòng huấn luyện mô hình trước.")
            else:
                model_options = {f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']})": run['run_id'] 
                                for _, run in runs.iterrows()}
                selected_model_name = st.selectbox("Chọn một mô hình đã huấn luyện", options=list(model_options.keys()))
                selected_run_id = model_options[selected_model_name]

                # Load mô hình từ MLflow để lấy thông tin cột huấn luyện
                try:
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()
                except:
                    st.error(f"Không thể tải mô hình với ID Run: {selected_run_id}. Vui lòng kiểm tra MLflow.")
                else:
                    # Cho người dùng nhập dữ liệu với kiểm soát miền giá trị và kiểu dữ liệu
                    st.write("Nhập giá trị cho từng cột (dựa trên dữ liệu huấn luyện của mô hình):")
                    input_data = {}
                    for col in expected_columns:
                        if col in X_full.columns:
                            if col == 'Age':
                                input_data[col] = st.number_input(
                                    f"Nhập giá trị cho '{col}' (0-150, số nguyên)",
                                    min_value=0,
                                    max_value=150,
                                    value=int(X_full[col].mean()) if not pd.isna(X_full[col].mean()) else 0,
                                    step=1,  # Đảm bảo chỉ nhập số nguyên
                                    key=f"input_{col}"
                                )
                            elif col == 'Sex':
                                input_data[col] = st.selectbox(
                                    f"Chọn giá trị cho '{col}' (0=nam, 1=nữ)",
                                    options=[0, 1],
                                    key=f"input_{col}"
                                )
                            elif col.startswith('Embarked_') or col in ['Pclass', 'SibSp', 'Parch']:
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
                            input_data[col] = 0  # Cột từ One-Hot Encoding không có trong X_full

                    # Tạo DataFrame từ dữ liệu người dùng nhập
                    X_selected = pd.DataFrame([input_data])

                    # Hiển thị dữ liệu đã chọn
                    st.write("Dữ liệu Nhập của Bạn cho Dự đoán:")
                    st.write(X_selected)

                    # Dự đoán và log
                    if st.button("Thực hiện Dự đoán"):
                        try:
                            predictions = model.predict(X_selected)
                            result_df = pd.DataFrame({
                                "Dự đoán Sinh tồn": predictions
                            })
                            st.write("Kết quả Dự đoán:")
                            st.write(result_df)

                            # Hiển thị lại thông tin nhập
                            st.write("Dữ liệu Nhập của Bạn (Tóm tắt):")
                            st.write(pd.DataFrame([input_data]))

                            # Cho người dùng đặt tên run
                            run_name = st.text_input("Nhập tên cho run dự đoán này", value="Run_Dự_đoán")
                            if st.button("Log Dự đoán vào MLflow"):
                                with mlflow.start_run(run_name=run_name) as run:
                                    # Log tất cả thông tin dưới dạng parameters
                                    for col, value in input_data.items():
                                        mlflow.log_param(f"input_{col}", value)
                                    mlflow.log_param("model_run_id", selected_run_id)
                                    mlflow.log_param("predicted_survival", predictions[0])
                                    mlflow.log_param("run_name", run_name)
                                    mlflow.log_param("run_id", run.info.run_id)

                                    # Hiển thị bản log
                                    st.write("Thông tin Đã Log:")
                                    log_info = {
                                        "Tên Run": run_name,
                                        "ID Run": run.info.run_id,
                                        "Dữ liệu Nhập": input_data,
                                        "ID Run Mô hình": selected_run_id,
                                        "Dự đoán Sinh tồn": predictions[0]
                                    }
                                    st.write(log_info)

                                    # Thông báo log predict kèm link MLflow
                                    run_id = run.info.run_id
                                    experiment_id = experiments[experiment_name]
                                    mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{run_id}"
                                    st.success(f"Dự đoán đã được log thành công vào MLflow!\n- Experiment: '{experiment_name}'\n- Tên Run: '{run_name}'\n- ID Run: {run_id}\n- Link: [Xem trong MLflow UI]({mlflow_ui_link})")

                                    # Liên kết tới các tab khác
                                    st.write("Bạn muốn làm gì tiếp theo?")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("Xem run này trong 'Xem Kết quả Đã Log'"):
                                            st.session_state['selected_run_id'] = run_id
                                            st.session_state['active_tab'] = 1
                                            st.info("Vui lòng chuyển sang tab 'Xem Kết quả Đã Log' để xem run này.")
                                    with col2:
                                        if st.button("Xóa run này trong 'Xóa Log'"):
                                            st.session_state['selected_run_id'] = run_id
                                            st.session_state['active_tab'] = 2
                                            st.info("Vui lòng chuyển sang tab 'Xóa Log' để xóa run này.")

                        except ValueError as e:
                            st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo dữ liệu nhập khớp với các cột mong đợi của mô hình: {expected_columns}")

    # Tab 2: Hiển thị thông tin log
    with tab2:
        st.subheader("Kết quả Dự đoán Đã Log")
        runs = get_mlflow_runs()
        if runs.empty:
            st.write("Chưa có run dự đoán nào được log.")
        else:
            st.write("Danh sách các run đã log:")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
                columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
            )
            st.write(display_runs)

            # Tự động chọn run vừa log nếu có trong session_state
            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0])
            selected_run_id = st.selectbox(
                "Chọn một run để xem chi tiết",
                options=runs['run_id'].tolist(),
                index=runs['run_id'].tolist().index(default_run) if default_run in runs['run_id'].tolist() else 0
            )
            if selected_run_id:
                run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                st.write("Chi tiết Run:")
                st.write(f"ID Run: {run_details['run_id']}")
                st.write(f"Tên Run: {run_details.get('tags.mlflow.runName', 'Không tên')}")
                st.write(f"ID Experiment: {run_details['experiment_id']}")
                st.write(f"Thời gian Bắt đầu: {run_details['start_time']}")

                st.write("Thông số Đã Log:")
                params = mlflow.get_run(selected_run_id).data.params
                st.write(params)

                mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments/{run_details['experiment_id']}/runs/{selected_run_id}"
                st.write(f"Xem run này trong MLflow UI: [Nhấn vào đây]({mlflow_ui_link})")

    # Tab 3: Xóa log
    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        runs = get_mlflow_runs()
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
                    delete_mlflow_run(run_id)
                st.success("Các run đã chọn đã được xóa. Làm mới trang để cập nhật danh sách.")

if __name__ == "__main__":
    show_demo()