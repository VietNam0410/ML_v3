import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub

# Thiết lập thông tin DagsHub
DAGSHUB_USERNAME = "VietNam0410"
DAGSHUB_REPO = "vn0410"

try:
    dagshub.init(repo_owner=DAGSHUB_USERNAME, repo_name=DAGSHUB_REPO, mlflow=True)
    mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}.mlflow")
    os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USERNAME
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    st.success("Đã kết nối với DagsHub thành công!")
except Exception as e:
    st.error(f"Không thể kết nối với DagsHub: {str(e)}. Sử dụng MLflow cục bộ.")
    mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def get_mlflow_experiments():
    try:
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        return {exp.name: exp.experiment_id for exp in experiments}
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các experiment từ MLflow: {str(e)}")
        return {}

def get_mlflow_runs():
    try:
        runs = mlflow.search_runs()
        return runs
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
        return pd.DataFrame()

def delete_mlflow_run(run_id):
    try:
        mlflow.delete_run(run_id)
        st.success(f"Đã xóa run có ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể xóa run {run_id}: {str(e)}")

def show_demo():
    st.header("Dự đoán Sinh tồn Titanic")

    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    experiments = get_mlflow_experiments()
    experiment_options = list(experiments.keys()) if experiments else ["Titanic_Demo"]
    experiment_name = st.selectbox(
        "Chọn hoặc nhập tên Experiment cho Demo",
        options=experiment_options,
        index=experiment_options.index("Titanic_Demo") if "Titanic_Demo" in experiment_options else 0,
        help="Chọn một experiment hiện có hoặc nhập tên mới."
    )
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán")
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            data = load_data(processed_file)
            X_full = data.drop(columns=['Survived', 'Name'] if 'Name' in data.columns else ['Survived'])
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
        else:
            runs = get_mlflow_runs()
            if runs.empty:
                st.error("Không tìm thấy mô hình đã huấn luyện trong MLflow. Vui lòng huấn luyện mô hình trước.")
            else:
                model_options = {f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']})": run['run_id'] 
                                for _, run in runs.iterrows()}
                selected_model_name = st.selectbox("Chọn một mô hình đã huấn luyện", options=list(model_options.keys()))
                selected_run_id = model_options[selected_model_name]

                try:
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()
                except:
                    st.error(f"Không thể tải mô hình với ID Run: {selected_run_id}. Vui lòng kiểm tra MLflow.")
                else:
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
                                    step=1,
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
                            input_data[col] = 0

                    X_selected = pd.DataFrame([input_data])
                    st.write("Dữ liệu Nhập của Bạn cho Dự đoán:")
                    st.write(X_selected)

                    if st.button("Thực hiện Dự đoán"):
                        try:
                            predictions = model.predict(X_selected)
                            result_df = pd.DataFrame({"Dự đoán Sinh tồn": predictions})
                            st.write("Kết quả Dự đoán:")
                            st.write(result_df)

                            st.write("Dữ liệu Nhập của Bạn (Tóm tắt):")
                            st.write(pd.DataFrame([input_data]))

                            run_name = st.text_input("Nhập tên cho run dự đoán này", value="Run_Dự_đoán")
                            if st.button("Log Dự đoán vào MLflow"):
                                with mlflow.start_run(run_name=run_name) as run:
                                    for col, value in input_data.items():
                                        mlflow.log_param(f"input_{col}", value)
                                    mlflow.log_param("model_run_id", selected_run_id)
                                    mlflow.log_param("predicted_survival", predictions[0])
                                    mlflow.log_param("run_name", run_name)
                                    mlflow.log_param("run_id", run.info.run_id)

                                    st.write("Thông tin Đã Log:")
                                    log_info = {
                                        "Tên Run": run_name,
                                        "ID Run": run.info.run_id,
                                        "Dữ liệu Nhập": input_data,
                                        "ID Run Mô hình": selected_run_id,
                                        "Dự đoán Sinh tồn": predictions[0]
                                    }
                                    st.write(log_info)

                                    run_id = run.info.run_id
                                    dagshub_link = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments/#/experiment/{experiment_name}/{run_id}"
                                    st.success(f"Dự đoán đã được log thành công!\n- Experiment: '{experiment_name}'\n- Tên Run: '{run_name}'\n- ID Run: {run_id}")
                                    st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")

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

                dagshub_link = f"https://dagshub.com/{DAGSHUB_USERNAME}/{DAGSHUB_REPO}/experiments/#/experiment/{experiment_name}/{selected_run_id}"
                st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({dagshub_link})")

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