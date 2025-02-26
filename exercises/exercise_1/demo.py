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
        st.error(f"Failed to fetch experiments from MLflow: {str(e)}")
        return {}

def get_mlflow_runs():
    """Lấy danh sách các run từ MLflow cục bộ."""
    try:
        runs = mlflow.search_runs()
        return runs
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Failed to fetch runs from MLflow: {str(e)}")
        return pd.DataFrame()

def delete_mlflow_run(run_id):
    """Xóa một run từ MLflow."""
    try:
        mlflow.delete_run(run_id)
        st.success(f"Deleted run with ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Failed to delete run {run_id}: {str(e)}")

def show_demo():
    st.header("Titanic Survival Prediction Demo")

    # Lấy danh sách Experiment hiện có
    experiments = get_mlflow_experiments()
    experiment_options = list(experiments.keys()) if experiments else ["Titanic_Demo"]

    # Cho người dùng chọn hoặc nhập tên Experiment
    experiment_name = st.selectbox(
        "Select or enter Experiment Name for Demo",
        options=experiment_options,
        index=experiment_options.index("Titanic_Demo") if "Titanic_Demo" in experiment_options else 0,
        help="Choose an existing experiment or type a new one."
    )
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Tạo các tab
    tab1, tab2, tab3 = st.tabs(["Make Predictions", "View Logged Results", "Delete Logs"])

    # Tab 1: Dự đoán
    with tab1:
        st.subheader("Step 1: Customize Input Data for Prediction")

        # Load dữ liệu đã tiền xử lý để lấy thông tin cột
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            data = load_data(processed_file)
            X_full = data.drop(columns=['Survived'])
        except FileNotFoundError:
            st.error("Processed data not found. Please preprocess the data first.")
        else:
            # Lấy danh sách mô hình từ MLflow
            runs = get_mlflow_runs()
            if runs.empty:
                st.error("No trained models found in MLflow. Please train a model first.")
            else:
                model_options = {f"Run ID: {run['run_id']} - {run.get('tags.mlflow.runName', 'Unnamed')} (Exp: {run['experiment_id']})": run['run_id'] 
                                 for _, run in runs.iterrows()}
                selected_model_name = st.selectbox("Select a trained model", options=list(model_options.keys()))
                selected_run_id = model_options[selected_model_name]

                # Load mô hình từ MLflow để lấy thông tin cột huấn luyện
                try:
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()
                except:
                    st.error(f"Failed to load model with Run ID: {selected_run_id}. Please check MLflow.")
                else:
                    # Cho người dùng nhập dữ liệu với kiểm soát miền giá trị và kiểu dữ liệu
                    st.write("Enter values for each column (based on the model's training data):")
                    input_data = {}
                    for col in expected_columns:
                        if col in X_full.columns:
                            if col == 'Age':
                                input_data[col] = st.number_input(
                                    f"Enter value for '{col}' (0-100)",
                                    min_value=0.0,
                                    max_value=100.0,
                                    value=float(X_full[col].mean()),
                                    key=f"input_{col}"
                                )
                            elif col == 'Sex':
                                input_data[col] = st.selectbox(
                                    f"Choose value for '{col}' (0=male, 1=female)",
                                    options=[0, 1],
                                    key=f"input_{col}"
                                )
                            elif col.startswith('Embarked_') or col in ['Pclass', 'SibSp', 'Parch']:
                                unique_vals = X_full[col].unique().tolist()
                                input_data[col] = st.selectbox(
                                    f"Choose value for '{col}'",
                                    options=unique_vals,
                                    key=f"input_{col}"
                                )
                            else:
                                min_val = float(X_full[col].min())
                                max_val = float(X_full[col].max())
                                input_data[col] = st.number_input(
                                    f"Enter value for '{col}' (Range: {min_val} to {max_val})",
                                    min_value=min_val,
                                    max_value=max_val,
                                    value=float(X_full[col].mean()),
                                    key=f"input_{col}"
                                )
                        else:
                            input_data[col] = 0  # Cột từ One-Hot Encoding không có trong X_full

                    # Tạo DataFrame từ dữ liệu người dùng nhập
                    X_selected = pd.DataFrame([input_data])

                    # Hiển thị dữ liệu đã chọn
                    st.write("Your Input Data for Prediction:")
                    st.write(X_selected)

                    # Dự đoán và log
                    if st.button("Make Predictions"):
                        try:
                            predictions = model.predict(X_selected)
                            result_df = pd.DataFrame({
                                "Predicted Survival": predictions
                            })
                            st.write("Prediction Result:")
                            st.write(result_df)

                            # Hiển thị lại thông tin nhập
                            st.write("Your Input Data (Recap):")
                            st.write(pd.DataFrame([input_data]))

                            # Cho người dùng đặt tên run
                            run_name = st.text_input("Enter a name for this prediction run", value="Prediction_Run")
                            if st.button("Log Predictions to MLflow"):
                                # Kiểm tra và kết thúc run hiện tại nếu có
                                # Sửa đổi bởi Grok 3: Thêm kiểm tra run hiện tại và kết thúc nếu cần
                                active_run = mlflow.active_run()
                                if active_run:
                                    mlflow.end_run()

                                with mlflow.start_run(run_name=run_name):
                                    # Log tất cả thông tin đầu vào
                                    for col, value in input_data.items():
                                        mlflow.log_param(f"input_{col}", value)
                                    mlflow.log_param("model_run_id", selected_run_id)
                                    mlflow.log_param("predicted_survival", predictions[0])
                                    mlflow.log_param("run_name", run_name)
                                    mlflow.log_param("run_timestamp", pd.Timestamp.now().isoformat())

                                    # Log dữ liệu mẫu
                                    mlflow.log_text(X_selected.to_csv(), "input_data.csv")
                                    mlflow.log_text(result_df.to_csv(), "prediction_result.csv")

                                    # Thêm thông tin về run huấn luyện
                                    model_run = mlflow.get_run(selected_run_id)
                                    mlflow.log_param("training_run_id", selected_run_id)
                                    mlflow.log_param("training_experiment_id", model_run.info.experiment_id)

                                    # Hiển thị thông tin log
                                    st.write("Logged Information:")
                                    log_info = {
                                        "Run Name": run_name,
                                        "Run ID": mlflow.active_run().info.run_id,
                                        "Input Data": input_data,
                                        "Model Run ID": selected_run_id,
                                        "Predicted Survival": predictions[0]
                                    }
                                    st.write(log_info)

                                    # Tạo liên kết MLflow
                                    experiment_id = experiments[experiment_name]
                                    mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments/{experiment_id}/runs/{mlflow.active_run().info.run_id}"
                                    st.success(f"Prediction logged successfully to MLflow!\n- Experiment: '{experiment_name}'\n- Run Name: '{run_name}'\n- Run ID: {mlflow.active_run().info.run_id}\n- Link: [View in MLflow UI]({mlflow_ui_link})")

                                    # Cập nhật session_state để chuyển tab
                                    st.session_state['selected_run_id'] = mlflow.active_run().info.run_id

                                    # Liên kết tới các tab khác
                                    st.write("What would you like to do next?")
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        if st.button("View this run in 'View Logged Results'"):
                                            st.session_state['selected_run_id'] = mlflow.active_run().info.run_id
                                            st.session_state['active_tab'] = 1
                                            st.info("Please switch to the 'View Logged Results' tab to see this run.")
                                    with col2:
                                        if st.button("Delete this run in 'Delete Logs'"):
                                            st.session_state['selected_run_id'] = mlflow.active_run().info.run_id
                                            st.session_state['active_tab'] = 2
                                            st.info("Please switch to the 'Delete Logs' tab to delete this run.")

                        except ValueError as e:
                            st.error(f"Prediction failed: {str(e)}. Ensure input data matches the model's expected columns: {expected_columns}")

    # Tab 2: Hiển thị thông tin log
    with tab2:
        st.subheader("Logged Prediction Results")
        runs = get_mlflow_runs()
        if runs.empty:
            st.write("No prediction runs logged yet.")
        else:
            st.write("List of logged runs:")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id', 'params.predicted_survival']].rename(
                columns={'tags.mlflow.runName': 'Run Name', 'start_time': 'Start Time', 'experiment_id': 'Experiment ID', 'params.predicted_survival': 'Prediction'}
            )
            st.write(display_runs)

            # Tự động chọn run vừa log nếu có trong session_state
            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0])
            selected_run_id = st.selectbox(
                "Select a run to view details",
                options=runs['run_id'].tolist(),
                index=runs['run_id'].tolist().index(default_run) if default_run in runs['run_id'].tolist() else 0
            )
            if selected_run_id:
                run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                st.write("Run Details:")
                st.write(f"Run ID: {run_details['run_id']}")
                st.write(f"Run Name: {run_details.get('tags.mlflow.runName', 'Unnamed')}")
                st.write(f"Experiment ID: {run_details['experiment_id']}")
                st.write(f"Start Time: {run_details['start_time']}")
                st.write(f"Prediction: {run_details.get('params.predicted_survival', 'N/A')}")

                st.write("Logged Parameters:")
                params = mlflow.get_run(selected_run_id).data.params
                st.write(params)

                mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments/{run_details['experiment_id']}/runs/{selected_run_id}"
                st.write(f"View this run in MLflow UI: [Click here]({mlflow_ui_link})")

    # Tab 3: Xóa log
    with tab3:
        st.subheader("Delete Unnecessary Logs")
        runs = get_mlflow_runs()
        if runs.empty:
            st.write("No runs available to delete.")
        else:
            st.write("Select runs to delete:")
            run_options = [f"Run ID: {run['run_id']} - {run.get('tags.mlflow.runName', 'Unnamed')} (Exp: {run['experiment_id']})" 
                           for _, run in runs.iterrows()]
            default_delete = [f"Run ID: {st.session_state['selected_run_id']} - {runs[runs['run_id'] == st.session_state['selected_run_id']]['tags.mlflow.runName'].iloc[0]} (Exp: {runs[runs['run_id'] == st.session_state['selected_run_id']]['experiment_id'].iloc[0]})" 
                             if 'selected_run_id' in st.session_state and st.session_state['selected_run_id'] in runs['run_id'].tolist() else None]
            runs_to_delete = st.multiselect(
                "Choose runs",
                options=run_options,
                default=[d for d in default_delete if d],
                key="delete_runs"
            )
            if st.button("Delete Selected Runs"):
                for run_str in runs_to_delete:
                    run_id = run_str.split("Run ID: ")[1].split(" - ")[0]
                    delete_mlflow_run(run_id)
                st.success("Selected runs have been deleted. Refresh the page to update the list.")

if __name__ == "__main__":
    show_demo()