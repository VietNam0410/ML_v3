# import streamlit as st
# import pandas as pd
# import mlflow
# import mlflow.sklearn
# from common.utils import load_data
# import os

# mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")


# def get_mlflow_runs():
#     """Lấy danh sách các run từ MLflow."""
#     runs = mlflow.search_runs(experiment_ids=["0"])  # Giả sử experiment_id mặc định là "0"
#     return runs

# def show_demo():
#     st.header("Titanic Survival Prediction Demo")

#     # Tạo các tab
#     tab1, tab2 = st.tabs(["Make Predictions", "View Logged Results"])

#     # Tab 1: Dự đoán
#     with tab1:
#         st.subheader("Step 1: Select Data and Model for Prediction")

#         # Load dữ liệu đã tiền xử lý
#         processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
#         try:
#             data = load_data(processed_file)
#             X_full = data.drop(columns=['Survived'])
#             y_true_full = data['Survived']
#         except FileNotFoundError:
#             st.error("Processed data not found. Please preprocess the data first.")
#             return

#         # Cho người dùng chọn cột
#         st.write("Select columns for prediction (Survived is the target):")
#         selected_columns = st.multiselect(
#             "Choose columns",
#             options=X_full.columns.tolist(),
#             default=X_full.columns.tolist()  # Mặc định chọn tất cả
#         )
        
#         # Cho người dùng chọn số lượng hàng
#         num_rows = st.slider(
#             "Select number of rows to predict",
#             min_value=1,
#             max_value=len(X_full),
#             value=min(10, len(X_full))  # Mặc định chọn 10 hoặc tổng số hàng nếu ít hơn
#         )

#         # Cập nhật dữ liệu dựa trên lựa chọn
#         X_selected = X_full[selected_columns].iloc[:num_rows]
#         y_true_selected = y_true_full.iloc[:num_rows]

#         # Hiển thị dữ liệu đã chọn
#         st.write("Selected Data Preview:")
#         st.write(X_selected)

#         # Lấy danh sách mô hình từ MLflow
#         runs = get_mlflow_runs()
#         model_options = {f"Run ID: {run['run_id']} - {run['tags.mlflow.runName']}": run['run_id'] 
#                          for _, run in runs.iterrows() if 'tags.mlflow.runName' in run}
#         if not model_options:
#             st.error("No trained models found in MLflow. Please train a model first.")
#             return

#         # Cho người dùng chọn mô hình
#         selected_model_name = st.selectbox("Select a trained model", options=list(model_options.keys()))
#         selected_run_id = model_options[selected_model_name]

#         # Load mô hình từ MLflow
#         try:
#             model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
#         except:
#             st.error(f"Failed to load model with Run ID: {selected_run_id}. Please check MLflow.")
#             return

#         # Dự đoán và log
#         if st.button("Make Predictions"):
#             predictions = model.predict(X_selected)
#             result_df = pd.DataFrame({
#                 "True Survival": y_true_selected,
#                 "Predicted Survival": predictions
#             })
#             st.write("Prediction Results:")
#             st.write(result_df)

#             # Cho người dùng đặt tên run
#             run_name = st.text_input("Enter a name for this prediction run", value="Prediction_Run")
#             if st.button("Log Predictions to MLflow"):
#                 with mlflow.start_run(run_name=run_name):
#                     # Log dữ liệu đầu vào
#                     mlflow.log_param("selected_columns", selected_columns)
#                     mlflow.log_param("num_rows", num_rows)
#                     mlflow.log_param("model_run_id", selected_run_id)

#                     # Log kết quả dự đoán
#                     result_df.to_csv("temp_predictions.csv", index=False)
#                     mlflow.log_artifact("temp_predictions.csv", "predictions")
#                     os.remove("temp_predictions.csv")  # Xóa file tạm sau khi log

#                     st.success(f"Predictions logged to MLflow under run name: '{run_name}'")

#     # Tab 2: Hiển thị thông tin log
#     with tab2:
#         st.subheader("Logged Prediction Results")
#         runs = get_mlflow_runs()
#         if runs.empty:
#             st.write("No prediction runs logged yet.")
#         else:
#             st.write("List of logged runs:")
#             display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time']].rename(
#                 columns={'tags.mlflow.runName': 'Run Name', 'start_time': 'Start Time'}
#             )
#             st.write(display_runs)

#             # Cho phép người dùng chọn run để xem chi tiết
#             selected_run_id = st.selectbox("Select a run to view details", options=runs['run_id'].tolist())
#             if selected_run_id:
#                 run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
#                 st.write("Run Details:")
#                 st.write(f"Run ID: {run_details['run_id']}")
#                 st.write(f"Run Name: {run_details.get('tags.mlflow.runName', 'Unnamed')}")
#                 st.write(f"Start Time: {run_details['start_time']}")

#                 # Hiển thị tham số và artifact nếu có
#                 st.write("Logged Parameters:")
#                 params = mlflow.get_run(selected_run_id).data.params
#                 st.write(params)

#                 st.write("Logged Artifacts (Predictions):")
#                 artifacts = mlflow.artifacts.list_artifacts(selected_run_id)
#                 for artifact in artifacts:
#                     if artifact.path.endswith(".csv"):
#                         artifact_file = mlflow.artifacts.download_artifacts(run_id=selected_run_id, artifact_path=artifact.path)
#                         artifact_data = pd.read_csv(artifact_file)
#                         st.write(f"Artifact: {artifact.path}")
#                         st.write(artifact_data.head())

# if __name__ == "__main__":
#     show_demo()
import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os


# Thiết lập tracking URI cục bộ
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def get_mlflow_runs():
    """Lấy danh sách các run từ MLflow cục bộ."""
    try:
        runs = mlflow.search_runs(experiment_ids=["0"])
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
        return

    # Lấy danh sách mô hình từ MLflow
    runs = get_mlflow_runs()
    if runs.empty:
        st.error("No trained models found in MLflow. Please train a model first.")
        return

    model_options = {f"Run ID: {run['run_id']} - {run.get('tags.mlflow.runName', 'Unnamed')}": run['run_id'] 
                     for _, run in runs.iterrows()}
    selected_model_name = st.selectbox("Select a trained model", options=list(model_options.keys()))
    selected_run_id = model_options[selected_model_name]

    # Load mô hình từ MLflow để lấy thông tin cột huấn luyện
    try:
        model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()
    except:
        st.error(f"Failed to load model with Run ID: {selected_run_id}. Please check MLflow.")
        return

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
                with mlflow.start_run(run_name=run_name) as run:
                    # Log dữ liệu đầu vào
                    mlflow.log_param("input_data", input_data)
                    mlflow.log_param("model_run_id", selected_run_id)

                    # Log kết quả dự đoán
                    result_df.to_csv("temp_predictions.csv", index=False)
                    mlflow.log_artifact("temp_predictions.csv", "predictions")
                    os.remove("temp_predictions.csv")

                    # Hiển thị bản log
                    st.write("Logged Information:")
                    log_info = {
                        "Run Name": run_name,
                        "Run ID": run.info.run_id,
                        "Input Data": input_data,
                        "Model Run ID": selected_run_id
                    }
                    st.write(log_info)

                    # Thông báo và liên kết tới các tab khác
                    run_id = run.info.run_id
                    mlflow_ui_link = f"http://127.0.0.1:5000/#/experiments/0/runs/{run_id}"
                    st.success(f"Predictions logged to MLflow under run name: '{run_name}' (Run ID: {run_id})")
                    st.write(f"View your run in MLflow UI: [Click here]({mlflow_ui_link})")
                    st.write("What would you like to do next?")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("View this run in 'View Logged Results'"):
                            st.session_state['selected_run_id'] = run_id
                            st.session_state['active_tab'] = 1  # Chuyển sang Tab 2
                    with col2:
                        if st.button("Delete this run in 'Delete Logs'"):
                            st.session_state['selected_run_id'] = run_id
                            st.session_state['active_tab'] = 2  # Chuyển sang Tab 3

        except ValueError as e:
            st.error(f"Prediction failed: {str(e)}. Ensure input data matches the model's expected columns: {expected_columns}")

# (Giữ nguyên phần code của Tab 2 và Tab 3 từ phiên bản trước)
    # Tab 2: Hiển thị thông tin log
    with tab2:
        st.subheader("Logged Prediction Results")
        runs = get_mlflow_runs()
        if runs.empty:
            st.write("No prediction runs logged yet.")
        else:
            st.write("List of logged runs:")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time']].rename(
                columns={'tags.mlflow.runName': 'Run Name', 'start_time': 'Start Time'}
            )
            st.write(display_runs)

            # Cho phép người dùng chọn run để xem chi tiết
            selected_run_id = st.selectbox("Select a run to view details", options=runs['run_id'].tolist())
            if selected_run_id:
                run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
                st.write("Run Details:")
                st.write(f"Run ID: {run_details['run_id']}")
                st.write(f"Run Name: {run_details.get('tags.mlflow.runName', 'Unnamed')}")
                st.write(f"Start Time: {run_details['start_time']}")

                st.write("Logged Parameters:")
                params = mlflow.get_run(selected_run_id).data.params
                st.write(params)

                st.write("Logged Artifacts (Predictions):")
                artifacts = mlflow.artifacts.list_artifacts(selected_run_id)
                for artifact in artifacts:
                    if artifact.path.endswith(".csv"):
                        artifact_file = mlflow.artifacts.download_artifacts(run_id=selected_run_id, artifact_path=artifact.path)
                        artifact_data = pd.read_csv(artifact_file)
                        st.write(f"Artifact: {artifact.path}")
                        st.write(artifact_data.head())

                mlflow_ui_link = f"http://localhost:5000/#/experiments/0/runs/{selected_run_id}"
                st.write(f"View this run in MLflow UI: [Click here]({mlflow_ui_link})")

    # Tab 3: Xóa log
    with tab3:
        st.subheader("Delete Unnecessary Logs")
        runs = get_mlflow_runs()
        if runs.empty:
            st.write("No runs available to delete.")
        else:
            st.write("Select runs to delete:")
            runs_to_delete = st.multiselect(
                "Choose runs",
                options=[f"Run ID: {run['run_id']} - {run.get('tags.mlflow.runName', 'Unnamed')}" for _, run in runs.iterrows()],
                key="delete_runs"
            )
            if st.button("Delete Selected Runs"):
                for run_str in runs_to_delete:
                    run_id = run_str.split("Run ID: ")[1].split(" - ")[0]
                    delete_mlflow_run(run_id)
                st.success("Selected runs have been deleted. Refresh the page to update the list.")

if __name__ == "__main__":
    show_demo()