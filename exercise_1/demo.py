import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub
import datetime
from sklearn.ensemble import RandomForestClassifier  # Thêm để kiểm tra nếu cần

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

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

def get_prediction_confidence(model, X_selected):
    """Tính độ tin cậy của dự đoán (probability) từ mô hình."""
    try:
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_selected)
            confidence = probabilities[0][1]  # Xác suất dự đoán lớp 1 (survived)
            return confidence
        else:
            st.warning("Mô hình không hỗ trợ predict_proba. Đặt độ tin cậy mặc định là 0.5.")
            return 0.5
    except Exception as e:
        st.error(f"Không thể tính độ tin cậy dự đoán: {str(e)}")
        return 0.5

def show_demo():
    st.header("Dự đoán Sinh tồn Titanic")

    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    DAGSHUB_REPO = mlflow_input()

    training_experiment_name = "Titanic_Training"
    with st.spinner("Đang thiết lập Experiment huấn luyện trên DagsHub..."):
        try:
            client = mlflow.tracking.MlflowClient()
            experiment = client.get_experiment_by_name(training_experiment_name)
            if not experiment:
                client.create_experiment(training_experiment_name)
                mlflow.set_experiment(training_experiment_name)
                st.success(f"Đã tạo Experiment mới '{training_experiment_name}' thành công!")
            elif experiment and experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                mlflow.set_experiment(training_experiment_name)
                st.success(f"Đã khôi phục Experiment '{training_experiment_name}' thành công!")
            else:
                mlflow.set_experiment(training_experiment_name)
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment huấn luyện: {str(e)}")
            return

    demo_experiment_name = "Titanic_Demo"
    with st.spinner("Đang thiết lập Experiment demo trên DagsHub..."):
        try:
            experiment = client.get_experiment_by_name(demo_experiment_name)
            if not experiment:
                client.create_experiment(demo_experiment_name)
                st.success(f"Đã tạo Experiment mới '{demo_experiment_name}' thành công!")
            elif experiment and experiment.lifecycle_stage == "deleted":
                client.restore_experiment(experiment.experiment_id)
                st.success(f"Đã khôi phục Experiment '{demo_experiment_name}' thành công!")
            else:
                st.success(f"Đã thiết lập Experiment '{demo_experiment_name}' thành công!")
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment demo: {str(e)}")
            return

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán")
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            with st.spinner("Đang tải dữ liệu đã xử lý..."):
                data = load_cached_data(processed_file)
            X_full = data
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
            return

        st.write(f"Chọn mô hình đã huấn luyện từ Experiment '{training_experiment_name}':")
        with st.spinner("Đang tải danh sách runs từ DagsHub..."):
            training_experiment_id = client.get_experiment_by_name(training_experiment_name).experiment_id
            # Chỉ lấy các run chứa mô hình (giả định có tag hoặc artifact mô hình)
            runs = mlflow.search_runs(experiment_ids=[training_experiment_id])
            # Lọc các run chứa mô hình (giả định mô hình được log với mlflow.sklearn.log_model)
            model_runs = runs[runs['tags.mlflow.log-model.history'].str.contains('sklearn', na=False, case=True) | 
                             runs['artifact_uri'].str.contains('model', na=False, case=True)].copy()
            if model_runs.empty:
                st.error(f"Không tìm thấy mô hình nào trong experiment '{training_experiment_name}'. Vui lòng huấn luyện mô hình trong 'train.py' trước.")
                return

        run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')}" for _, run in model_runs.iterrows()]
        selected_run = st.selectbox("Chọn run chứa mô hình", options=run_options)
        selected_run_id = selected_run.split("ID Run: ")[1].split(" - ")[0]

        if 'model' not in st.session_state or st.session_state.get('selected_run_id') != selected_run_id:
            try:
                with st.spinner("Đang tải mô hình từ DagsHub MLflow..."):
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                st.session_state['model'] = model
                st.session_state['selected_run_id'] = selected_run_id
                st.write(f"Mô hình đã được tải từ MLflow (Run ID: {selected_run_id})")
            except Exception as e:
                st.error(f"Không thể tải mô hình từ Run ID {selected_run_id}: {str(e)}")
                return

        model = st.session_state['model']
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

        predict_name = st.text_input(
            "Đặt tên cho dự đoán này (tối đa 50 ký tự, để trống để tự động tạo)",
            value=f"Prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}",
            max_chars=50,
            key="predict_name"
        )
        if not predict_name.strip():
            predict_name = f"Prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Dự đoán
        if st.button("Thực hiện Dự đoán"):
            with st.spinner("Đang thực hiện dự đoán..."):
                try:
                    predictions = model.predict(X_selected)
                    confidence = get_prediction_confidence(model, X_selected)
                    st.session_state['predictions'] = predictions
                    st.session_state['confidence'] = confidence
                    st.session_state['input_data'] = input_data

                    result_df = pd.DataFrame({"Dự đoán Sinh tồn": predictions, "Độ Tin Cậy": [confidence]})
                    st.write("Kết quả Dự đoán:")
                    st.write(result_df)

                    st.write("Dữ liệu Nhập của Bạn (Tóm tắt):")
                    st.write(pd.DataFrame([input_data]))
                except ValueError as e:
                    st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo dữ liệu nhập khớp với các cột: {expected_columns}")

        # Log kết quả dự đoán
        if 'predictions' in st.session_state and st.button("Log Dự đoán vào MLflow"):
            with st.spinner("Đang log dữ liệu dự đoán vào DagsHub..."):
                try:
                    demo_experiment_id = client.get_experiment_by_name(demo_experiment_name).experiment_id
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    demo_run_name = f"{predict_name}_{timestamp}"

                    st.info("Bắt đầu một run mới trong MLflow...")
                    with mlflow.start_run(run_name=demo_run_name, experiment_id=demo_experiment_id) as run:
                        st.info("Đang log các tham số đầu vào...")
                        for col, value in st.session_state['input_data'].items():
                            mlflow.log_param(f"input_{col}", value)

                        st.info("Đang log kết quả dự đoán và độ tin cậy...")
                        mlflow.log_param("predicted_survival", int(st.session_state['predictions'][0]))
                        mlflow.log_param("predict_name", predict_name)
                        mlflow.log_param("confidence", float(st.session_state['confidence']))
                        mlflow.log_param("demo_name", "Titanic_Survival_Demo")
                        mlflow.log_param("demo_run_id", run.info.run_id)
                        mlflow.log_param("model_run_id", selected_run_id)

                        st.info("Hoàn tất log các tham số. Đang hiển thị thông tin...")
                        st.write("Thông tin Đã Log:")
                        log_info = {
                            "Tên Run": demo_run_name,
                            "ID Run": run.info.run_id,
                            "Tên Dự đoán": predict_name,
                            "Dữ liệu Nhập": st.session_state['input_data'],
                            "Dự đoán Sinh tồn": int(st.session_state['predictions'][0]),
                            "Độ Tin Cậy": float(st.session_state['confidence']),
                            "Tên Demo": "Titanic_Survival_Demo",
                            "Mô hình Nguồn": selected_run_id
                        }
                        st.write(log_info)

                        st.session_state['last_run_id'] = run.info.run_id
                        mlflow_uri = DAGSHUB_REPO
                        st.success(f"Dự đoán đã được log thành công!\n- Experiment: '{demo_experiment_name}'\n- Tên Run: '{demo_run_name}'\n- ID Run: {run.info.run_id}")
                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")
                except mlflow.exceptions.MlflowException as e:
                    st.error(f"Lỗi khi log vào MLflow: {str(e)}. Vui lòng kiểm tra kết nối DagsHub hoặc quyền truy cập.")
                except AttributeError as e:
                    st.error(f"Lỗi: Experiment '{demo_experiment_name}' không tồn tại hoặc không thể truy cập: {str(e)}")


    with tab2:
        st.subheader("Xem Kết quả Đã Log")
        with st.spinner("Đang tải danh sách các experiment từ DagsHub..."):
            experiments = get_mlflow_experiments()
        if not experiments:
            st.error("Không tìm thấy experiment nào trong DagsHub MLflow. Vui lòng kiểm tra kết nối.")
            return

        experiment_options = list(experiments.keys())
        selected_experiment = st.selectbox(
            "Chọn Experiment để xem runs",
            options=experiment_options,
            help="Chọn một experiment để xem các run đã log."
        )

        with st.spinner(f"Đang tải danh sách runs từ Experiment '{selected_experiment}'..."):
            runs = mlflow.search_runs(experiment_ids=[experiments[selected_experiment]])
        if runs.empty:
            st.write(f"Chưa có run dự đoán nào được log trong experiment '{selected_experiment}'.")
        else:
            st.write(f"Danh sách các run đã log trong Experiment '{selected_experiment}':")
            display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
                columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
            )
            st.write(display_runs)

            default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0] if not runs.empty else None)
            if default_run and default_run in runs['run_id'].tolist():
                default_index = runs['run_id'].tolist().index(default_run)
            else:
                default_index = 0 if not runs.empty else None

            if not runs.empty:
                selected_run_id = st.selectbox(
                    "Chọn một run để xem chi tiết",
                    options=runs['run_id'].tolist(),
                    index=default_index,
                    format_func=lambda x: f"ID Run: {x} - {runs[runs['run_id'] == x]['tags.mlflow.runName'].iloc[0] if not runs[runs['run_id'] == x]['tags.mlflow.runName'].empty else 'Không tên'}"
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

                        mlflow_uri = DAGSHUB_REPO
                        st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({mlflow_uri})")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        with st.spinner("Đang tải danh sách các experiment từ DagsHub..."):
            experiments = get_mlflow_experiments()
        if not experiments:
            st.error("Không tìm thấy experiment nào trong DagsHub MLflow. Vui lòng kiểm tra kết nối.")
            return

        experiment_options = list(experiments.keys())
        selected_experiment = st.selectbox(
            "Chọn Experiment để xóa runs",
            options=experiment_options,
            help="Chọn một experiment để xóa các run không cần thiết."
        )

        with st.spinner(f"Đang tải danh sách runs từ Experiment '{selected_experiment}'..."):
            runs = mlflow.search_runs(experiment_ids=[experiments[selected_experiment]])
        if runs.empty:
            st.write(f"Không có run nào để xóa trong experiment '{selected_experiment}'.")
        else:
            st.write(f"Chọn các run để xóa trong Experiment '{selected_experiment}':")
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