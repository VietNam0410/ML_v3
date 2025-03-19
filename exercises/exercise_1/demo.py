import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub
import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
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

    # Khởi tạo DagsHub/MLflow chỉ một lần
    if 'dagshub_initialized' not in st.session_state:
        DAGSHUB_REPO = mlflow_input()
        st.session_state['dagshub_initialized'] = True
        st.session_state['mlflow_url'] = DAGSHUB_REPO
    else:
        DAGSHUB_REPO = st.session_state['mlflow_url']

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
        st.subheader("Bước 1: Chọn Mô hình Đã Huấn luyện")
        # Tải dữ liệu gốc từ titanic.csv để lấy thông tin cột và giá trị hợp lệ
        original_file = "exercises/exercise_1/data/raw/titanic.csv"
        try:
            with st.spinner("Đang tải dữ liệu gốc từ titanic.csv..."):
                original_data = load_cached_data(original_file)
        except FileNotFoundError:
            st.error("File titanic.csv không tìm thấy. Vui lòng kiểm tra đường dẫn.")
            return

        # Tải dữ liệu đã xử lý để kiểm tra các cột mà mô hình yêu cầu
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            with st.spinner("Đang tải dữ liệu đã xử lý..."):
                processed_data = load_cached_data(processed_file)
            X_full = processed_data
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
            return

        st.write(f"Chọn mô hình từ Experiment '{training_experiment_name}':")
        with st.spinner("Đang tải danh sách runs từ DagsHub..."):
            client = mlflow.tracking.MlflowClient()
            experiment_id = client.get_experiment_by_name(training_experiment_name).experiment_id
            runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])

        if runs.empty:
            st.error(f"Không tìm thấy mô hình nào trong experiment '{training_experiment_name}'. Vui lòng huấn luyện mô hình trước.")
            return

        # Lọc các run chứa mô hình
        model_runs = []
        for _, run in runs.iterrows():
            run_id = run['run_id']
            try:
                artifacts = client.list_artifacts(run_id)
                if any(artifact.path == "model" for artifact in artifacts):
                    model_runs.append(run)
            except Exception:
                continue

        if not model_runs:
            st.error(f"Không tìm thấy run nào chứa mô hình trong experiment '{training_experiment_name}'.")
            return

        run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Thời gian: {run['start_time'].strftime('%Y-%m-%d %H:%M:%S') if run['start_time'] else 'Không rõ'})" for run in model_runs]
        selected_run = st.selectbox("Chọn run chứa mô hình", options=run_options, key="model_select")
        selected_run_id = selected_run.split("ID Run: ")[1].split(" - ")[0]

        if 'model' not in st.session_state or st.session_state.get('selected_run_id') != selected_run_id:
            try:
                with st.spinner("Đang tải mô hình từ DagsHub MLflow..."):
                    model = mlflow.sklearn.load_model(f"runs:/{selected_run_id}/model")
                st.session_state['model'] = model
                st.session_state['selected_run_id'] = selected_run_id
                st.success(f"Mô hình đã được tải từ MLflow (Run ID: {selected_run_id})")
            except Exception as e:
                st.error(f"Không thể tải mô hình từ Run ID {selected_run_id}: {str(e)}")
                return

        # Bước 2: Người dùng nhập dữ liệu dựa trên các cột trong titanic.csv
        model = st.session_state['model']
        expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()

        # Loại bỏ cột không cần thiết
        input_columns = [col for col in original_data.columns if col not in ['PassengerId', 'Survived', 'Name']]

        st.subheader("Bước 2: Nhập Dữ liệu để Dự đoán (Dựa trên titanic.csv)")
        input_data = {}
        for col in input_columns:
            if col in original_data.columns:
                if col == 'Pclass':
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}' (Lớp vé: 1, 2, 3)",
                        options=[1, 2, 3],
                        key=f"input_{col}"
                    )
                elif col == 'Sex':
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}'",
                        options=['male', 'female'],
                        key=f"input_{col}"
                    )
                elif col == 'Age':
                    # Age có thể để trống (NaN), nên thêm tùy chọn "Không xác định"
                    age_options = ['Không xác định'] + list(range(0, 151))
                    selected_age = st.selectbox(
                        f"Chọn giá trị cho '{col}' (0-150, hoặc Không xác định)",
                        options=age_options,
                        key=f"input_{col}"
                    )
                    input_data[col] = None if selected_age == 'Không xác định' else selected_age
                elif col in ['SibSp', 'Parch']:
                    input_data[col] = st.number_input(
                        f"Nhập giá trị cho '{col}' (Số lượng anh chị em/vợ chồng hoặc cha mẹ/con cái, số nguyên không âm)",
                        min_value=0,
                        max_value=10,
                        value=0,
                        step=1,
                        key=f"input_{col}"
                    )
                elif col == 'Ticket':
                    unique_tickets = original_data['Ticket'].dropna().unique().tolist()
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}' (Mã vé, có thể để trống)",
                        options=[''] + unique_tickets,
                        key=f"input_{col}"
                    )
                elif col == 'Fare':
                    input_data[col] = st.number_input(
                        f"Nhập giá trị cho '{col}' (Giá vé, số thực không âm)",
                        min_value=0.0,
                        max_value=600.0,
                        value=32.2,  # Giá trị trung bình trong dữ liệu gốc
                        step=0.1,
                        key=f"input_{col}"
                    )
                elif col == 'Cabin':
                    unique_cabins = original_data['Cabin'].dropna().unique().tolist()
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}' (Số hiệu cabin, có thể để trống)",
                        options=[''] + unique_cabins,
                        key=f"input_{col}"
                    )
                elif col == 'Embarked':
                    input_data[col] = st.selectbox(
                        f"Chọn giá trị cho '{col}' (Cảng lên tàu: S, C, Q, hoặc để trống)",
                        options=['', 'S', 'C', 'Q'],
                        key=f"input_{col}"
                    )
                else:
                    st.warning(f"Cột '{col}' không được hỗ trợ để nhập. Sử dụng giá trị mặc định 0.")
                    input_data[col] = 0
            else:
                st.warning(f"Cột '{col}' không tồn tại trong dữ liệu gốc. Sử dụng giá trị mặc định 0.")
                input_data[col] = 0

        # Kiểm tra dữ liệu trước khi tạo DataFrame
        for col in input_columns:
            if col in ['Sex'] and input_data[col] == '':
                st.error(f"Vui lòng chọn giá trị hợp lệ cho cột '{col}'.")
                return

        # Tạo DataFrame từ dữ liệu người dùng nhập (chưa mã hóa)
        X_selected = pd.DataFrame([input_data])

        # Mã hóa dữ liệu trước khi dự đoán để khớp với mô hình
        for col in X_selected.columns:
            if col in ['Sex', 'Ticket', 'Cabin', 'Embarked'] and X_selected[col].dtype == 'object':
                # Lấy giá trị unique từ dữ liệu huấn luyện để mã hóa
                unique_vals = original_data[col].dropna().unique()
                value_map = {val: idx for idx, val in enumerate(unique_vals)}
                if col == 'Sex':
                    value_map = {'male': 0, 'female': 1}  # Đảm bảo Sex được mã hóa 0, 1
                X_selected[col] = X_selected[col].map(value_map).fillna(0).astype(int)

        st.write("Dữ liệu Nhập của Bạn (Chưa mã hóa):")
        st.write(pd.DataFrame([input_data]))
        st.write("Dữ liệu Đã Mã hóa để Dự đoán:")
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
                    # Đảm bảo tất cả dữ liệu là số trước khi dự đoán
                    X_selected_numeric = X_selected.astype(float)
                    predictions = model.predict(X_selected_numeric)
                    confidence = get_prediction_confidence(model, X_selected_numeric)
                    st.session_state['predictions'] = predictions
                    st.session_state['confidence'] = confidence
                    st.session_state['input_data'] = input_data

                    prediction_result = predictions[0]
                    # Hiển thị độ tin cậy của lớp được dự đoán
                    if prediction_result == 1:
                        display_confidence = confidence
                        status_message = st.success(f"Hành khách này sẽ sống với độ tin cậy {display_confidence:.2%}")
                    else:
                        display_confidence = 1 - confidence
                        status_message = st.error(f"Hành khách này sẽ chết với độ tin cậy {display_confidence:.2%}")

                    # Hiển thị bảng kết quả với độ tin cậy của lớp dự đoán
                    result_df = pd.DataFrame({"Dự đoán Sinh tồn": predictions, "Độ Tin Cậy": [display_confidence]})
                    st.write("Kết quả Dự đoán (Chi tiết):")
                    st.write(result_df)

                    st.write("Dữ liệu Nhập của Bạn (Tóm tắt, chưa mã hóa):")
                    st.write(pd.DataFrame([input_data]))
                except ValueError as e:
                    st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo dữ liệu nhập khớp với các cột và đúng kiểu dữ liệu.")
                except Exception as e:
                    st.error(f"Lỗi không xác định khi dự đoán: {str(e)}")

        # Log kết quả dự đoán
        if 'predictions' in st.session_state and st.button("Log Dự đoán vào MLflow"):
            with st.spinner("Đang log dữ liệu dự đoán vào DagsHub..."):
                try:
                    demo_experiment_id = client.get_experiment_by_name(demo_experiment_name).experiment_id
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    demo_run_name = f"{predict_name}_{timestamp}"

                    st.info("Bắt đầu một run mới trong MLflow...")
                    with mlflow.start_run(run_name=demo_run_name, experiment_id=demo_experiment_id) as run:
                        st.info("Đang log các tham số đầu vào (chưa mã hóa)...")
                        for col, value in st.session_state['input_data'].items():
                            mlflow.log_param(f"input_{col}", value)

                        st.info("Đang log kết quả dự đoán và độ tin cậy...")
                        prediction_result = st.session_state['predictions'][0]
                        logged_confidence = st.session_state['confidence'] if prediction_result == 1 else 1 - st.session_state['confidence']
                        mlflow.log_param("predicted_survival", int(prediction_result))
                        mlflow.log_param("predict_name", predict_name)
                        mlflow.log_param("confidence", float(logged_confidence))
                        mlflow.log_param("demo_name", "Titanic_Survival_Demo")
                        mlflow.log_param("demo_run_id", run.info.run_id)
                        mlflow.log_param("model_run_id", selected_run_id)

                        st.info("Hoàn tất log các tham số...")
                        st.write("Thông tin Đã Log:")
                        log_info = {
                            "Tên Run": demo_run_name,
                            "ID Run": run.info.run_id,
                            "Tên Dự đoán": predict_name,
                            "Dữ liệu Nhập (chưa mã hóa)": st.session_state['input_data'],
                            "Dự đoán Sinh tồn": int(prediction_result),
                            "Độ Tin Cậy": float(logged_confidence),
                            "Tên Demo": "Titanic_Survival_Demo",
                            "Mô hình Nguồn": selected_run_id
                        }
                        st.write(log_info)

                        st.session_state['last_run_id'] = run.info.run_id
                        mlflow_uri = st.session_state['mlflow_url']
                        st.success(f"Dự đoán đã được log thành công!\n- Experiment: '{demo_experiment_name}'\n- Tên Run: '{demo_run_name}'\n- ID Run: {run.info.run_id}")
                        st.markdown(f"Xem chi tiết tại: [DagsHub MLflow Tracking]({mlflow_uri})")
                except mlflow.exceptions.MlflowException as e:
                    st.error(f"Lỗi khi log vào MLflow: {str(e)}")
                except AttributeError as e:
                    st.error(f"Lỗi: Experiment '{demo_experiment_name}' không tồn tại hoặc không thể truy cập: {str(e)}")

    with tab2:
        st.subheader("Xem Kết quả Đã Log")
        with st.spinner("Đang tải danh sách các experiment từ DagsHub..."):
            experiments = get_mlflow_experiments()
        if not experiments:
            st.error("Không tìm thấy experiment nào trong DagsHub MLflow.")
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

                        mlflow_uri = st.session_state['mlflow_url']
                        st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({mlflow_uri})")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        with st.spinner("Đang tải danh sách các experiment từ DagsHub..."):
            experiments = get_mlflow_experiments()
        if not experiments:
            st.error("Không tìm thấy experiment nào trong DagsHub MLflow.")
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
