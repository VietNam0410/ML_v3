import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from common.utils import load_data
import os
import dagshub

# Phần khởi tạo kết nối với DagsHub được comment để không truy cập ngay lập tức
# with st.spinner("Đang kết nối với DagsHub..."):
#     dagshub.init(repo_owner='VietNam0410', repo_name='vn0410', mlflow=True)
#     # Cấu hình MLflow tracking URI
#     mlflow.set_tracking_uri(f"https://dagshub.com/VietNam0410/vn0410.mlflow")
# st.success("Đã kết nối với DagsHub thành công!")

# Hàm tải dữ liệu với cache
@st.cache_data
def load_cached_data(file_path):
    """Tải dữ liệu từ file CSV và lưu vào bộ nhớ đệm."""
    return load_data(file_path)

# Hàm lấy danh sách experiment với cache (comment vì không cần khi không log)
# @st.cache_data
# def get_mlflow_experiments_cached():
#     """Lấy danh sách các experiment từ MLflow và lưu vào bộ nhớ đệm."""
#     try:
#         client = mlflow.tracking.MlflowClient()
#         experiments = client.search_experiments()
#         return {exp.name: exp.experiment_id for exp in experiments}
#     except mlflow.exceptions.MlflowException as e:
#         st.error(f"Không thể lấy danh sách các experiment từ MLflow: {str(e)}")
#         return {}

# Hàm lấy danh sách runs với cache (comment vì không cần khi không log)
# @st.cache_data
# def get_mlflow_runs_cached():
#     """Lấy danh sách các run từ MLflow và lưu vào bộ nhớ đệm."""
#     try:
#         runs = mlflow.search_runs()
#         return runs
#     except mlflow.exceptions.MlflowException as e:
#         st.error(f"Không thể lấy danh sách các run từ MLflow: {str(e)}")
#         return pd.DataFrame()

def delete_mlflow_run(run_id):
    try:
        with st.spinner(f"Đang xóa Run {run_id}..."):
            mlflow.delete_run(run_id)
        st.success(f"Đã xóa run có ID: {run_id}")
    except mlflow.exceptions.MlflowException as e:
        st.error(f"Không thể xóa run {run_id}: {str(e)}")

def show_demo():
    st.header("Dự đoán Sinh tồn Titanic")

    # Đóng bất kỳ run nào đang hoạt động để tránh xung đột khi bắt đầu
    if mlflow.active_run():
        mlflow.end_run()
        st.info("Đã đóng run MLflow đang hoạt động trước đó.")

    # Comment phần chọn experiment vì không cần khi không log
    # experiments = get_mlflow_experiments_cached()  # Sử dụng hàm có cache
    # experiment_options = list(experiments.keys()) if experiments else ["Titanic_Demo"]
    # experiment_name = st.selectbox(
    #     "Chọn hoặc nhập tên Experiment cho Demo",
    #     options=experiment_options,
    #     index=experiment_options.index("Titanic_Demo") if "Titanic_Demo" in experiment_options else 0,
    #     help="Chọn một experiment hiện có hoặc nhập tên mới."
    # )
    # if experiment_name:
    #     with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
    #         mlflow.set_experiment(experiment_name)

    tab1, tab2, tab3 = st.tabs(["Dự đoán", "Xem Kết quả Đã Log", "Xóa Log"])

    with tab1:
        st.subheader("Bước 1: Tùy chỉnh Dữ liệu Nhập cho Dự đoán")
        processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
        try:
            with st.spinner("Đang tải dữ liệu đã xử lý..."):
                data = load_cached_data(processed_file)  # Sử dụng hàm có cache
            X_full = data.drop(columns=['Survived', 'Name'] if 'Name' in data.columns else ['Survived'])
        except FileNotFoundError:
            st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng tiền xử lý dữ liệu trước.")
        else:
            # Comment phần tải danh sách runs từ DagsHub vì không cần khi không log
            # with st.spinner("Đang tải danh sách Runs từ DagsHub..."):
            #     runs = get_mlflow_runs_cached()  # Sử dụng hàm có cache
            # if runs.empty:
            #     st.error("Không tìm thấy mô hình đã huấn luyện trong MLflow. Vui lòng huấn luyện mô hình trước.")
            # else:
            #     model_options = {f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']})": run['run_id'] 
            #                     for _, run in runs.iterrows()}
            #     selected_model_name = st.selectbox("Chọn một mô hình đã huấn luyện", options=list(model_options.keys()))
            #     selected_run_id = model_options[selected_model_name]

            # Thay vì tải từ MLflow, yêu cầu người dùng nhập đường dẫn mô hình cục bộ (nếu có)
            st.info("Vì logging vào DagsHub đã bị tắt, hãy cung cấp đường dẫn đến mô hình cục bộ nếu muốn dự đoán.")
            model_path = st.text_input("Nhập đường dẫn đến file mô hình cục bộ (ví dụ: 'path/to/model.pkl')", value="")
            if not model_path:
                st.warning("Vui lòng cung cấp đường dẫn đến mô hình để tiếp tục.")
            else:
                try:
                    with st.spinner("Đang tải mô hình cục bộ..."):
                        model = mlflow.sklearn.load_model(model_path)  # Tải mô hình từ đường dẫn cục bộ
                    expected_columns = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_full.columns.tolist()
                except Exception as e:
                    st.error(f"Không thể tải mô hình từ đường dẫn: {model_path}. Lỗi: {str(e)}")
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
                        with st.spinner("Đang thực hiện dự đoán..."):
                            try:
                                predictions = model.predict(X_selected)
                                result_df = pd.DataFrame({"Dự đoán Sinh tồn": predictions})
                                st.write("Kết quả Dự đoán:")
                                st.write(result_df)

                                st.write("Dữ liệu Nhập của Bạn (Tóm tắt):")
                                st.write(pd.DataFrame([input_data]))

                                # Comment phần logging dự đoán
                                # run_name = st.text_input("Nhập tên cho run dự đoán này", value="Run_Dự_đoán")
                                # if st.button("Log Dự đoán vào MLflow"):
                                #     with st.spinner("Đang log dữ liệu dự đoán vào DagsHub..."):
                                #         with mlflow.start_run(run_name=run_name) as run:
                                #             for col, value in input_data.items():
                                #                 mlflow.log_param(f"input_{col}", value)
                                #             mlflow.log_param("model_run_id", selected_run_id)
                                #             mlflow.log_param("predicted_survival", predictions[0])
                                #             mlflow.log_param("run_name", run_name)
                                #             mlflow.log_param("run_id", run.info.run_id)

                                #             st.write("Thông tin Đã Log:")
                                #             log_info = {
                                #                 "Tên Run": run_name,
                                #                 "ID Run": run.info.run_id,
                                #                 "Dữ liệu Nhập": input_data,
                                #                 "ID Run Mô hình": selected_run_id,
                                #                 "Dự đoán Sinh tồn": predictions[0]
                                #             }
                                #             st.write(log_info)

                                #             run_id = run.info.run_id
                                #             dagshub_link = f"https://dagshub.com/VietNam0410/vn0410/experiments/#/experiment/{experiment_name}/{run_id}"
                                #             st.success(f"Dự đoán đã được log thành công!\n- Experiment: '{experiment_name}'\n- Tên Run: '{run_name}'\n- ID Run: {run_id}")
                                #             st.markdown(f"Xem chi tiết tại: [DagsHub Experiment]({dagshub_link})")

                                #             st.write("Bạn muốn làm gì tiếp theo?")
                                #             col1, col2 = st.columns(2)
                                #             with col1:
                                #                 if st.button("Xem run này trong 'Xem Kết quả Đã Log'"):
                                #                     st.session_state['selected_run_id'] = run_id
                                #                     st.session_state['active_tab'] = 1
                                #                     st.info("Vui lòng chuyển sang tab 'Xem Kết quả Đã Log'.")
                                #             with col2:
                                #                 if st.button("Xóa run này trong 'Xóa Log'"):
                                #                     st.session_state['selected_run_id'] = run_id
                                #                     st.session_state['active_tab'] = 2
                                #                     st.info("Vui lòng chuyển sang tab 'Xóa Log'.")

                            except ValueError as e:
                                st.error(f"Dự đoán thất bại: {str(e)}. Đảm bảo dữ liệu nhập khớp với các cột: {expected_columns}")

    with tab2:
        st.subheader("Xem Kết quả Đã Log")
        # Comment phần tải runs từ DagsHub
        # with st.spinner("Đang tải danh sách runs từ DagsHub..."):
        #     runs = get_mlflow_runs_cached()  # Sử dụng hàm có cache
        # if runs.empty:
        #     st.write("Chưa có run dự đoán nào được log.")
        # else:
        #     st.write("Danh sách các run đã log:")
        #     display_runs = runs[['run_id', 'tags.mlflow.runName', 'start_time', 'experiment_id']].rename(
        #         columns={'tags.mlflow.runName': 'Tên Run', 'start_time': 'Thời gian Bắt đầu', 'experiment_id': 'ID Experiment'}
        #     )
        #     st.write(display_runs)

        #     default_run = st.session_state.get('selected_run_id', runs['run_id'].iloc[0])
        #     selected_run_id = st.selectbox(
        #         "Chọn một run để xem chi tiết",
        #         options=runs['run_id'].tolist(),
        #         index=runs['run_id'].tolist().index(default_run) if default_run in runs['run_id'].tolist() else 0
        #     )
        #     if selected_run_id:
        #         with st.spinner("Đang tải chi tiết run từ DagsHub..."):
        #             run_details = runs[runs['run_id'] == selected_run_id].iloc[0]
        #             st.write("Chi tiết Run:")
        #             st.write(f"ID Run: {run_details['run_id']}")
        #             st.write(f"Tên Run: {run_details.get('tags.mlflow.runName', 'Không tên')}")
        #             st.write(f"ID Experiment: {run_details['experiment_id']}")
        #             st.write(f"Thời gian Bắt đầu: {run_details['start_time']}")

        #             st.write("Thông số Đã Log:")
        #             params = mlflow.get_run(selected_run_id).data.params
        #             st.write(params)

        #             dagshub_link = f"https://dagshub.com/VietNam0410/vn0410/experiments/#/experiment/{experiment_name}/{selected_run_id}"
        #             st.markdown(f"Xem run này trong DagsHub UI: [Nhấn vào đây]({dagshub_link})")
        st.info("Chức năng xem kết quả log tạm thời bị tắt vì logging vào DagsHub đã bị vô hiệu hóa.")

    with tab3:
        st.subheader("Xóa Log Không Cần Thiết")
        # Comment phần tải runs từ DagsHub
        # with st.spinner("Đang tải danh sách runs từ DagsHub..."):
        #     runs = get_mlflow_runs_cached()  # Sử dụng hàm có cache
        # if runs.empty:
        #     st.write("Không có run nào để xóa.")
        # else:
        #     st.write("Chọn các run để xóa:")
        #     run_options = [f"ID Run: {run['run_id']} - {run.get('tags.mlflow.runName', 'Không tên')} (Exp: {run['experiment_id']})" 
        #                   for _, run in runs.iterrows()]
        #     default_delete = [f"ID Run: {st.session_state['selected_run_id']} - {runs[runs['run_id'] == st.session_state['selected_run_id']]['tags.mlflow.runName'].iloc[0]} (Exp: {runs[runs['run_id'] == st.session_state['selected_run_id']]['experiment_id'].iloc[0]})" 
        #                      if 'selected_run_id' in st.session_state and st.session_state['selected_run_id'] in runs['run_id'].tolist() else None]
        #     runs_to_delete = st.multiselect(
        #         "Chọn các run",
        #         options=run_options,
        #         default=[d for d in default_delete if d],
        #         key="delete_runs"
        #     )
        #     if st.button("Xóa Các Run Đã Chọn"):
        #         for run_str in runs_to_delete:
        #             run_id = run_str.split("ID Run: ")[1].split(" - ")[0]
        #             delete_mlflow_run(run_id)
        #         st.success("Các run đã chọn đã được xóa. Làm mới trang để cập nhật danh sách.")
        #         # Xóa cache sau khi xóa run để cập nhật dữ liệu
        #         get_mlflow_runs_cached.clear()
        st.info("Chức năng xóa log tạm thời bị tắt vì logging vào DagsHub đã bị vô hiệu hóa.")

if __name__ == "__main__":
    show_demo()