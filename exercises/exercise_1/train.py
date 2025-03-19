import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from common.utils import load_data
import os
import dagshub
import datetime
import time  # Thêm để mô phỏng tiến độ

# Hàm khởi tạo MLflow
def mlflow_input():
    DAGSHUB_MLFLOW_URI = "https://dagshub.com/VietNam0410/ML_v3.mlflow"
    mlflow.set_tracking_uri(DAGSHUB_MLFLOW_URI)
    
    # Thiết lập thông tin xác thực
    os.environ["MLFLOW_TRACKING_USERNAME"] = "VietNam0410"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "c9db6bdcca1dfed76d2af2cdb15a9277e6732d6b"
    dagshub.auth.add_app_token(token=os.environ["MLFLOW_TRACKING_PASSWORD"])
    
    # Khởi tạo repository
    dagshub.init("vn0410", "VietNam0410", mlflow=True)
    return DAGSHUB_MLFLOW_URI

def train_model():
    st.header("Train Titanic Survival Model 🧑‍🚀")

    # Khởi tạo MLflow
    try:
        mlflow_uri = mlflow_input()
        st.session_state['mlflow_url'] = mlflow_uri
    except Exception as e:
        st.error(f"Lỗi khi thiết lập MLflow: {str(e)}")
        return

    # Thiết lập experiment
    experiment_name = st.text_input("Enter Experiment Name for Training", value="Titanic_Training")
    with st.spinner("Đang thiết lập Experiment trên DagsHub..."):
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if not experiment:
                mlflow.create_experiment(experiment_name)
            elif experiment.lifecycle_stage == "deleted":
                st.warning(f"Experiment '{experiment_name}' đã bị xóa trước đó.")
                new_experiment_name = st.text_input(
                    "Nhập tên Experiment mới", 
                    value=f"{experiment_name}_Restored_{datetime.datetime.now().strftime('%Y%m%d')}"
                )
                if not new_experiment_name:
                    st.error("Vui lòng nhập tên experiment mới để tiếp tục.")
                    return
                experiment_name = new_experiment_name
                mlflow.create_experiment(experiment_name)
            mlflow.set_experiment(experiment_name)
            st.success(f"Đã thiết lập Experiment '{experiment_name}' thành công!")
        except Exception as e:
            st.error(f"Lỗi khi thiết lập experiment: {str(e)}")
            return

    # Tải dữ liệu đã xử lý
    processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
    try:
        with st.spinner("Đang tải dữ liệu đã xử lý..."):
            data = load_data(processed_file)
        st.subheader("Dữ liệu đã tiền xử lý 📝")
        st.write(data.head())
        st.write("Các cột hiện có:", data.columns.tolist())
    except FileNotFoundError:
        st.error("File 'titanic_processed.csv' không tìm thấy. Vui lòng hoàn tất tiền xử lý trước.")
        return

    # Kiểm tra cột 'Survived'
    if 'Survived' not in data.columns:
        st.error("Cột 'Survived' không tồn tại. Vui lòng kiểm tra file CSV hoặc bước tiền xử lý.")
        return

    X = data.drop(columns=['Survived'])
    y = data['Survived']

    # Chia dữ liệu
    st.subheader("Chia dữ liệu 🔀")
    test_size = st.slider("Kích thước tập kiểm tra (%)", 10, 50, 20, 5) / 100
    valid_size_relative = st.slider(
        "Kích thước tập validation (% dữ liệu còn lại)", 0, 50, 20, 5
    ) / 100
    remaining_size = 1 - test_size
    valid_size = remaining_size * valid_size_relative
    train_size = remaining_size - valid_size

    if st.button("Chia dữ liệu"):
        # Tạo thanh tiến trình
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        with st.spinner("Đang chia dữ liệu..."):
            # Bước 1: Chia tập train/test (50%)
            progress_text.text("Đang chia tập train/test... (Bước 1/2)")
            X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            time.sleep(0.5)  # Mô phỏng thời gian xử lý
            progress_bar.progress(50)

            # Bước 2: Chia tập train/valid (100%)
            progress_text.text("Đang chia tập train/valid... (Bước 2/2)")
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_temp, y_temp, test_size=valid_size / remaining_size, random_state=42
            )
            time.sleep(0.5)  # Mô phỏng thời gian xử lý
            progress_bar.progress(100)

            st.write(f"Tập huấn luyện: {len(X_train)} mẫu ({train_size*100:.1f}%)")
            st.write(f"Tập validation: {len(X_valid)} mẫu ({valid_size*100:.1f}%)")
            st.write(f"Tập kiểm tra: {len(X_test)} mẫu ({test_size*100:.1f}%)")
            st.write("X_train:", X_train.head())
            st.write("X_valid:", X_valid.head())
            st.write("X_test:", X_test.head())

            # Log chia dữ liệu
            with mlflow.start_run(run_name=f"Data_Split_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_param("test_size", test_size)
                mlflow.log_param("valid_size", valid_size)
                mlflow.log_param("train_size", train_size)
                mlflow.log_metric("train_samples", len(X_train))
                mlflow.log_metric("valid_samples", len(X_valid))
                mlflow.log_metric("test_samples", len(X_test))
                run_id = mlflow.active_run().info.run_id
                experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                run_url = f"{mlflow_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                st.success("Dữ liệu đã được chia và log vào MLflow ✅")
                st.markdown(f"Xem chi tiết: [{run_url}]({run_url})")

            st.session_state['X_train'] = X_train
            st.session_state['X_valid'] = X_valid
            st.session_state['X_test'] = X_test
            st.session_state['y_train'] = y_train
            st.session_state['y_valid'] = y_valid
            st.session_state['y_test'] = y_test

    # Cross Validation
    st.subheader("Cross Validation (Tùy chọn) 🔄")
    use_cv = st.checkbox("Sử dụng Cross Validation")
    if use_cv:
        if 'X_train' not in st.session_state:
            st.warning("Vui lòng chia dữ liệu trước khi dùng Cross Validation.")
            return

        k_folds = st.slider("Số lượng folds (k)", 2, 10, 5)
        if 'valid_sizes' not in st.session_state:
            st.session_state['valid_sizes'] = [int(100 / k_folds)] * k_folds

        st.write("Phân bổ tỷ lệ tập valid cho từng fold (tổng = 100%)")
        valid_sizes = []
        total_valid = 0
        for i in range(k_folds):
            fold_num = i + 1
            valid_size = st.slider(
                f"Kích thước tập valid Fold {fold_num} (%)",
                0, 100 - total_valid, st.session_state['valid_sizes'][i],
                key=f"valid_size_fold_{fold_num}"
            )
            valid_sizes.append(valid_size)
            total_valid += valid_size
            if total_valid > 100:
                st.error("Tổng tỷ lệ valid vượt quá 100%. Vui lòng điều chỉnh.")
                return

        if st.button("Tạo Cross Validation Folds"):
            if total_valid != 100:
                st.error("Tổng tỷ lệ valid phải bằng 100%. Vui lòng điều chỉnh.")
                return

            # Tạo thanh tiến trình
            progress_text = st.empty()
            progress_bar = st.progress(0)

            X_train = st.session_state['X_train']
            y_train = st.session_state['y_train']
            with st.spinner("Đang tạo Cross Validation Folds..."):
                kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
                fold_configs = {}
                fold_indices = list(kf.split(X_train))

                # Chia tiến độ thành các bước dựa trên số fold
                progress_per_fold = 100 // k_folds

                for fold, (train_idx, valid_idx) in enumerate(fold_indices):
                    fold_num = fold + 1
                    progress_text.text(f"Đang tạo Fold {fold_num}/{k_folds}... (Bước {fold_num}/{k_folds})")
                    X_remaining = X_train.iloc[train_idx.tolist() + valid_idx.tolist()]
                    y_remaining = y_train.iloc[train_idx.tolist() + valid_idx.tolist()]
                    valid_size_fold = valid_sizes[fold] / 100

                    X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
                        X_remaining, y_remaining, test_size=valid_size_fold, random_state=42
                    )

                    fold_configs[fold_num] = {
                        "X_train": X_train_fold,
                        "y_train": y_train_fold,
                        "X_valid": X_valid_fold,
                        "y_valid": y_valid_fold
                    }

                    # Cập nhật tiến độ
                    progress_bar.progress(min((fold + 1) * progress_per_fold, 100))
                    time.sleep(0.5)  # Mô phỏng thời gian xử lý

                st.session_state['fold_configs'] = fold_configs
                st.session_state['valid_sizes'] = valid_sizes

                fold_summary = [{"Fold": k, "Train Size": len(v["X_train"]), "Valid Size": len(v["X_valid"])} 
                              for k, v in fold_configs.items()]
                st.write("Tổng quan các Fold:", pd.DataFrame(fold_summary))

                # Log CV
                with mlflow.start_run(run_name=f"CV_{k_folds}_Folds_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                    mlflow.log_param("k_folds", k_folds)
                    for i, size in enumerate(valid_sizes):
                        mlflow.log_param(f"fold_{i+1}_valid_size", size)
                    for fold in fold_summary:
                        mlflow.log_metric(f"fold_{fold['Fold']}_train_size", fold["Train Size"])
                        mlflow.log_metric(f"fold_{fold['Fold']}_valid_size", fold["Valid Size"])
                    run_id = mlflow.active_run().info.run_id
                    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                    run_url = f"{mlflow_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                    st.success(f"Đã tạo {k_folds}-fold CV và log vào MLflow ✅")
                    st.markdown(f"Xem chi tiết: [{run_url}]({run_url})")

    # Huấn luyện mô hình
    st.subheader("Huấn luyện mô hình 🎯")
    if 'X_train' not in st.session_state:
        st.warning("Vui lòng chia dữ liệu trước khi huấn luyện.")
        return

    st.write("### Tham số mô hình")
    model_choice = st.selectbox("Chọn mô hình", ["Random Forest", "Logistic Regression", "Polynomial Regression"])
    
    if model_choice == "Random Forest":
        st.write("- **n_estimators**: Số cây (tăng cải thiện hiệu suất, tốn tài nguyên)")
        st.write("- **max_depth**: Độ sâu tối đa (giới hạn tránh overfitting)")
        n_estimators = st.slider("n_estimators", 10, 200, 100, 10)
        max_depth = st.slider("max_depth", 1, 20, 10, 1)
        model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
    elif model_choice == "Logistic Regression":
        st.write("- **max_iter**: Số vòng lặp tối đa (tăng nếu không hội tụ)")
        max_iter = st.slider("max_iter", 100, 1000, 100, 100)
        model_params = {"max_iter": max_iter, "random_state": 42}
    elif model_choice == "Polynomial Regression":
        st.write("- **degree**: Bậc đa thức (tăng để phức tạp hơn)")
        degree = st.slider("degree", 1, 5, 2, 1)
        model_params = {"degree": degree, "random_state": 42}

    if st.button("Huấn luyện mô hình"):
        # Tạo thanh tiến trình
        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Đang huấn luyện mô hình..."):
            if use_cv and 'fold_configs' in st.session_state:
                X_train = pd.concat([config['X_train'] for config in st.session_state['fold_configs'].values()])
                y_train = pd.concat([config['y_train'] for config in st.session_state['fold_configs'].values()])
                X_valid = pd.concat([config['X_valid'] for config in st.session_state['fold_configs'].values()])
                y_valid = pd.concat([config['y_valid'] for config in st.session_state['fold_configs'].values()])
                train_source = "All CV folds"
            else:
                X_train = st.session_state['X_train']
                y_train = st.session_state['y_train']
                X_valid = st.session_state['X_valid']
                y_valid = st.session_state['y_valid']
                train_source = "Initial split"

            # Bước 1: Khởi tạo mô hình (20%)
            progress_text.text("Đang khởi tạo mô hình... (Bước 1/5)")
            if model_choice == "Random Forest":
                model = RandomForestClassifier(**model_params)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(**model_params)
            elif model_choice == "Polynomial Regression":
                model = Pipeline([
                    ("poly", PolynomialFeatures(degree=model_params["degree"])),
                    ("logistic", LogisticRegression(random_state=42))
                ])
            time.sleep(0.5)  # Mô phỏng thời gian xử lý
            progress_bar.progress(20)

            # Bước 2: Huấn luyện mô hình (50%)
            progress_text.text("Đang huấn luyện mô hình... (Bước 2/5)")
            model.fit(X_train, y_train)
            time.sleep(0.5)  # Mô phỏng thời gian xử lý
            progress_bar.progress(50)

            # Bước 3: Đánh giá mô hình (70%)
            progress_text.text("Đang đánh giá mô hình... (Bước 3/5)")
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)
            time.sleep(0.5)  # Mô phỏng thời gian xử lý
            progress_bar.progress(70)

            # Bước 4: Log parameters và metrics (90%)
            progress_text.text("Đang log parameters và metrics lên MLflow... (Bước 4/5)")
            with mlflow.start_run(run_name=f"{model_choice}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_params(model_params)
                mlflow.log_param("train_source", train_source)
                mlflow.log_metric("train_accuracy", train_score)
                mlflow.log_metric("valid_accuracy", valid_score)
                time.sleep(0.5)  # Mô phỏng thời gian xử lý
                progress_bar.progress(90)

                # Bước 5: Log mô hình (100%)
                progress_text.text("Đang log mô hình lên MLflow... (Bước 5/5)")
                mlflow.sklearn.log_model(model, "model")
                run_id = mlflow.active_run().info.run_id
                experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
                run_url = f"{mlflow_uri}/#/experiments/{experiment_id}/runs/{run_id}"
                time.sleep(0.5)  # Mô phỏng thời gian xử lý
                progress_bar.progress(100)

                st.write(f"Mô hình: {model_choice}")
                st.write(f"Nguồn dữ liệu: {train_source}")
                st.write(f"Tham số: {model_params}")
                st.write(f"Độ chính xác train: {train_score:.4f}")
                st.write(f"Độ chính xác valid: {valid_score:.4f}")
                st.success(f"Đã huấn luyện và log {model_choice} vào MLflow ✅")
                st.markdown(f"Xem chi tiết: [{run_url}]({run_url})")

            st.session_state['model'] = model
            st.info("Mô hình đã được lưu vào session để sử dụng sau.")
