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

# Thiết lập tracking URI cục bộ
mlflow.set_tracking_uri(f"file://{os.path.abspath('mlruns')}")

def train_model():
    st.header("Train Titanic Survival Model 🧑‍🚀")

    # Cho người dùng đặt tên Experiment
    experiment_name = st.text_input("Enter Experiment Name for Training", value="Titanic_Training")
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # Đọc file đã tiền xử lý
    processed_file = "exercises/exercise_1/data/processed/titanic_processed.csv"
    try:
        data = load_data(processed_file)
        st.subheader("Dữ liệu đã tiền xử lý (Sau khi xử lý) 📝")
        st.write("Đây là dữ liệu sau các bước tiền xử lý trong 'Tiền xử lý dữ liệu Titanic':")
        st.write(data.head())
    except FileNotFoundError:
        st.error("Dữ liệu đã xử lý không tìm thấy. Vui lòng hoàn tất tiền xử lý trong 'Tiền xử lý dữ liệu Titanic' trước.")
        return

    # Chia dữ liệu thành X và y
    X = data.drop(columns=['Survived'])
    y = data['Survived']

    # 1. Chia dữ liệu train/test/validation
    st.subheader("Chia dữ liệu 🔀")
    test_size = st.slider("Chọn kích thước tập kiểm tra (%)", min_value=10, max_value=50, value=20, step=5) / 100
    remaining_size = 1 - test_size
    valid_size_relative = st.slider(
        "Chọn kích thước tập validation (% dữ liệu còn lại sau khi chia test)",
        min_value=0, max_value=50, value=20, step=5
    ) / 100
    valid_size = remaining_size * valid_size_relative
    train_size = remaining_size - valid_size

    if st.button("Chia dữ liệu"):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train_initial, X_valid, y_train_initial, y_valid = train_test_split(
            X_temp, y_temp, test_size=valid_size / (1 - test_size), random_state=42
        )

        # Hiển thị kết quả chia dữ liệu
        st.write(f"Tập huấn luyện ban đầu: {len(X_train_initial)} mẫu ({train_size*100:.1f}%)")
        st.write(f"Tập validation: {len(X_valid)} mẫu ({valid_size*100:.1f}%)")
        st.write(f"Tập kiểm tra: {len(X_test)} mẫu ({test_size*100:.1f}%)")
        st.write("Dữ liệu huấn luyện (X_train):", X_train_initial.head())
        st.write("Dữ liệu validation (X_valid):", X_valid.head())
        st.write("Dữ liệu kiểm tra (X_test):", X_test.head())

        # Log vào MLflow
        with mlflow.start_run(run_name="Data_Split"):
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("valid_size", valid_size)
            mlflow.log_param("train_size", train_size)
            st.success("Dữ liệu đã được chia và log vào MLflow ✅.")

        # Lưu dữ liệu vào session để dùng sau
        st.session_state['X_train_initial'] = X_train_initial
        st.session_state['X_valid'] = X_valid
        st.session_state['X_test'] = X_test
        st.session_state['y_train_initial'] = y_train_initial
        st.session_state['y_valid'] = y_valid
        st.session_state['y_test'] = y_test

    # 2. Cross Validation
    st.subheader("Cross Validation (Tùy chọn) 🔄")
    use_cv = st.checkbox("Sử dụng Cross Validation")
    if use_cv:
        if 'X_train_initial' not in st.session_state or 'X_test' not in st.session_state:
            st.warning("Vui lòng chia dữ liệu trước khi thực hiện Cross Validation.")
            return

        k_folds = st.slider("Chọn số lượng folds (k)", min_value=2, max_value=10, value=5)

        # Khởi tạo danh sách tỷ lệ valid cho từng fold (tổng = 100%)
        if 'valid_sizes' not in st.session_state:
            st.session_state['valid_sizes'] = [20] * k_folds  # Mặc định 20% cho mỗi fold

        # Slider cho từng fold, tổng tỷ lệ valid phải bằng 100%
        st.write("Phân bổ tỷ lệ tập valid cho từng fold (tổng = 100%)")
        valid_sizes = []
        total_valid = 0
        for i in range(k_folds):
            fold_num = i + 1
            valid_size = st.slider(
                f"Chọn kích thước tập valid cho Fold {fold_num} (%)",
                min_value=0, max_value=100 - total_valid, value=st.session_state['valid_sizes'][i],
                key=f"valid_size_fold_{fold_num}"
            )
            valid_sizes.append(valid_size)
            total_valid += valid_size
            if total_valid > 100:
                st.error("Tổng tỷ lệ valid vượt quá 100%. Vui lòng điều chỉnh lại.")
                return

        if st.button("Tạo và tùy chỉnh Cross Validation Folds"):
            if total_valid != 100:
                st.error("Tổng tỷ lệ valid phải bằng 100%. Vui lòng điều chỉnh lại.")
                return

            X_train_initial = st.session_state['X_train_initial']
            y_train_initial = st.session_state['y_train_initial']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_configs = {}
            fold_indices = list(kf.split(X_train_initial))

            # Tạo và tùy chỉnh từng fold với tỷ lệ valid đã chọn
            for fold, (train_idx, valid_idx) in enumerate(fold_indices):
                fold_num = fold + 1
                X_remaining = X_train_initial.iloc[train_idx.tolist() + valid_idx.tolist()]
                y_remaining = y_train_initial.iloc[train_idx.tolist() + valid_idx.tolist()]

                valid_size_fold_relative = valid_sizes[fold] / 100  # Chuyển về tỷ lệ (0-1)

                # Chia dữ liệu cho fold này
                X_train_fold, X_valid_fold, y_train_fold, y_valid_fold = train_test_split(
                    X_remaining, y_remaining, test_size=valid_size_fold_relative, random_state=42
                )

                # Lưu cấu hình fold
                fold_configs[fold_num] = {
                    "X_train": X_train_fold,
                    "y_train": y_train_fold,
                    "X_valid": X_valid_fold,
                    "y_valid": y_valid_fold,
                    "train_size": len(X_train_fold),
                    "valid_size": len(X_valid_fold),
                    "valid_size_relative": valid_size_fold_relative
                }

            st.session_state['fold_configs'] = fold_configs
            st.session_state['valid_sizes'] = valid_sizes  # Cập nhật tỷ lệ valid đã chọn

            # Hiển thị tổng quan tất cả các fold đã tùy chỉnh
            st.subheader("Tổng quan các Fold đã tùy chỉnh")
            fold_summary = []
            for fold_num, config in fold_configs.items():
                fold_summary.append({
                    "Fold": fold_num,
                    "Train Size": config["train_size"],
                    "Valid Size": config["valid_size"],
                    "Test Size": len(X_test)
                })
            fold_summary_df = pd.DataFrame(fold_summary)
            st.write(fold_summary_df)

            # Log tổng quan folds vào MLflow
            with mlflow.start_run(run_name=f"CV_{k_folds}_Folds_Summary"):
                mlflow.log_param("k_folds", k_folds)
                for i, size in enumerate(valid_sizes):
                    mlflow.log_param(f"fold_{i+1}_valid_size", size)
                for _, row in fold_summary_df.iterrows():
                    mlflow.log_metric(f"fold_{row['Fold']}_train_size", row["Train Size"])
                    mlflow.log_metric(f"fold_{row['Fold']}_valid_size", row["Valid Size"])
                    mlflow.log_metric(f"fold_{row['Fold']}_test_size", row["Test Size"])
            st.success(f"Tạo và tùy chỉnh {k_folds}-fold cross validation, log vào MLflow ✅.")

    # 3. Huấn luyện mô hình
    st.subheader("Huấn luyện mô hình 🎯")
    if 'X_train_initial' not in st.session_state:
        st.warning("Vui lòng chia dữ liệu trước khi huấn luyện.")
        return

    model_choice = st.selectbox(
        "Chọn mô hình",
        ["Random Forest", "Logistic Regression", "Polynomial Regression"]
    )

    # Tham số cho từng mô hình
    if model_choice == "Random Forest":
        n_estimators = st.slider("Số lượng cây (n_estimators)", 10, 200, 100, step=10)
        max_depth = st.slider("Độ sâu tối đa", 1, 20, 10, step=1)
        model_params = {"n_estimators": n_estimators, "max_depth": max_depth, "random_state": 42}
    elif model_choice == "Logistic Regression":
        C = st.slider("Cường độ điều chỉnh (C)", 0.01, 10.0, 1.0, step=0.01)
        max_iter = st.slider("Số vòng lặp tối đa", 100, 1000, 100, step=100)
        model_params = {"C": C, "max_iter": max_iter, "random_state": 42}
    elif model_choice == "Polynomial Regression":
        degree = st.slider("Bậc đa thức", 1, 5, 2, step=1)
        C = st.slider("Cường độ điều chỉnh (C)", 0.01, 10.0, 1.0, step=0.01)
        model_params = {"degree": degree, "C": C, "random_state": 42}

    if st.button("Huấn luyện mô hình"):
        if use_cv and 'fold_configs' in st.session_state and st.session_state['fold_configs']:
            selected_fold_train = st.selectbox("Chọn fold để huấn luyện", list(st.session_state['fold_configs'].keys()))
            config = st.session_state['fold_configs'][selected_fold_train]
            X_train = config['X_train']
            X_valid = config['X_valid']
            y_train = config['y_train']
            y_valid = config['y_valid']
            train_source = f"Fold {selected_fold_train} (đã tùy chỉnh)"
        else:
            X_train = st.session_state['X_train_initial']
            X_valid = st.session_state['X_valid']
            y_train = st.session_state['y_train_initial']
            y_valid = st.session_state['y_valid']
            train_source = "Dữ liệu ban đầu"

        with mlflow.start_run(run_name=f"{model_choice}_Titanic"):
            # Khởi tạo mô hình dựa trên lựa chọn
            if model_choice == "Random Forest":
                model = RandomForestClassifier(**model_params)
            elif model_choice == "Logistic Regression":
                model = LogisticRegression(**model_params)
            elif model_choice == "Polynomial Regression":
                model = Pipeline([
                    ("poly", PolynomialFeatures(degree=model_params["degree"])),
                    ("logistic", LogisticRegression(C=model_params["C"], random_state=42))
                ])

            # Huấn luyện mô hình
            model.fit(X_train, y_train)
            train_score = model.score(X_train, y_train)
            valid_score = model.score(X_valid, y_valid)

            # Hiển thị thông tin
            st.write(f"Mô hình đã chọn: {model_choice}")
            st.write(f"Nguồn dữ liệu huấn luyện: {train_source}")
            st.write(f"Tham số: {model_params}")
            st.write(f"Độ chính xác huấn luyện: {train_score:.4f}")
            st.write(f"Độ chính xác validation: {valid_score:.4f}")

            # Log vào MLflow
            mlflow.log_params(model_params)
            mlflow.log_param("train_source", train_source)
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("valid_accuracy", valid_score)
            mlflow.sklearn.log_model(model, "model")
            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow ✅.")

if __name__ == "__main__":
    train_model()