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
        X_train, X_valid, y_train, y_valid = train_test_split(X_temp, y_temp, test_size=valid_size / (1 - test_size), random_state=42)

        # Hiển thị kết quả chia dữ liệu
        st.write(f"Tập huấn luyện: {len(X_train)} mẫu ({train_size*100:.1f}%)")
        st.write(f"Tập validation: {len(X_valid)} mẫu ({valid_size*100:.1f}%)")
        st.write(f"Tập kiểm tra: {len(X_test)} mẫu ({test_size*100:.1f}%)")
        st.write("Dữ liệu huấn luyện (X_train):", X_train.head())
        st.write("Dữ liệu validation (X_valid):", X_valid.head())
        st.write("Dữ liệu kiểm tra (X_test):", X_test.head())

        # Log vào MLflow
        with mlflow.start_run(run_name="Data_Split"):
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("valid_size", valid_size)
            mlflow.log_param("train_size", train_size)
            st.success("Dữ liệu đã được chia và log vào MLflow ✅.")

        # Lưu dữ liệu vào session để dùng sau
        st.session_state['X_train'] = X_train
        st.session_state['X_valid'] = X_valid
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_valid'] = y_valid
        st.session_state['y_test'] = y_test

    # 2. Cross Validation
    st.subheader("Cross Validation (Tùy chọn) 🔄")
    use_cv = st.checkbox("Sử dụng Cross Validation")
    if use_cv:
        k_folds = st.slider("Chọn số lượng folds (k)", min_value=2, max_value=10, value=5)
        if st.button("Tạo Cross Validation Folds"):
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
            fold_data = []
            for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
                fold_data.append({
                    "fold": fold + 1,
                    "train_size": len(train_idx),
                    "valid_size": len(valid_idx)
                })
            fold_df = pd.DataFrame(fold_data)
            st.write("Cross Validation Folds:", fold_df)

            # Log folds vào MLflow
            with mlflow.start_run(run_name=f"CV_{k_folds}_Folds"):
                mlflow.log_param("k_folds", k_folds)
                for _, row in fold_df.iterrows():
                    mlflow.log_metric(f"fold_{row['fold']}_train_size", row["train_size"])
                    mlflow.log_metric(f"fold_{row['fold']}_valid_size", row["valid_size"])
            st.success(f"Tạo {k_folds}-fold cross validation và log vào MLflow ✅.")

    # 3. Huấn luyện mô hình
    st.subheader("Huấn luyện mô hình 🎯")
    if 'X_train' not in st.session_state:
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
        X_train = st.session_state['X_train']
        X_valid = st.session_state['X_valid']
        y_train = st.session_state['y_train']
        y_valid = st.session_state['y_valid']

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
            st.write(f"Tham số: {model_params}")
            st.write(f"Độ chính xác huấn luyện: {train_score:.4f}")
            st.write(f"Độ chính xác validation: {valid_score:.4f}")

            # Log vào MLflow
            mlflow.log_params(model_params)
            mlflow.log_metric("train_accuracy", train_score)
            mlflow.log_metric("valid_accuracy", valid_score)
            mlflow.sklearn.log_model(model, "model")
            st.success(f"Huấn luyện {model_choice} hoàn tất và log vào MLflow ✅.")

if __name__ == "__main__":
    train_model()