import mlflow

def log_preprocessing_params(params):
    with mlflow.start_run(run_name="Preprocessing"):
        for key, value in params.items():
            mlflow.log_param(key, value)