import mlflow

def log_preprocessing_params(steps):
    for key, value in steps.items():
        if isinstance(value, list):
            mlflow.log_param(key, ",".join(map(str, value)))
        else:
            mlflow.log_param(key, str(value))