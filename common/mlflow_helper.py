import mlflow
from typing import Dict, Any
import json

def log_preprocessing_params(steps: Dict[str, Any]) -> None:
    """
    Log preprocessing parameters to MLflow in an optimized way.
    
    Args:
        steps (dict): Dictionary containing preprocessing parameters
    """
    try:
        # Flatten nested dictionaries and handle various data types
        def flatten_params(params: Dict[str, Any], parent_key: str = "") -> Dict[str, str]:
            items = {}
            for key, value in params.items():
                new_key = f"{parent_key}_{key}" if parent_key else key
                
                if isinstance(value, (dict, list, tuple)):
                    # Convert complex types to JSON string
                    items[new_key] = json.dumps(value, ensure_ascii=False)
                elif isinstance(value, (int, float, bool, str)):
                    # Direct conversion to string for simple types
                    items[new_key] = str(value)
                elif value is None:
                    items[new_key] = "None"
                else:
                    # Fallback for other types
                    items[new_key] = str(value)
            return items

        # Flatten and log parameters efficiently
        flattened_params = flatten_params(steps)
        
        # Batch log parameters instead of individual calls
        mlflow.log_params(flattened_params)
        
    except Exception as e:
        raise Exception(f"Error logging preprocessing params: {str(e)}")

def log_metrics(metrics: Dict[str, float], step: int = None) -> None:
    """
    Log metrics to MLflow in an optimized way.
    
    Args:
        metrics (dict): Dictionary containing metrics
        step (int, optional): Step number for time-series metrics
    """
    try:
        # Ensure all values are numeric
        cleaned_metrics = {k: float(v) for k, v in metrics.items()}
        
        # Batch log metrics
        if step is None:
            mlflow.log_metrics(cleaned_metrics)
        else:
            # Log metrics with step for time-series tracking
            for metric_name, value in cleaned_metrics.items():
                mlflow.log_metric(metric_name, value, step=step)
                
    except Exception as e:
        raise Exception(f"Error logging metrics: {str(e)}")