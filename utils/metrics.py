import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculates standard regression metrics for the air quality forecasting.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2)
    }
