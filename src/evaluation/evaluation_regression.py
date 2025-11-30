import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score
)
from utils import setup_logging

logger = setup_logging()

def evaluate_regression_model(model_name, model, X_test, y_test):
    logger.info(f"Evaluating regression model: {model_name}")
    
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    residuals = y_test - y_pred
    
    evaluation_result = {
        'model_name': model_name,
        'problem_type': 'regression',
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mse': mse,
            'mape': mape
        },
        'predictions': {
            'y_true': y_test.tolist(),
            'y_pred': y_pred.tolist(),
            'residuals': residuals.tolist()
        }
    }
    
    logger.info(f"{model_name} - RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return evaluation_result
