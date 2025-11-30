import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, mean_squared_error, 
    mean_absolute_error, r2_score
)
from utils import setup_logging, load_pickle
import os

logger = setup_logging()

def compare_models(trained_models, training_history, X_test, y_test, problem_type):
    logger.info("Performing model comparison...")
    
    comparison_results = []
    
    for model_name, model in trained_models.items():
        try:
            predictions = model.predict(X_test)
            
            if problem_type == 'classification':
                accuracy = accuracy_score(y_test, predictions)
                result = {
                    'Model': model_name,
                    'Accuracy': accuracy,
                    'CV_Score': training_history[model_name].get('best_cv_score', 0)
                }
            else:
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, predictions)
                r2 = r2_score(y_test, predictions)
                
                result = {
                    'Model': model_name,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2_Score': r2,
                    'CV_Score': -training_history[model_name].get('best_cv_score', 0)
                }
            
            comparison_results.append(result)
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if not comparison_df.empty:
        if problem_type == 'classification':
            comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
            logger.info(f"Best model: {comparison_df.iloc[0]['Model']}")
        else:
            comparison_df = comparison_df.sort_values('RMSE', ascending=True)
            logger.info(f"Best model: {comparison_df.iloc[0]['Model']}")
    
    return comparison_df
