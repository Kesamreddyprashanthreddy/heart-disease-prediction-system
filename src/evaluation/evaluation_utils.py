import pandas as pd
import os
import shutil
from utils import setup_logging, save_json

logger = setup_logging()

def create_model_comparison_table(evaluation_results, models_dir):
    logger.info("Creating model comparison table...")
    
    comparison_data = []
    
    for model_name, result in evaluation_results.items():
        row = {'Model': model_name, 'Problem_Type': result['problem_type']}
        
        if result['problem_type'] == 'classification':
            row.update({
                'Accuracy': result['metrics']['accuracy'],
                'Precision': result['metrics']['precision'],
                'Recall': result['metrics']['recall'],
                'F1_Score': result['metrics']['f1_score'],
                'ROC_AUC': result['metrics']['roc_auc']
            })
        else:
            row.update({
                'RMSE': result['metrics']['rmse'],
                'MAE': result['metrics']['mae'],
                'R2_Score': result['metrics']['r2_score'],
                'MAPE': result['metrics']['mape']
            })
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv(os.path.join(models_dir, 'detailed_model_comparison.csv'), index=False)
    
    return comparison_df

def identify_best_model(evaluation_results, models_dir):
    logger.info("Identifying best model...")
    
    class_results = {k: v for k, v in evaluation_results.items() 
                    if v['problem_type'] == 'classification'}
    reg_results = {k: v for k, v in evaluation_results.items() 
                  if v['problem_type'] == 'regression'}
    
    best_model_info = {}
    
    if class_results:
        best_class_model = max(class_results.items(), 
                             key=lambda x: x[1]['metrics']['f1_score'])
        
        best_model_info['classification'] = {
            'model_name': best_class_model[0],
            'f1_score': best_class_model[1]['metrics']['f1_score'],
            'accuracy': best_class_model[1]['metrics']['accuracy'],
            'all_metrics': best_class_model[1]['metrics']
        }
        
        logger.info(f"Best classification model: {best_class_model[0]} (F1: {best_class_model[1]['metrics']['f1_score']:.4f})")
    
    if reg_results:
        best_reg_model = max(reg_results.items(), 
                           key=lambda x: x[1]['metrics']['r2_score'])
        
        best_model_info['regression'] = {
            'model_name': best_reg_model[0],
            'r2_score': best_reg_model[1]['metrics']['r2_score'],
            'rmse': best_reg_model[1]['metrics']['rmse'],
            'all_metrics': best_reg_model[1]['metrics']
        }
        
        logger.info(f"Best regression model: {best_reg_model[0]} (R²: {best_reg_model[1]['metrics']['r2_score']:.4f})")
    
    save_json(best_model_info, os.path.join(models_dir, 'best_model_info.json'))
    
    if class_results:
        best_model_name = best_model_info['classification']['model_name']
    elif reg_results:
        best_model_name = best_model_info['regression']['model_name']
    else:
        logger.error("No models found")
        return best_model_info
    
    original_path = os.path.join(models_dir, f'{best_model_name}.pkl')
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    
    try:
        shutil.copy2(original_path, best_model_path)
        logger.info(f"Best model saved as 'best_model.pkl'")
    except Exception as e:
        logger.error(f"Error copying best model: {str(e)}")
    
    return best_model_info
