import pandas as pd
import os
from utils import (
    setup_logging, 
    ensure_directory, 
    load_pickle,
    save_json,
    validate_target_column
)
from evaluation_classification import evaluate_classification_model
from evaluation_regression import evaluate_regression_model
from evaluation_plots_classification import plot_classification_metrics
from evaluation_plots_regression import plot_regression_metrics
from evaluation_utils import create_model_comparison_table, identify_best_model

logger = setup_logging()

def load_trained_models(models_dir):
    logger.info("Loading trained models...")
    
    models = {}
    model_files = [f for f in os.listdir(models_dir) 
                  if f.endswith('.pkl') and not f.startswith('scaler') 
                  and not f.startswith('label_encoder') and not f.startswith('onehot')]
    
    for model_file in model_files:
        model_name = model_file.replace('.pkl', '')
        try:
            model = load_pickle(os.path.join(models_dir, model_file))
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error loading {model_file}: {str(e)}")
    
    return models

def evaluate_all_models(X_test, y_test, models_dir, plots_dir):
    logger.info("Starting comprehensive model evaluation...")
    
    ensure_directory(plots_dir)
    
    problem_type = validate_target_column(
        pd.concat([X_test, y_test], axis=1), 
        y_test.name
    )
    
    models = load_trained_models(models_dir)
    
    if not models:
        logger.error("No trained models found")
        return {}
    
    evaluation_results = {}
    
    for model_name, model in models.items():
        try:
            if problem_type == 'classification':
                result = evaluate_classification_model(model_name, model, X_test, y_test)
            else:
                result = evaluate_regression_model(model_name, model, X_test, y_test)
            
            evaluation_results[model_name] = result
            
        except Exception as e:
            logger.error(f"Error evaluating {model_name}: {str(e)}")
    
    if problem_type == 'classification':
        plot_classification_metrics(evaluation_results, plots_dir)
    else:
        plot_regression_metrics(evaluation_results, plots_dir)
    
    comparison_df = create_model_comparison_table(evaluation_results, models_dir)
    print("\nModel Comparison:")
    print(comparison_df.to_string(index=False))
    
    best_model_info = identify_best_model(evaluation_results, models_dir)
    
    evaluation_report = {
        'evaluation_date': pd.Timestamp.now().isoformat(),
        'problem_type': problem_type,
        'test_set_size': len(X_test),
        'models_evaluated': len(evaluation_results),
        'best_model': best_model_info,
        'detailed_results': evaluation_results
    }
    
    save_json(evaluation_report, os.path.join(models_dir, 'evaluation_report.json'))
    
    logger.info(f"Evaluation complete! Evaluated {len(evaluation_results)} models")
    return evaluation_results

def main():
    models_dir = "d:/Python/machine-learning-pipeline/models"
    plots_dir = "d:/Python/machine-learning-pipeline/plots"
    
    try:
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]
        
        logger.info(f"Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}")
        
    except FileNotFoundError:
        logger.error("Test data not found. Run model training first.")
        return
    
    results = evaluate_all_models(X_test, y_test, models_dir, plots_dir)
    
    if results:
        print(f"\nEvaluation Complete!")
        print(f"Evaluated {len(results)} models")
        print(f"Results saved in: {models_dir}")
        print(f"Plots saved in: {plots_dir}")
    else:
        print("No models found for evaluation.")

if __name__ == "__main__":
    main()
