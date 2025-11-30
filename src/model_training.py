import pandas as pd
import os
from utils import (
    setup_logging, 
    ensure_directory, 
    save_json,
    validate_target_column
)
from training.model_config import setup_models
from training.train_model import train_single_model
from training.model_comparison import compare_models

logger = setup_logging()

def train_all_models(X_train, y_train, X_test, y_test, models_dir, 
                    use_grid_search=True, cv_folds=5):
    logger.info("Starting training for all models...")
    
    ensure_directory(models_dir)
    
    problem_type = validate_target_column(
        pd.concat([X_train, y_train], axis=1), 
        y_train.name
    )
    
    model_configs = setup_models(problem_type)
    
    trained_models = {}
    training_history = {}
    
    for model_name, config in model_configs.items():
        model, result = train_single_model(
            model_name, config, X_train, y_train, 
            problem_type, models_dir, use_grid_search, cv_folds
        )
        
        if model is not None:
            trained_models[model_name] = model
        training_history[model_name] = result
    
    training_summary = {
        'problem_type': problem_type,
        'training_date': pd.Timestamp.now().isoformat(),
        'dataset_info': {
            'training_samples': len(X_train),
            'features': len(X_train.columns),
            'target_name': y_train.name
        },
        'models_trained': len(training_history),
        'successful_models': sum(1 for r in training_history.values() if r['status'] == 'success'),
        'failed_models': sum(1 for r in training_history.values() if r['status'] == 'failed'),
        'training_results': training_history
    }
    
    save_json(training_summary, os.path.join(models_dir, 'training_summary.json'))
    
    logger.info(f"Training complete! {training_summary['successful_models']}/{training_summary['models_trained']} models trained successfully")
    
    # Compare models
    comparison_df = compare_models(trained_models, training_history, X_test, y_test, problem_type)
    
    return trained_models, training_history, comparison_df

def main():
    models_dir = "d:/Python/machine-learning-pipeline/models"
    
    try:
        X_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_train.csv")
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]
        
        logger.info(f"Loaded data: X_train {X_train.shape}, X_test {X_test.shape}")
        
    except FileNotFoundError:
        logger.warning("Processed data not found. Run feature engineering first.")
        from feature_engineering import main as run_feature_engineering
        run_feature_engineering()
        
        X_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_train.csv")
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]
    
    trained_models, training_history, comparison = train_all_models(
        X_train, y_train, X_test, y_test,
        models_dir, use_grid_search=True, cv_folds=5
    )
    
    print("\nModel Comparison:")
    print(comparison)
    
    comparison.to_csv(os.path.join(models_dir, 'model_comparison.csv'), index=False)
    
    print(f"\nModel Training Complete!")
    print(f"Trained {len(trained_models)} models")
    print(f"Models saved in: {models_dir}")

if __name__ == "__main__":
    main()
