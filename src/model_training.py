import pandas as pd
import numpy as np
import os
import warnings
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from utils import (
    setup_logging, 
    ensure_directory, 
    save_pickle,
    load_pickle,
    save_json,
    validate_target_column
)

warnings.filterwarnings('ignore')
logger = setup_logging()

class ModelTrainer:
    def __init__(self, models_dir="d:/Python/machine-learning-pipeline/models"):
        self.models_dir = models_dir
        ensure_directory(models_dir)
        
        self.trained_models = {}
        self.model_configs = {}
        self.training_history = {}
        self.problem_type = None
    
    def setup_models(self, problem_type):
        self.problem_type = problem_type
        
        if problem_type == 'classification':
            models = {
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'params': {
                        'C': [0.1, 1.0, 10.0],
                        'solver': ['liblinear', 'lbfgs']
                    }
                },
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1]
                    }
                }
            }
        else:
            models = {
                'linear_regression': {
                    'model': LinearRegression(),
                    'params': {
                        'fit_intercept': [True, False]
                    }
                },
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42, n_jobs=-1),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [None, 10, 20],
                        'min_samples_split': [2, 5]
                    }
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'params': {
                        'n_estimators': [100, 200],
                        'max_depth': [3, 6],
                        'learning_rate': [0.01, 0.1]
                    }
                }
            }
        
        self.model_configs = models
        logger.info(f"Set up {len(models)} models for {problem_type}")
        return models
    
    def train_single_model(self, model_name, X_train, y_train, use_grid_search=True, cv_folds=5):
        logger.info(f"Training {model_name}...")
        
        if model_name not in self.model_configs:
            raise ValueError(f"Model {model_name} not configured")
        
        config = self.model_configs[model_name]
        base_model = config['model']
        param_grid = config['params']
        
        training_result = {
            'model_name': model_name,
            'problem_type': self.problem_type,
            'training_samples': len(X_train),
            'features': X_train.columns.tolist(),
            'feature_count': len(X_train.columns)
        }
        
        try:
            if use_grid_search and param_grid:
                scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
                
                grid_search = GridSearchCV(
                    base_model,
                    param_grid,
                    cv=cv_folds,
                    scoring=scoring,
                    n_jobs=-1,
                    verbose=0
                )
                
                grid_search.fit(X_train, y_train)
                
                best_model = grid_search.best_estimator_
                training_result.update({
                    'best_params': grid_search.best_params_,
                    'best_cv_score': grid_search.best_score_,
                    'grid_search_used': True
                })
                
                logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
                logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
                
            else:
                best_model = base_model
                best_model.fit(X_train, y_train)
                
                scoring = 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error'
                cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
                
                training_result.update({
                    'best_params': best_model.get_params(),
                    'best_cv_score': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'grid_search_used': False
                })
            
            self.trained_models[model_name] = best_model
            
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            save_pickle(best_model, model_path)
            training_result['model_path'] = model_path
            
            if hasattr(best_model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
                
                feature_importance = dict(sorted(feature_importance.items(), 
                                               key=lambda x: x[1], reverse=True))
                training_result['feature_importance'] = feature_importance
                
                top_features = list(feature_importance.keys())[:10]
                logger.info(f"Top 10 features for {model_name}: {top_features}")
            
            elif hasattr(best_model, 'coef_'):
               
                if len(best_model.coef_.shape) == 1:
                    coefficients = dict(zip(X_train.columns, best_model.coef_))
                else:
                   
                    coefficients = dict(zip(X_train.columns, best_model.coef_[0]))
                
                coefficients = dict(sorted(coefficients.items(), 
                                         key=lambda x: abs(x[1]), reverse=True))
                training_result['coefficients'] = coefficients
                
                top_features = list(coefficients.keys())[:10]
                logger.info(f"Top 10 features by coefficient for {model_name}: {top_features}")
            
            training_result['status'] = 'success'
            logger.info(f"Successfully trained {model_name}")
            
        except Exception as e:
            error_msg = f"Error training {model_name}: {str(e)}"
            logger.error(error_msg)
            training_result.update({
                'status': 'failed',
                'error': error_msg
            })
        
        self.training_history[model_name] = training_result
        return training_result
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                        use_grid_search: bool = True, cv_folds: int = 5) -> Dict[str, Dict]:
        
        logger.info("Starting training for all models...")
        
        problem_type = validate_target_column(
            pd.concat([X_train, y_train], axis=1), 
            y_train.name
        )
        
        self.setup_models(problem_type)
        
        all_results = {}
        
        for model_name in self.model_configs.keys():
            result = self.train_single_model(
                model_name, X_train, y_train, use_grid_search, cv_folds
            )
            all_results[model_name] = result
        
        # Save training summary
        training_summary = {
            'problem_type': problem_type,
            'training_date': pd.Timestamp.now().isoformat(),
            'dataset_info': {
                'training_samples': len(X_train),
                'features': len(X_train.columns),
                'target_name': y_train.name
            },
            'models_trained': len(all_results),
            'successful_models': sum(1 for r in all_results.values() if r['status'] == 'success'),
            'failed_models': sum(1 for r in all_results.values() if r['status'] == 'failed'),
            'training_results': all_results
        }
        
        save_json(training_summary, os.path.join(self.models_dir, 'training_summary.json'))
        
        logger.info(f"Training complete! {training_summary['successful_models']}/{training_summary['models_trained']} models trained successfully")
        
        return all_results
    
    def get_model_predictions(self, model_name: str, X_test: pd.DataFrame) -> np.ndarray:
        if model_name not in self.trained_models:
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                self.trained_models[model_name] = load_pickle(model_path)
            else:
                raise ValueError(f"Model {model_name} not found")
        
        model = self.trained_models[model_name]
        predictions = model.predict(X_test)
        return predictions
    
    def get_model_probabilities(self, model_name: str, X_test: pd.DataFrame) -> Optional[np.ndarray]:
        if self.problem_type != 'classification':
            logger.warning("Probabilities only available for classification models")
            return None
        
        if model_name not in self.trained_models:
            model_path = os.path.join(self.models_dir, f'{model_name}.pkl')
            if os.path.exists(model_path):
                self.trained_models[model_name] = load_pickle(model_path)
            else:
                raise ValueError(f"Model {model_name} not found")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X_test)
            return probabilities
        else:
            logger.warning(f"Model {model_name} does not support probability predictions")
            return None
    
    def compare_models_quick(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        logger.info("Performing quick model comparison...")
        
        comparison_results = []
        
        for model_name in self.trained_models.keys():
            try:
                predictions = self.get_model_predictions(model_name, X_test)
                
                if self.problem_type == 'classification':
                    accuracy = accuracy_score(y_test, predictions)
                    result = {
                        'Model': model_name,
                        'Accuracy': accuracy,
                        'CV_Score': self.training_history[model_name].get('best_cv_score', 0)
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
                        'CV_Score': -self.training_history[model_name].get('best_cv_score', 0)  # Convert back from negative
                    }
                
                comparison_results.append(result)
                
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
        
        comparison_df = pd.DataFrame(comparison_results)
        
        if not comparison_df.empty:
            if self.problem_type == 'classification':
                comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
                logger.info(f"Best model by accuracy: {comparison_df.iloc[0]['Model']}")
            else:
                comparison_df = comparison_df.sort_values('RMSE', ascending=True)
                logger.info(f"Best model by RMSE: {comparison_df.iloc[0]['Model']}")
        
        return comparison_df


def main():
    trainer = ModelTrainer()
    
    try:
        X_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_train.csv")
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_train.csv").iloc[:, 0]  # First column
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]    # First column
        
        logger.info(f"Loaded processed data: X_train {X_train.shape}, X_test {X_test.shape}")
        
    except FileNotFoundError:
        logger.warning("Processed data not found. Running feature engineering first...")
        from feature_engineering import main as run_feature_engineering
        run_feature_engineering()
        
        X_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_train.csv")
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_train = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_train.csv").iloc[:, 0]
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]
    
    training_results = trainer.train_all_models(
        X_train, y_train, 
        use_grid_search=True, 
        cv_folds=5
    )
    
    comparison = trainer.compare_models_quick(X_test, y_test)
    print("\nModel Comparison:")
    print(comparison)
    
    comparison.to_csv(os.path.join(trainer.models_dir, 'model_comparison.csv'), index=False)
    
    print(f"\nModel Training Complete!")
    print(f"Trained {len(trainer.trained_models)} models")
    print(f"Models saved in: {trainer.models_dir}")


if __name__ == "__main__":
    main()