import os
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from utils import setup_logging, save_pickle

logger = setup_logging()

def train_single_model(model_name, config, X_train, y_train, problem_type, models_dir, 
                      use_grid_search=True, cv_folds=5):
    logger.info(f"Training {model_name}...")
    
    base_model = config['model']
    param_grid = config['params']
    
    training_result = {
        'model_name': model_name,
        'problem_type': problem_type,
        'training_samples': len(X_train),
        'features': X_train.columns.tolist(),
        'feature_count': len(X_train.columns)
    }
    
    try:
        if use_grid_search and param_grid:
            scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
            
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
            
            logger.info(f"Best parameters: {grid_search.best_params_}")
            logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
            
        else:
            best_model = base_model
            best_model.fit(X_train, y_train)
            
            scoring = 'accuracy' if problem_type == 'classification' else 'neg_mean_squared_error'
            cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv_folds, scoring=scoring)
            
            training_result.update({
                'best_params': best_model.get_params(),
                'best_cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'grid_search_used': False
            })
        
        model_path = os.path.join(models_dir, f'{model_name}.pkl')
        save_pickle(best_model, model_path)
        training_result['model_path'] = model_path
        
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
            training_result['feature_importance'] = feature_importance
            
            top_features = list(feature_importance.keys())[:10]
            logger.info(f"Top 10 features: {top_features}")
        
        elif hasattr(best_model, 'coef_'):
            if len(best_model.coef_.shape) == 1:
                coefficients = dict(zip(X_train.columns, best_model.coef_))
            else:
                coefficients = dict(zip(X_train.columns, best_model.coef_[0]))
            
            coefficients = dict(sorted(coefficients.items(), 
                                     key=lambda x: abs(x[1]), reverse=True))
            training_result['coefficients'] = coefficients
            
            top_features = list(coefficients.keys())[:10]
            logger.info(f"Top 10 features by coefficient: {top_features}")
        
        training_result['status'] = 'success'
        logger.info(f"Successfully trained {model_name}")
        
    except Exception as e:
        error_msg = f"Error training {model_name}: {str(e)}"
        logger.error(error_msg)
        training_result.update({
            'status': 'failed',
            'error': error_msg
        })
        best_model = None
    
    return best_model, training_result
