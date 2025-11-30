import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, List, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    mean_squared_error, mean_absolute_error, r2_score,
    precision_recall_curve, auc
)
from utils import (
    setup_logging, 
    ensure_directory, 
    load_pickle,
    save_json,
    validate_target_column
)

logger = setup_logging()

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class ModelEvaluator:
 
    def __init__(self, models_dir: str = "d:/Python/machine-learning-pipeline/models", plots_dir: str = "d:/Python/machine-learning-pipeline/plots"):
        self.models_dir = models_dir
        self.plots_dir = plots_dir
        ensure_directory(plots_dir)
        
        self.evaluation_results = {}
        self.best_model_info = {}
        self.problem_type = None
    
    def load_models(self):
        logger.info("Loading trained models...")
        
        models = {}
        model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pkl') and not f.startswith('scaler') and not f.startswith('label_encoder') and not f.startswith('onehot')]
        
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            try:
                model = load_pickle(os.path.join(self.models_dir, model_file))
                models[model_name] = model
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Error loading model {model_file}: {str(e)}")
        
        return models
    
    def evaluate_classification_model(self, model_name, model, X_test, y_test):
        logger.info(f"Evaluating classification model: {model_name}")
        
        y_pred = model.predict(X_test)
        
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        cm = confusion_matrix(y_test, y_pred)
        
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        
        roc_auc = None
        if y_pred_proba is not None:
            try:
                if len(np.unique(y_test)) == 2:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
            except Exception as e:
                logger.warning(f"Could not compute ROC AUC for {model_name}: {str(e)}")
        
        evaluation_result = {
            'model_name': model_name,
            'problem_type': 'classification',
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc
            },
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report,
            'predictions': {
                'y_true': y_test.tolist(),
                'y_pred': y_pred.tolist(),
                'y_pred_proba': y_pred_proba.tolist() if y_pred_proba is not None else None
            }
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
        return evaluation_result
    
    def evaluate_regression_model(self, model_name, model, X_test, y_test):
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
    
    def plot_classification_metrics(self, evaluation_results):
        logger.info("Creating classification evaluation plots...")
        
        class_results = {k: v for k, v in evaluation_results.items() 
                        if v['problem_type'] == 'classification'}
        
        if not class_results:
            logger.warning("No classification results found for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(class_results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [class_results[model]['metrics'][metric] for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(f'{metric.upper().replace("_", " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            if len(models) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'classification_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        n_models = len(class_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(class_results.items()):
            cm = np.array(result['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'confusion_matrices.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in class_results.items():
            if result['predictions']['y_pred_proba'] is not None:
                y_true = np.array(result['predictions']['y_true'])
                y_proba = np.array(result['predictions']['y_pred_proba'])
                
                if len(np.unique(y_true)) == 2 and y_proba.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'roc_curves.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Classification evaluation plots saved")
    
    def plot_regression_metrics(self, evaluation_results):
        logger.info("Creating regression evaluation plots...")
        
        reg_results = {k: v for k, v in evaluation_results.items() 
                      if v['problem_type'] == 'regression'}
        
        if not reg_results:
            logger.warning("No regression results found for plotting")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        models = list(reg_results.keys())
        metrics = ['rmse', 'mae', 'r2_score', 'mape']
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, metric in enumerate(metrics):
            ax = axes[i//2, i%2]
            values = [reg_results[model]['metrics'][metric] for model in models]
            
            bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black')
            ax.set_title(f'{metric.upper().replace("_", " ")}', fontsize=12, fontweight='bold')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
            
            if len(models) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'regression_metrics_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        n_models = len(reg_results)
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(reg_results.items()):
            y_true = np.array(result['predictions']['y_true'])
            y_pred = np.array(result['predictions']['y_pred'])
            
            axes[i].scatter(y_true, y_pred, alpha=0.6, color=colors[i])
            axes[i].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Values')
            axes[i].set_ylabel('Predicted Values')
            axes[i].set_title(f'Actual vs Predicted - {model_name}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            r2 = result['metrics']['r2_score']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'actual_vs_predicted.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
        if n_models == 1:
            axes = [axes]
        
        for i, (model_name, result) in enumerate(reg_results.items()):
            y_pred = np.array(result['predictions']['y_pred'])
            residuals = np.array(result['predictions']['residuals'])
            
            axes[i].scatter(y_pred, residuals, alpha=0.6, color=colors[i])
            axes[i].axhline(y=0, color='r', linestyle='--')
            axes[i].set_xlabel('Predicted Values')
            axes[i].set_ylabel('Residuals')
            axes[i].set_title(f'Residual Plot - {model_name}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'residual_plots.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Regression evaluation plots saved")
    
    def create_model_comparison_table(self, evaluation_results: Dict[str, Dict]) -> pd.DataFrame:
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
            else:  # regression
                row.update({
                    'RMSE': result['metrics']['rmse'],
                    'MAE': result['metrics']['mae'],
                    'R2_Score': result['metrics']['r2_score'],
                    'MAPE': result['metrics']['mape']
                })
            
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        
        comparison_df.to_csv(os.path.join(self.models_dir, 'detailed_model_comparison.csv'), index=False)
        
        return comparison_df
    
    def identify_best_model(self, evaluation_results: Dict[str, Dict]) -> Dict[str, Any]:
        
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
        
        save_json(best_model_info, os.path.join(self.models_dir, 'best_model_info.json'))
        
        if class_results:
            best_model_name = best_model_info['classification']['model_name']
        elif reg_results:
            best_model_name = best_model_info['regression']['model_name']
        else:
            logger.error("No models found for evaluation")
            return best_model_info
        
        import shutil
        original_path = os.path.join(self.models_dir, f'{best_model_name}.pkl')
        best_model_path = os.path.join(self.models_dir, 'best_model.pkl')
        
        try:
            shutil.copy2(original_path, best_model_path)
            logger.info(f"Best model saved as 'best_model.pkl'")
        except Exception as e:
            logger.error(f"Error copying best model: {str(e)}")
        
        self.best_model_info = best_model_info
        return best_model_info
    
    def evaluate_all_models(self, X_test, y_test):
        logger.info("Starting comprehensive model evaluation...")
        
        self.problem_type = validate_target_column(
            pd.concat([X_test, y_test], axis=1), 
            y_test.name
        )
        
        models = self.load_trained_models()
        
        if not models:
            logger.error("No trained models found for evaluation")
            return {}
        
        evaluation_results = {}
        
        for model_name, model in models.items():
            try:
                if self.problem_type == 'classification':
                    result = self.evaluate_classification_model(model_name, model, X_test, y_test)
                else:
                    result = self.evaluate_regression_model(model_name, model, X_test, y_test)
                
                evaluation_results[model_name] = result
                
            except Exception as e:
                logger.error(f"Error evaluating model {model_name}: {str(e)}")
        
        self.evaluation_results = evaluation_results
        
        if self.problem_type == 'classification':
            self.plot_classification_metrics(evaluation_results)
        else:
            self.plot_regression_metrics(evaluation_results)
        
        comparison_df = self.create_model_comparison_table(evaluation_results)
        print("\nModel Comparison:")
        print(comparison_df.to_string(index=False))
        
        best_model_info = self.identify_best_model(evaluation_results)
        
        evaluation_report = {
            'evaluation_date': pd.Timestamp.now().isoformat(),
            'problem_type': self.problem_type,
            'test_set_size': len(X_test),
            'models_evaluated': len(evaluation_results),
            'best_model': best_model_info,
            'detailed_results': evaluation_results
        }
        
        save_json(evaluation_report, os.path.join(self.models_dir, 'evaluation_report.json'))
        
        logger.info(f"Evaluation complete! Evaluated {len(evaluation_results)} models")
        return evaluation_results


def main():
    evaluator = ModelEvaluator()
    
    try:
        X_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/X_test.csv")
        y_test = pd.read_csv("d:/Python/machine-learning-pipeline/data/processed/y_test.csv").iloc[:, 0]  # First column
        
        logger.info(f"Loaded test data: X_test {X_test.shape}, y_test {y_test.shape}")
        
    except FileNotFoundError:
        logger.error("Test data not found. Please run model training first.")
        return
    
    results = evaluator.evaluate_all_models(X_test, y_test)
    
    if results:
        print(f"\nEvaluation Complete!")
        print(f"Evaluated {len(results)} models")
        print(f"Results saved in: {evaluator.models_dir}")
        print(f"Plots saved in: {evaluator.plots_dir}")
    else:
        print("No models found for evaluation.")


if __name__ == "__main__":
    main()