import numpy as np
import matplotlib.pyplot as plt
import os
from utils import setup_logging

logger = setup_logging()

def plot_regression_metrics(evaluation_results, plots_dir):
    logger.info("Creating regression evaluation plots...")
    
    reg_results = {k: v for k, v in evaluation_results.items() 
                  if v['problem_type'] == 'regression'}
    
    if not reg_results:
        logger.warning("No regression results found")
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
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                   f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        if len(models) > 3:
            ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'regression_metrics_comparison.png'), 
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
    plt.savefig(os.path.join(plots_dir, 'actual_vs_predicted.png'), 
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
    plt.savefig(os.path.join(plots_dir, 'residual_plots.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Regression plots saved")
