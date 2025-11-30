import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import roc_curve, auc
from utils import setup_logging

logger = setup_logging()

def plot_classification_metrics(evaluation_results, plots_dir):
    logger.info("Creating classification evaluation plots...")
    
    class_results = {k: v for k, v in evaluation_results.items() 
                    if v['problem_type'] == 'classification'}
    
    if not class_results:
        logger.warning("No classification results found")
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
    plt.savefig(os.path.join(plots_dir, 'classification_metrics_comparison.png'), 
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
    plt.savefig(os.path.join(plots_dir, 'confusion_matrices.png'), 
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
    plt.savefig(os.path.join(plots_dir, 'roc_curves.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Classification plots saved")
