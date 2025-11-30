import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from utils import setup_logging

logger = setup_logging()

def evaluate_classification_model(model_name, model, X_test, y_test):
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
            logger.warning(f"Could not compute ROC AUC: {str(e)}")
    
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
