from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate and return metrics"""
    metrics = {
        'dataset': dataset_name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred)
    }
    return metrics
