from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


def create_model(preprocessor, algorithm, **model_params):
    """Create ML pipeline with preprocessor and classifier."""
    
    if algorithm == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(**model_params)
    elif algorithm == "xgboost":
        from xgboost import XGBClassifier
        classifier = XGBClassifier(**model_params)
    elif algorithm == "lightgbm":
        from lightgbm import LGBMClassifier
        classifier = LGBMClassifier(**model_params)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', classifier)
    ])
    
    return pipeline


def save_model(model, path):
    """Save trained model"""
    try:
        joblib.dump(model, path)
        print(f'model saved to {path}')
    except Exception as e:
        print(f'Error saving model: {e}')


def load_model(path):
    """Load trained model"""
    return joblib.load(path)