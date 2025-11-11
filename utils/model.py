from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import joblib


def create_model(preprocessor, **model_params):
    """Create full modeling pipeline"""
    rf = RandomForestClassifier(**model_params)
 
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', rf)
    ])

    return model


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