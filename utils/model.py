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
    joblib.dump(model, path)


def load_model(path):
    """Load trained model"""
    return joblib.load(path)