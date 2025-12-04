import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import json
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import prepare_data, get_preprocessor
from utils.model import create_model, save_model

# GET PARAMETERS
# Fix: Use correct relative path from scripts/ directory
params_path = Path(__file__).parent.parent / 'params.yaml'
with open(params_path, 'r') as f:
    params = yaml.safe_load(f)

numeric_features = params['features']['numeric']
categorical_features = params['features']['categorical']
experiment_name = params['experiment_name']
#experiment_name = "homecredit-default-11102025"

def train_with_mlflow(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    model_params,
    experiment_name="default"
):
    """Train model with MLflow tracking"""
    
    # Set experiment before starting run
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        try:
            # Log parameters
            mlflow.log_params(params['model'])
            
            # Log training config
            mlflow.log_params({
                f'train_{k}': v for k, v in params['train'].items()
            })

            # Log features
            mlflow.log_param('n_numeric_features', len(params['features']['numeric']))
            mlflow.log_param('numeric_features', ','.join(params['features']['numeric']))

            # Create and train model
            model = create_model(preprocessor, **model_params)  # Fix: unpack dict
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Fix: Wrong metrics assigned
            mlflow.log_metrics({
                'train_accuracy': accuracy_score(y_train, y_train_pred),
                'train_precision': precision_score(y_train, y_train_pred),
                'train_recall': recall_score(y_train, y_train_pred),
                'train_f1': f1_score(y_train, y_train_pred),
                'test_accuracy': accuracy_score(y_test, y_test_pred),
                'test_precision': precision_score(y_test, y_test_pred),
                'test_recall': recall_score(y_test, y_test_pred),
                'test_f1': f1_score(y_test, y_test_pred)
            })
            
            # Log model to MLflow
            mlflow.sklearn.log_model(model, "model")
            
            # Also save to artifacts directory (for DVC tracking)
            artifacts_dir = Path(__file__).parent.parent / 'artifacts'
            artifacts_dir.mkdir(exist_ok=True)
            
            model_path = artifacts_dir / 'model_classifier.pkl'
            save_model(model, path=str(model_path))
            
            print(f"Model saved to: {model_path}")
            print(f"MLflow run logged")
        except Exception as e:
            print(f"Error in process: {e}")

        return model


def main():
    print('\n' + '='*60)
    print('PARAMETERS')
    print('='*60)
    print(json.dumps(params, indent=2))
    print('='*60 + '\n')

    # Read data from source
    print(f"Loading data from: {params['filepath']['source_data']}")
    data_path = Path(__file__).parent.parent / params['filepath']['source_data']
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    # Extract features and create train/test split
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Get the preprocessor
    print("\nCreating preprocessor...")
    preprocessor = get_preprocessor(numeric_features, categorical_features)

    # Train the model
    print("\nTraining model with MLflow...")
    model = train_with_mlflow(
        X_train,
        X_test,
        y_train, 
        y_test,
        preprocessor,
        params['model'], 
        experiment_name
    )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    print('\n' + '*'*60)
    print(f'Running train.py from {os.getcwd()}')
    print('*'*60)
    main()