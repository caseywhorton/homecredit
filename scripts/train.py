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
params_path = Path(__file__).parent.parent / "params.yaml"
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

numeric_features = params["features"].get("numeric")
categorical_features = params["features"].get("categorical")
ordinal_features = params["features"].get("ordinal")
flag_features = params["features"].get("flag")

experiment_name = params["experiment_name"]


from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def train_with_mlflow(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    model_params,
    experiment_name="default",
):
    """Train model with MLflow tracking"""

    # Set experiment before starting run
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            # Log parameters
            print("Log model parameters")
            mlflow.log_params(params["model"])

            # Log training config
            print("Log training config")
            mlflow.log_params({f"train_{k}": v for k, v in params["train"].items()})

            # Log features
            print("Log features")
            if numeric_features:
                mlflow.log_param("numeric_features", ",".join(numeric_features))
            if categorical_features:
                mlflow.log_param("categorical_features", ",".join(categorical_features))
            if ordinal_features:
                mlflow.log_param("ordinal_features", ",".join(ordinal_features))
            if flag_features:
                mlflow.log_param("flag_features", ",".join(flag_features))

            # Create and train model
            print("creating model")
            model = create_model(preprocessor, **model_params)
            print("fitting model")
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Get probability predictions for AUC
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]

            # Calculate AUC scores
            train_auc = roc_auc_score(y_train, y_train_proba)
            test_auc = roc_auc_score(y_test, y_test_proba)

            # Log metrics (including AUC)
            print("Logging metrics")
            mlflow.log_metrics(
                {
                    "train_accuracy": accuracy_score(y_train, y_train_pred),
                    "train_precision": precision_score(y_train, y_train_pred),
                    "train_recall": recall_score(y_train, y_train_pred),
                    "train_f1": f1_score(y_train, y_train_pred),
                    "train_auc": train_auc,
                    "test_accuracy": accuracy_score(y_test, y_test_pred),
                    "test_precision": precision_score(y_test, y_test_pred),
                    "test_recall": recall_score(y_test, y_test_pred),
                    "test_f1": f1_score(y_test, y_test_pred),
                    "test_auc": test_auc,
                }
            )

            # Create and log ROC curve
            print("Creating ROC curve")
            fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
            fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)

            plt.figure(figsize=(10, 6))
            plt.plot(
                fpr_train,
                tpr_train,
                label=f"Train ROC (AUC = {train_auc:.3f})",
                linewidth=2,
            )
            plt.plot(
                fpr_test,
                tpr_test,
                label=f"Test ROC (AUC = {test_auc:.3f})",
                linewidth=2,
            )
            plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
            plt.xlabel("False Positive Rate", fontsize=12)
            plt.ylabel("True Positive Rate", fontsize=12)
            plt.title("ROC Curve", fontsize=14)
            plt.legend(loc="lower right", fontsize=10)
            plt.grid(alpha=0.3)
            plt.tight_layout()

            # Save and log the figure
            roc_path = "roc_curve.png"
            plt.savefig(roc_path)
            mlflow.log_artifact(roc_path)
            plt.close()
            print("ROC curve logged")

            # Log model to MLflow
            print("Logging model")
            mlflow.sklearn.log_model(model, "model")

            # Also save to artifacts directory (for DVC tracking)
            artifacts_dir = Path(__file__).parent.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            model_path = artifacts_dir / "model_classifier.pkl"
            save_model(model, path=str(model_path))

            print(f"Model saved to: {model_path}")
            print(f"MLflow run logged")
            print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")

        except Exception as e:
            print(f"Error in process: {e}")

        return model


def main():
    print("\n" + "=" * 60)
    print("PARAMETERS")
    print("=" * 60)
    print(json.dumps(params, indent=2))
    print("=" * 60 + "\n")

    # Read data from source
    print(f"Loading data from: {params['filepath']['source_data']}")
    data_path = Path(__file__).parent.parent / params["filepath"]["source_data"]
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    # Read Credit Card Data from Source
    print(f"Loading data from: {params['filepath']['source_data_cc']}")
    data_path = Path(__file__).parent.parent / params["filepath"]["source_data_cc"]
    cc = pd.read_csv(data_path)
    print(f"   Shape: {cc.shape}")

    # Extract features and create train/test split
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df, cc)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    print("Dtypes of all training columns:")
    # Get the preprocessor
    print("\nCreating preprocessor...")

    feature_types = []

    # if feature_types are passed, then add them to the list
    if numeric_features:
        feature_types.append(numeric_features)
    if categorical_features:
        feature_types.append(categorical_features)
    if ordinal_features:
        feature_types.append(ordinal_features)
    if flag_features:
        feature_types.append(flag_features)

    preprocessor = get_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        flag_features=flag_features,
    )

    print("Numeric:", numeric_features)
    print("Categorical:", categorical_features)
    print("Ordinal:", ordinal_features)
    print("Flag:", flag_features)

    # Train the model
    print("\nTraining model with MLflow...")
    model = train_with_mlflow(
        X_train, X_test, y_train, y_test, preprocessor, params["model"], experiment_name
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    print("\n" + "*" * 60)
    print(f"Running train.py from {os.getcwd()}")
    print("*" * 60)
    main()
