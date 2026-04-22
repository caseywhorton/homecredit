"""
Model training script with MLflow tracking.

This script loads data, prepares features, trains a classification model,
and logs all experiments to MLflow for tracking and comparison.
"""

import json
import os
import sys
from pathlib import Path
import shap

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model import create_model, save_model
from utils.preprocessing import get_preprocessor, prepare_data
from utils.evaluate import (
    cross_validate_model,
    _find_optimal_threshold,
    _create_and_log_roc_curve,
    _create_and_log_pr_curve,
)

# =============================================================================
# Load Configuration
# =============================================================================

# Load parameters from params.yaml
params_path = Path(__file__).parent.parent / "params.yaml"
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

# Extract feature lists from parameters
numeric_features = params["features"].get("numeric")
categorical_features = params["features"].get("categorical")
ordinal_features = params["features"].get("ordinal")
flag_features = params["features"].get("flag")

experiment_name = params["experiment_name"]


# =============================================================================
# Training Functions
# =============================================================================


def train_with_mlflow(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    model_params,
    experiment_name="default",
):
    """
    Train model with MLflow experiment tracking.

    Args:
        X_train: Training feature matrix
        X_test: Test feature matrix
        y_train: Training target vector
        y_test: Test target vector
        preprocessor: Sklearn preprocessing pipeline
        model_params: Dictionary of model hyperparameters
        experiment_name: Name of MLflow experiment

    Returns:
        Trained model object
    """
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():
        try:
            # Log model parameters
            print("Logging model parameters...")
            mlflow.log_params(params["model"])

            # Log training configuration
            print("Logging training config...")
            mlflow.log_params({f"train_{k}": v for k, v in params["train"].items()})

            # Log feature lists
            print("Logging features...")
            if numeric_features:
                mlflow.log_param("numeric_features", ",".join(numeric_features))
            if categorical_features:
                mlflow.log_param("categorical_features", ",".join(categorical_features))
            if ordinal_features:
                mlflow.log_param("ordinal_features", ",".join(ordinal_features))
            if flag_features:
                mlflow.log_param("flag_features", ",".join(flag_features))

            # Create and train model
            print("Creating model with model_params...")
            model = create_model(preprocessor, algorithm="lightgbm", **model_params)

            # Perform cross-validation
            if params["train"].get("use_cv", True):
                cv_folds = params["train"].get("cv_folds", 5)
                cv_results = cross_validate_model(model, X_train, y_train, cv=cv_folds)

                # Log CV metrics to MLflow
                print("Logging cross-validation metrics...")
                for metric in ["roc_auc", "f1", "precision", "recall"]:
                    mlflow.log_metric(
                        f"cv_{metric}_mean",
                        cv_results[f"test_{metric}"].mean(),
                    )
                    mlflow.log_metric(
                        f"cv_{metric}_std",
                        cv_results[f"test_{metric}"].std(),
                    )
                    mlflow.log_metric(
                        f"cv_train_{metric}_mean",
                        cv_results[f"train_{metric}"].mean(),
                    )

            model = create_model(preprocessor, algorithm="lightgbm", **model_params)

            print("Fitting final model on full training set...")
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

            # Log all metrics
            print("Logging metrics...")
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

            # Find optimal threshold based on F1
            optimal_threshold, best_f1, threshold_df = _find_optimal_threshold(
                y_test, y_test_proba
            )

            # Create and log ROC curve
            print("Creating ROC curve...")
            _create_and_log_roc_curve(
                y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc
            )

            # Create and log PR curve and threshold analysis
            print("Creating PR curve and threshold analysis...")
            _create_and_log_pr_curve(
                y_test, y_test_proba, optimal_threshold, threshold_df
            )

            # Create and log SHAP feature importance
            print("Creating SHAP feature importance plot...")
            _create_and_log_shap_importance(model, X_train)

            # Log model to MLflow
            print("Logging model to MLflow...")
            mlflow.sklearn.log_model(model, "model")

            # Save model to artifacts directory for DVC tracking
            artifacts_dir = Path(__file__).parent.parent / "artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            model_path = artifacts_dir / "model_classifier.pkl"
            save_model(model, path=str(model_path))

            print(f"Model saved to: {model_path}")
            print("MLflow run logged successfully")
            print(f"Train AUC: {train_auc:.3f}, Test AUC: {test_auc:.3f}")

        except Exception as e:
            print(f"Error in training process: {e}")
            raise

    return model


def _create_and_log_shap_importance(model, X_train):
    """
    Compute global SHAP feature importance and log plots to MLflow.

    Samples up to 1000 rows from the training set for performance.
    Logs both a summary (beeswarm) plot and a bar chart.

    Args:
        model: Trained sklearn pipeline with preprocessor and classifier steps
        X_train: Training feature matrix (pre-transformation)
    """
    import shap

    # Sample training data for performance on large datasets
    sample_size = min(1000, len(X_train))
    X_sample = X_train.sample(sample_size, random_state=42)

    # Transform using the fitted preprocessor step
    X_transformed = model.named_steps["preprocessor"].transform(X_sample)

    # Compute SHAP values using the classifier step
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    shap_values = explainer.shap_values(X_transformed)

    # For binary classification, shap_values may be a list — take positive class
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values

    # Collect all feature names across all feature type lists
    all_features = (
        (numeric_features or [])
        + (categorical_features or [])
        + (ordinal_features or [])
        + (flag_features or [])
    )

    artifacts_dir = Path(__file__).parent.parent / "artifacts"

    # --- Plot 1: Summary (beeswarm) ---
    shap.summary_plot(sv, X_transformed, feature_names=all_features, show=False)
    plt.tight_layout()
    summary_path = str(artifacts_dir / "shap_summary.png")
    plt.savefig(summary_path, bbox_inches="tight")
    mlflow.log_artifact(summary_path)
    plt.close()

    # --- Plot 2: Bar chart (mean absolute SHAP) ---
    shap.summary_plot(
        sv,
        X_transformed,
        feature_names=all_features,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    bar_path = str(artifacts_dir / "shap_bar.png")
    plt.savefig(bar_path, bbox_inches="tight")
    mlflow.log_artifact(bar_path)
    plt.close()

    print("SHAP importance plots logged to MLflow")


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Main training pipeline execution."""
    print("\n" + "=" * 60)
    print("PARAMETERS")
    print("=" * 60)
    print(json.dumps(params, indent=2))
    print("=" * 60 + "\n")

    # Load main application data
    print(f"Loading data from: {params['filepath']['source_data']}")
    data_path = Path(__file__).parent.parent / params["filepath"]["source_data"]
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    # Load credit card balance data
    print(f"Loading data from: " f"{params['filepath']['source_data_cc']}")
    cc_path = Path(__file__).parent.parent / params["filepath"]["source_data_cc"]
    cc = pd.read_csv(cc_path)
    print(f"   Shape: {cc.shape}")

    # Prepare features and create train/test split
    print("\nPreparing data...")
    X_train, X_test, y_train, y_test = prepare_data(df, cc)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Create preprocessing pipeline
    print("\nCreating preprocessor...")
    preprocessor = get_preprocessor(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        ordinal_features=ordinal_features,
        flag_features=flag_features,
    )

    # Display feature configuration
    print("Feature configuration:")
    print(f"  Numeric: {numeric_features}")
    print(f"  Categorical: {categorical_features}")
    print(f"  Ordinal: {ordinal_features}")
    print(f"  Flag: {flag_features}")

    # Train model with MLflow tracking
    print("\nTraining model with MLflow...")
    train_with_mlflow(
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        params["model"],
        experiment_name,
    )

    print("\nTraining complete!")


if __name__ == "__main__":
    print(f"Running train.py from {os.getcwd()}")
    main()
