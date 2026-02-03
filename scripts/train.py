"""
Model training script with MLflow tracking.

This script loads data, prepares features, trains a classification model,
and logs all experiments to MLflow for tracking and comparison.
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model import create_model, save_model
from utils.preprocessing import get_preprocessor, prepare_data

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


def cross_validate_model(model, X, y, cv=5):
    """
    Perform stratified cross-validation on the model.

    Args:
        model: Sklearn pipeline with preprocessor and classifier
        X: Feature matrix
        y: Target vector
        cv: Number of cross-validation folds

    Returns:
        Dictionary of cross-validation scores
    """
    print(f"\nPerforming {cv}-fold cross-validation...")

    # Use stratified K-fold to maintain class balance
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=101)

    # Define metrics to track
    scoring = {
        "roc_auc": "roc_auc",
        "f1": "f1",
        "precision": "precision",
        "recall": "recall",
        "accuracy": "accuracy",
    }

    # Run cross-validation
    cv_results = cross_validate(
        model,
        X,
        y,
        cv=skf,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1,  # Use all available cores
        verbose=1,
    )

    # Print summary
    print("\nCross-Validation Results:")
    print("-" * 60)
    for metric in ["roc_auc", "f1", "precision", "recall"]:
        train_mean = cv_results[f"train_{metric}"].mean()
        train_std = cv_results[f"train_{metric}"].std()
        test_mean = cv_results[f"test_{metric}"].mean()
        test_std = cv_results[f"test_{metric}"].std()
        print(
            f"  {metric:12s}: "
            f"Train={train_mean:.3f} (±{train_std:.3f}) | "
            f"Test={test_mean:.3f} (±{test_std:.3f})"
        )
    print("-" * 60)

    return cv_results


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
    # Set experiment before starting run
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
                mlflow.log_param(
                    "categorical_features",
                    ",".join(categorical_features),
                )
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
            print(f"Train AUC: {train_auc:.3f}, " f"Test AUC: {test_auc:.3f}")

        except Exception as e:
            print(f"Error in training process: {e}")
            raise

    return model


def _create_and_log_roc_curve(
    y_train, y_train_proba, y_test, y_test_proba, train_auc, test_auc
):
    """
    Create ROC curve visualization and log to MLflow.

    Args:
        y_train: True training labels
        y_train_proba: Predicted probabilities for training set
        y_test: True test labels
        y_test_proba: Predicted probabilities for test set
        train_auc: Training set AUC score
        test_auc: Test set AUC score
    """
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
    plt.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Classifier",
        linewidth=1,
    )
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # Save and log the figure
    roc_path = "/".join((params["filepath"]["model_artifact_dir"], "roc_curve.png"))
    plt.savefig(roc_path)
    mlflow.log_artifact(roc_path)
    plt.close()
    print("ROC curve logged to MLflow")


def _find_optimal_threshold(y_true, y_proba):
    """
    Find optimal classification threshold by maximizing F1 score
    using thresholds from the precision-recall curve.

    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities

    Returns:
        Tuple of (optimal_threshold, best_f1, results_df)
    """
    # Use thresholds from PR curve directly
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision/recall have one extra element vs thresholds
    # trim to match
    precision = precision[:-1]
    recall = recall[:-1]

    # Calculate F1 at each PR curve threshold
    # F1 = 2 * (precision * recall) / (precision + recall)
    denom = precision + recall
    f1_scores = np.where(
        denom > 0,
        2 * (precision * recall) / denom,
        0.0,
    )

    # Build results dataframe
    results_df = pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
        }
    )

    best_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    return optimal_threshold, best_f1, results_df


def _create_and_log_pr_curve(y_test, y_test_proba, optimal_threshold, results_df):
    """
    Create PR curve and threshold analysis plots, log to MLflow.

    Args:
        y_test: True test labels
        y_test_proba: Predicted probabilities for test set
        optimal_threshold: Optimal classification threshold
        results_df: DataFrame of threshold analysis results
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(y_test, y_test_proba)
    ap_score = average_precision_score(y_test, y_test_proba)

    axes[0].plot(recall, precision, linewidth=2, color="tab:blue")
    axes[0].set_xlabel("Recall", fontsize=12)
    axes[0].set_ylabel("Precision", fontsize=12)
    axes[0].set_title(f"Precision-Recall Curve (AP = {ap_score:.3f})", fontsize=14)
    axes[0].grid(alpha=0.3)

    # --- Plot 2: Threshold Analysis ---
    axes[1].plot(
        results_df["threshold"],
        results_df["f1"],
        label="F1",
        linewidth=2,
        color="tab:blue",
    )
    axes[1].plot(
        results_df["threshold"],
        results_df["precision"],
        label="Precision",
        linewidth=2,
        color="tab:orange",
    )
    axes[1].plot(
        results_df["threshold"],
        results_df["recall"],
        label="Recall",
        linewidth=2,
        color="tab:green",
    )
    axes[1].axvline(
        x=optimal_threshold,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal ({optimal_threshold})",
    )
    axes[1].set_xlabel("Threshold", fontsize=12)
    axes[1].set_ylabel("Score", fontsize=12)
    axes[1].set_title("Threshold Analysis", fontsize=14)
    axes[1].legend(loc="center right", fontsize=10)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    # Save and log
    pr_path = "/".join((params["filepath"]["model_artifact_dir"], "pr_curve.png"))
    plt.savefig(pr_path)
    mlflow.log_artifact(str(pr_path))
    plt.close()
    print("PR curve and threshold analysis logged to MLflow")


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
    model = train_with_mlflow(
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
