from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import yaml
import mlflow
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate


# Load parameters from params.yaml
params_path = Path(__file__).parent.parent / "params.yaml"
with open(params_path, "r") as f:
    params = yaml.safe_load(f)


def evaluate_model(y_true, y_pred, dataset_name=""):
    """Calculate and return metrics"""
    metrics = {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }
    return metrics


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
