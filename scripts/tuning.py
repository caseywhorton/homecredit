"""
Hyperparameter tuning script using RandomizedSearchCV.

This script performs randomized search over hyperparameter space
to find optimal model configuration, logging all trials to MLflow.
"""

import json
import os
import sys
from pathlib import Path
import warnings
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from utils.model import create_model
from utils.preprocessing import get_preprocessor, prepare_data


# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'

# Suppress sklearn convergence warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Suppress MLflow warnings
import logging
logging.getLogger('mlflow').setLevel(logging.ERROR)

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

experiment_name = params.get(
    "tuning_experiment_name", "hyperparameter_tuning"
)


# =============================================================================
# Hyperparameter Search Space
# =============================================================================


def get_param_distributions(model_type="random_forest"):
    """
    Get parameter distributions for randomized search.

    Args:
        model_type: Type of model ('random_forest', 'xgboost', 
                    'lightgbm')

    Returns:
        Dictionary of parameter distributions
    """
    if model_type == "random_forest":
        return {
            "classifier__n_estimators": [50, 100, 200, 300, 500],
            "classifier__max_depth": [5, 10, 15, 20, 25, 30, None],
            "classifier__min_samples_split": [2, 5, 10, 15, 20],
            "classifier__min_samples_leaf": [1, 2, 4, 8, 10],
            "classifier__max_features": ["sqrt", "log2", None],
            "classifier__bootstrap": [True, False],
            "classifier__class_weight": ["balanced", None],
        }
    elif model_type == "xgboost":
        return {
            "classifier__n_estimators": [50, 100, 200, 300, 500],
            "classifier__max_depth": [3, 5, 7, 9, 11],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_weight": [1, 3, 5, 7],
            "classifier__gamma": [0, 0.1, 0.2, 0.3, 0.4],
        }
    elif model_type == "lightgbm":
        return {
            "classifier__n_estimators": [50, 100, 200, 300, 500],
            "classifier__max_depth": [3, 5, 7, 9, 11, -1],
            "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "classifier__num_leaves": [15, 31, 63, 127],
            "classifier__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "classifier__min_child_samples": [5, 10, 20, 30],
            "classifier__reg_alpha": [0, 0.1, 0.5, 1.0],
            "classifier__reg_lambda": [0, 0.1, 0.5, 1.0],
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# =============================================================================
# Tuning Functions
# =============================================================================


def tune_hyperparameters(
    X_train,
    y_train,
    preprocessor,
    base_model_params,
    n_iter=50,
    cv=5,
):
    """
    Perform randomized hyperparameter search with MLflow tracking.

    Args:
        X_train: Training feature matrix
        y_train: Training target vector
        preprocessor: Sklearn preprocessing pipeline
        base_model_params: Base model parameters
        n_iter: Number of parameter combinations to try
        cv: Number of cross-validation folds

    Returns:
        Dictionary containing best parameters and results
    """
    print("\n" + "=" * 60)
    print("HYPERPARAMETER TUNING")
    print("=" * 60)

    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Get parameter distributions
    model_type = params["tuning"].get("model_type", "random_forest")
    param_distributions = get_param_distributions(model_type)

    print(f"\nModel type: {model_type}")
    print(f"Search iterations: {n_iter}")
    print(f"CV folds: {cv}")
    print(f"Scoring metric: roc_auc")
    print("\nParameter search space:")
    for param, values in param_distributions.items():
        print(f"  {param}: {values}")

    # Create base model with appropriate algorithm
    # Override base_model_params to use correct model type
    tuning_model_params = {
        "random_state": base_model_params.get("random_state", 42)
    }
    
    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        
        base_model = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(**tuning_model_params))
        ])
    elif model_type == "xgboost":
        try:
            from xgboost import XGBClassifier
            from sklearn.pipeline import Pipeline
            
            base_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', XGBClassifier(**tuning_model_params))
            ])
        except ImportError:
            raise ImportError(
                "XGBoost not installed. "
                "Install it with: pip install xgboost"
            )
    elif model_type == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
            from sklearn.pipeline import Pipeline
            
            # LightGBM needs verbose=-1 to suppress warnings
            tuning_model_params['verbose'] = -1
            base_model = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', LGBMClassifier(**tuning_model_params))
            ])
        except ImportError:
            raise ImportError(
                "LightGBM not installed. "
                "Install it with: pip install lightgbm"
            )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Set up stratified K-fold
    skf = StratifiedKFold(
        n_splits=cv, shuffle=True, random_state=101
    )

    # Create randomized search
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=skf,
        scoring="roc_auc",
        n_jobs=-1,
        verbose=2,
        random_state=42,
        return_train_score=True,
    )

    # Start MLflow parent run
    with mlflow.start_run(run_name="random_search_parent"):
        # Log search configuration
        mlflow.log_param("tuning_method", "RandomizedSearchCV")
        mlflow.log_param("n_iter", n_iter)
        mlflow.log_param("cv_folds", cv)
        mlflow.log_param("scoring_metric", "roc_auc")
        mlflow.log_param("model_type", model_type)

        # Perform search
        print("\n" + "=" * 60)
        print("Starting randomized search...")
        print("=" * 60 + "\n")

        random_search.fit(X_train, y_train)

        # Log best results
        print("\n" + "=" * 60)
        print("BEST RESULTS")
        print("=" * 60)
        print(f"Best CV AUC: {random_search.best_score_:.4f}")
        print(f"\nBest parameters:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
            mlflow.log_param(f"best_{param}", value)

        mlflow.log_metric("best_cv_auc", random_search.best_score_)

        # Log top 10 parameter combinations
        results_df = pd.DataFrame(random_search.cv_results_)
        results_df = results_df.sort_values(
            "mean_test_score", ascending=False
        )

        print("\nTop 10 parameter combinations:")
        print("-" * 60)
        for idx, row in results_df.head(10).iterrows():
            print(
                f"Rank {row['rank_test_score']:2d}: "
                f"AUC={row['mean_test_score']:.4f} "
                f"(±{row['std_test_score']:.4f})"
            )

        # Save results to CSV
        results_path = (
            Path(__file__).parent.parent
            / "artifacts"
            / "tuning_results.csv"
        )
        results_df.to_csv(results_path, index=False)
        mlflow.log_artifact(str(results_path))

        print(f"\nFull results saved to: {results_path}")
        print("=" * 60)

    return {
        "best_params": random_search.best_params_,
        "best_score": random_search.best_score_,
        "cv_results": random_search.cv_results_,
        "best_estimator": random_search.best_estimator_,
    }


# =============================================================================
# Main Execution
# =============================================================================


def main():
    """Main tuning pipeline execution."""
    print("\n" + "*" * 60)
    print(f"Running tune_hyperparameters.py from {os.getcwd()}")
    print("*" * 60)

    print("\n" + "=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(json.dumps(params.get("tuning", {}), indent=2))
    print("=" * 60 + "\n")

    # Load main application data
    print(
        f"Loading data from: {params['filepath']['source_data']}"
    )
    data_path = (
        Path(__file__).parent.parent
        / params["filepath"]["source_data"]
    )
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    # Load credit card balance data
    print(
        f"Loading data from: "
        f"{params['filepath']['source_data_cc']}"
    )
    cc_path = (
        Path(__file__).parent.parent
        / params["filepath"]["source_data_cc"]
    )
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

    # Get tuning configuration
    tuning_config = params.get("tuning", {})
    n_iter = tuning_config.get("n_iter", 50)
    cv_folds = tuning_config.get("cv_folds", 5)

    # Perform hyperparameter tuning
    results = tune_hyperparameters(
        X_train,
        y_train,
        preprocessor,
        params["model"],
        n_iter=n_iter,
        cv=cv_folds,
    )

    # Print recommendation
    print("\n" + "=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    print(
        "Update params.yaml with best parameters and retrain:\n"
    )
    print("model:")
    for param, value in results["best_params"].items():
        # Remove 'classifier__' prefix
        param_name = param.replace("classifier__", "")
        print(f"  {param_name}: {value}")
    print("=" * 60)

    print("\nHyperparameter tuning complete!")
    print(
        f"Best CV AUC: {results['best_score']:.4f}\n"
    )


if __name__ == "__main__":
    main()