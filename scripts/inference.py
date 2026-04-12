"""
Batch inference script.

Loads a trained model from the artifacts directory and runs predictions
on a preprocessed dataset, outputting results to a CSV file.
"""

import sys
from pathlib import Path

import pandas as pd
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from utils.model import load_model
from utils.preprocessing import prepare_data, get_preprocessor

# =============================================================================
# Load Configuration
# =============================================================================

params_path = Path(__file__).parent.parent / "params.yaml"
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

numeric_features = params["features"].get("numeric")
categorical_features = params["features"].get("categorical")
ordinal_features = params["features"].get("ordinal")
flag_features = params["features"].get("flag")

# =============================================================================
# Inference Functions
# =============================================================================


def load_inference_model():
    """Load trained model from artifacts directory."""
    model_path = Path(__file__).parent.parent / "artifacts" / "model_classifier.pkl"

    if not model_path.exists():
        raise FileNotFoundError(
            f"No model found at {model_path}. "
            "Run train.py first to generate a trained model."
        )

    print(f"Loading model from: {model_path}")
    model = load_model(str(model_path))
    print("Model loaded successfully")
    return model


def preprocess_datasets(*dataframes):
    """
    Merge and preprocess any number of input dataframes into a
    single feature matrix ready for inference.

    Args:
        *dataframes: Any number of raw dataframes to be passed
                     into prepare_data for merging and feature prep.

    Returns:
        Tuple of (X, y) where y is None if no target column exists.
    """
    print("Preprocessing datasets...")

    # Check if target column exists across the provided dataframes
    target_col = "TARGET"
    has_target = any(target_col in df.columns for df in dataframes)

    if has_target:
        X, _, y, _ = prepare_data(*dataframes, test_size=None)
    else:
        # Inject a dummy target so prepare_data doesn't crash,
        # then drop it and return y as None
        df_main = dataframes[0].copy()
        df_main[target_col] = 0
        X, _, _, _ = prepare_data(df_main, *dataframes[1:], test_size=None)
        y = None

    print(f"   Preprocessed shape: {X.shape}")
    return X, y


def run_batch_inference(model, X, y=None):
    """
    Run batch predictions on preprocessed data.

    Args:
        model: Trained sklearn pipeline
        X: Preprocessed feature matrix (output of preprocess_datasets)
        y: Optional true labels for comparison in output CSV

    Returns:
        DataFrame with predictions and probabilities
    """
    threshold = params["inference"].get("optimal_threshold", 0.5)

    print(f"Running inference on {len(X)} records (threshold={threshold})...")
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = X.copy()
    results["prediction"] = predictions
    results["probability"] = probabilities

    if y is not None:
        results["actual"] = y

    return results


def save_results(results):
    """Save predictions to CSV in artifacts directory."""
    output_dir = Path(__file__).parent.parent / "artifacts"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "predictions.csv"
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    return output_path


# =============================================================================
# Main Execution
# =============================================================================

# /Users/caseywhorton/ds/homecredit/data/inference


def main():
    """Main inference pipeline execution."""
    print("\n" + "=" * 60)
    print("BATCH INFERENCE")
    print("=" * 60 + "\n")

    # Load data
    print(f"Loading data from: {params['filepath']['inference_data']}")
    data_path = Path(__file__).parent.parent / params["filepath"]["inference_data"]
    df = pd.read_csv(data_path)
    print(f"   Shape: {df.shape}")

    print(f"Loading data from: {params['filepath']['inference_data_cc']}")
    cc_path = Path(__file__).parent.parent / params["filepath"]["inference_data_cc"]
    cc = pd.read_csv(cc_path)
    print(f"   Shape: {cc.shape}")

    # Load model
    model = load_inference_model()

    # Preprocess — swap in different datasets here without touching run_batch_inference
    X, y = preprocess_datasets(df, cc)

    # Run inference
    results = run_batch_inference(model, X, y)

    # Summary stats
    print(f"\nPrediction summary:")
    print(f"   Total records:      {len(results)}")
    print(
        f"   Predicted positive: {results['prediction'].sum()} ({results['prediction'].mean():.1%})"
    )
    print(f"   Avg probability:    {results['probability'].mean():.3f}")

    # Save results
    save_results(results)
    print("\nInference complete!")


if __name__ == "__main__":
    main()
