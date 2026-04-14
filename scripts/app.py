"""
Flask API for model inference.

Accepts a JSON payload with feature values, returns prediction,
probability, and a per-feature breakdown.
"""

from flask import Flask, request, jsonify, send_from_directory
import sys
from pathlib import Path
from flask_cors import CORS
import pandas as pd
import numpy as np
import yaml

sys.path.append(str(Path(__file__).parent.parent))

from utils.model import load_model

# =============================================================================
# Load Configuration and Model
# =============================================================================

params_path = Path(__file__).parent.parent / "params.yaml"
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

model_path = Path(__file__).parent.parent / "artifacts" / "model_classifier.pkl"
model = load_model(str(model_path))
threshold = params["inference"].get("optimal_threshold", 0.5)

# Pull feature list from params — these are the 7 features the model expects
expected_features = (
    (params["features"].get("numeric") or [])
    + (params["features"].get("categorical") or [])
    + (params["features"].get("ordinal") or [])
    + (params["features"].get("flag") or [])
)

app = Flask(__name__)
CORS(app)
# =============================================================================
# Routes
# =============================================================================


@app.route("/", methods=["GET"])
def index():
    return send_from_directory(Path(__file__).parent, "index.html")


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify(
        {"status": "ok", "threshold": threshold, "features": expected_features}
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run inference on a single row of data.

    Expected JSON payload:
    {
        "FEATURE_1": value,
        "FEATURE_2": value,
        ...
    }

    Returns prediction, probability, and per-feature breakdown.
    """
    data = request.get_json()

    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    # Validate all expected features are present
    missing = [f for f in expected_features if f not in data]
    if missing:
        return (
            jsonify(
                {
                    "error": "Missing required features",
                    "missing_features": missing,
                    "expected_features": expected_features,
                }
            ),
            400,
        )

    # Build input dataframe with only the expected features in the right order
    try:
        X = pd.DataFrame([{f: data[f] for f in expected_features}])
    except Exception as e:
        return jsonify({"error": f"Failed to construct feature matrix: {str(e)}"}), 400

    # Run inference
    try:
        probability = float(model.predict_proba(X)[:, 1][0])
        prediction = int(probability >= threshold)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # Build per-feature breakdown
    feature_breakdown = {
        feature: {"value": data[feature], "type": _get_feature_type(feature)}
        for feature in expected_features
    }

    return jsonify(
        {
            "prediction": prediction,
            "probability": round(probability, 4),
            "threshold": threshold,
            "feature_breakdown": feature_breakdown,
        }
    )


@app.route("/features", methods=["GET"])
def features():
    """Return expected feature names and types."""
    breakdown = {f: _get_feature_type(f) for f in expected_features}
    return jsonify({"expected_features": breakdown})


# =============================================================================
# Helpers
# =============================================================================


def _get_feature_type(feature):
    """Return the feature type category from params."""
    if feature in (params["features"].get("numeric") or []):
        return "numeric"
    if feature in (params["features"].get("categorical") or []):
        return "categorical"
    if feature in (params["features"].get("ordinal") or []):
        return "ordinal"
    if feature in (params["features"].get("flag") or []):
        return "flag"
    return "unknown"


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    print(f"Model loaded from: {model_path}")
    print(f"Threshold: {threshold}")
    print(f"Expected features: {expected_features}")
    app.run(debug=True, port=5001)
