# Loan Default Prediction

A machine learning pipeline for predicting loan defaults using credit card balance data and application features. This project implements end-to-end MLOps practices including experiment tracking, data versioning, and reproducible model training.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project builds a classification model to predict loan default risk by analyzing:
- Customer application data
- Credit card balance history
- Payment behavior patterns
- Demographic and financial indicators

The pipeline supports:
- ✅ Parameterized feature engineering
- ✅ Multiple model algorithms (Random Forest, XGBoost, LightGBM)
- ✅ Hyperparameter tuning with cross-validation
- ✅ MLflow experiment tracking
- ✅ Data versioning with DVC
- ✅ Reproducible training workflows

## Features

### Feature Engineering
- Credit utilization metrics (current and historical max)
- Days past due (DPD) indicators across multiple time windows
- Minimum payment behavior analysis
- Months on book (MOB) and installment tracking
- Demographic bucketing and categorical encoding

### Model Pipeline
- Scikit-learn preprocessing pipeline with multiple transformers
- Support for numeric, categorical, ordinal, and flag features
- Stratified train/test splitting
- Cross-validation for robust evaluation
- Comprehensive metrics: AUC, F1, Precision, Recall

### MLOps
- MLflow for experiment tracking and model registry
- DVC for data and model versioning
- YAML-based configuration management
- ROC curve visualization and artifact logging

## Project Structure

```
homecredit/
│
├── home-credit-default-risk/
│   ├── artifacts
│       ├── preprocessor.pkl
|       └── model_classifier.pkl
│   ├── data
│       ├── monitoring_id_set.csv
│       ├── test_id_set.csv
│       ├── train_id_set.csv
│   ├── home-credit-default-risk/
│       ├── application_test.csv
|        ├── application_train.csv
|        ├── credit_card_balance.csv
|        └── ...
│   └── processed/
│       └── .gitkeep
│
├── scripts/
│   └── train.py                 # Main training script
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py         # Feature engineering and data prep
│   └── model.py                 # Model creation and evaluation
│
│
├── mlruns/                      # MLflow experiment logs
│
├── notebooks/                   # Exploratory data analysis
│   └── credit_card.ipynb
│
├── tests/                       # Unit tests
│   ├── test_preprocessing.py
│   └── test_model.py
│
├── params.yaml                  # Model and training configuration
├── requirements.txt             # Python dependencies
├── .gitignore
├── .dvcignore
├── README.md
└── LICENSE
```

## Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/loan-default-prediction.git
   cd loan-default-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull data with DVC** (if using DVC)
   ```bash
   dvc pull
   ```

5. **Set up MLflow tracking** (optional)
   ```bash
   mlflow ui
   # Navigate to http://localhost:5000
   ```

## Usage

### Basic Training

Navigate to the scripts directory and run:

```bash
cd scripts
python train.py
```

This will:
1. Load data from `data/raw/application_train.csv` and `data/raw/credit_card_balance.csv`
2. Apply feature engineering based on `params.yaml` configuration
3. Train the model with specified hyperparameters
4. Log metrics and artifacts to MLflow
5. Save the trained model to `artifacts/model_classifier.pkl`

### Viewing Results

**MLflow UI:**
```bash
mlflow ui
```
Navigate to `http://localhost:5000` to compare experiments, view metrics, and analyze model performance.

**Saved Artifacts:**
- Trained model: `artifacts/model_classifier.pkl`
- ROC curve: `artifacts/roc_curve.png`

### Making Predictions

```python
import pickle
import pandas as pd

# Load trained model
with open('artifacts/model_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# Load new data
new_data = pd.read_csv('path/to/new_data.csv')

# Make predictions
predictions = model.predict(new_data)
probabilities = model.predict_proba(new_data)[:, 1]
```

## Configuration

Model training is controlled via `params.yaml`:

```yaml
experiment_name: "loan_default_experiment"

features:
  numeric:
    - AMT_INCOME_TOTAL
    - AMT_CREDIT
    - UTILIZATION
  categorical:
    - NAME_CONTRACT_TYPE
    - CODE_GENDER
  ordinal:
    - OCCUPATION_TYPE
  flag:
    - FLAG_OWN_CAR
    - FLAG_OWN_REALTY

model:
  n_estimators: 200
  max_depth: 15
  min_samples_split: 5
  min_samples_leaf: 2
  random_state: 42

train:
  test_size: 0.20
  random_state: 101
  cv_folds: 5

filepath:
  source_data: "data/raw/application_train.csv"
  source_data_cc: "data/raw/credit_card_balance.csv"
  model_artifact_dir: "artifacts"
```

### Adding New Features

1. Update `utils/preprocessing.py` with feature engineering logic
2. Add feature names to appropriate lists in `params.yaml`
3. Retrain the model: `python scripts/train.py`

## Model Performance

### Current Results

| Metric | Train | Test |
|--------|-------|------|
| AUC | 0.XXX | 0.XXX |
| F1 Score | 0.XXX | 0.XXX |
| Precision | 0.XXX | 0.XXX |
| Recall | 0.XXX | 0.XXX |

*Update these values with your actual model performance*

### Feature Importance

Top 10 most important features:
1. [Feature name] - [Importance score]
2. [Feature name] - [Importance score]
3. ...

*Add SHAP plots or feature importance visualizations*

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project follows PEP 8 style guidelines. Format code with:

```bash
black .
flake8 .
```

### Adding New Models

1. Define model in `utils/model.py`
2. Update `create_model()` function to support new algorithm
3. Add hyperparameters to `params.yaml`
4. Run training and compare results in MLflow

### Experiment Tracking

All experiments are logged to MLflow with:
- Hyperparameters
- Training/test metrics (AUC, F1, Precision, Recall)
- Feature lists
- Model artifacts
- ROC curves

Compare experiments:
```bash
mlflow ui
# Navigate to http://localhost:5000
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -m 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) (or specify your data source)
- MLflow documentation: https://mlflow.org/docs/latest/index.html
- Scikit-learn documentation: https://scikit-learn.org/

## Contact

**Your Name**  
Email: your.email@example.com  
LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)  
GitHub: [@yourusername](https://github.com/yourusername)

---

*Last updated: [Date]*