import os
import sys
from pathlib import Path
import yaml
import pandas as pd
import json

sys.path.append(str(Path(__file__).parent.parent))

from utils.preprocessing import prepare_data, get_preprocessor
from utils.model import create_model, save_model

# preprocesses data and trains a model

# pseudo code
# get data
# preprocess data
# train model
# save model to artifacts folder
# artifacts folder being tracked with dvc

# GET PARAMETERS
with open('../params.yaml', 'r') as f:
    params = yaml.safe_load(f)

numeric_features = params['features']['numeric']


def main():
    print('Parameters')
    print(json.dumps(params, indent=2))
    # read data from source
    print(os.getcwd())
    df = pd.read_csv(params['filepath']['source_data'])

    # extract features and create train/test split
    X_train, X_test, y_train, y_test = prepare_data(df, numeric_features)

    # get the preprocessor from artifacts
    preprocessor = get_preprocessor(numeric_features)

    # create a model
    model = create_model(preprocessor, **params['model'])

    # train the model
    # Fit the model
    print('model.fit')
    model.fit(X_train, y_train)

    # Save the model
    save_model(model, path=params['filepath']['model_sink'])


if __name__ == "__main__":
    print('*'*20)
    print(f'Running train.py from {os.getcwd()}')
    print('*'*20)
    main()
