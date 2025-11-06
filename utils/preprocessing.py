from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer


def get_preprocessor(numeric_features):
    "Gets the preprocessor artifact using submitted features"

    # categorical_transformer
    # passthrough features

    # Define transformers
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Combine preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ]
    )

    return preprocessor


def train_test_split_data(df: >>pd.DataFrame, target = "TARGET", test_prop = 0.20):
    """
    Prepares the dataframe by splitting into a train and test
    dataframe
    df:
    target: target column
    test_prop: proportion for test
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101, stratify=y
    )

    return X_train, X_test, y_train, y_test

# TO DO
# CREATE A FUNCTION THAT PREPROCESSES INDIVIDUAL FEATURES