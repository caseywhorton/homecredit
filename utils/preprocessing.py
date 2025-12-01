from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from typing import Tuple, List


def prepare_data(
    df: pd.DataFrame,
    numeric_features: List[str],
    target: str = "TARGET",
    test_prop: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesses dataframe for model fitting"""

    # clean the dataframe and add transformed features
    data_with_features = gen_features(df)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split_data(data_with_features, target, test_prop)

    return X_train, X_test, y_train, y_test  # Fixed typo: y_train_y_test


def get_preprocessor(numeric_features: List[str]) -> ColumnTransformer:
    """Gets the preprocessor artifact using submitted features"""

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


def train_test_split_data(
    df: pd.DataFrame, 
    target: str = "TARGET", 
    test_prop: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepares the dataframe by splitting into a train and test dataframe
    
    Args:
        df: Input dataframe
        target: Target column name
        test_prop: Proportion for test set
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """

    X = df.drop(columns=[target])  # Fixed: was target_col
    y = df[target]  # Fixed: was target_col

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_prop, random_state=101, stratify=y  # Fixed: use test_prop parameter
    )

    return X_train, X_test, y_train, y_test


def gen_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe and creates feature engineering columns"""
    
    # OCCUPATION_TYPE
    df['OCCUPATION_TYPE'] = df.OCCUPATION_TYPE.fillna("Unknown")

    # CNT_FAM_MEMBERS_BKT
    df['CNT_FAM_MEMBERS_BKT'] = pd.cut(
        df['CNT_FAM_MEMBERS'],
        bins=[0, 1, 2, 3, 4, float('inf')],
        labels=['0', '1', '2', '3', '4 or more'],
        right=False
    )

    df['CNT_FAM_MEMBERS_BKT'] = df['CNT_FAM_MEMBERS_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # EMPLOYMENT_BKT
    df['YEARS_EMPLOYED'] = abs(df['DAYS_EMPLOYED']) / 365

    df['EMPLOYMENT_BKT'] = pd.cut(
        df['YEARS_EMPLOYED'],
        bins=[0, 0.5, 1, 2, 5, 10, 20, 30, float('inf')],
        labels=[
            '< 0.5 years',
            '0.5 - 1 year',
            '1 - 2 years',
            '2 - 5 years',
            '5 - 10 years',
            '10 - 20 years',
            '20 - 30 years',
            '> 30 years'
        ],
        right=False
    )
    df['EMPLOYMENT_BKT'] = df['EMPLOYMENT_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # OBS_30_CNT_SOCIAL_CIRCLE_BKT
    df['OBS_30_CNT_SOCIAL_CIRCLE_BKT'] = pd.cut(
        df['OBS_30_CNT_SOCIAL_CIRCLE'],
        bins=[0, 1, float('inf')],
        labels=['0', '1 or more'],
        right=False
    )

    df['OBS_30_CNT_SOCIAL_CIRCLE_BKT'] = df['OBS_30_CNT_SOCIAL_CIRCLE_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # DEF_30_CNT_SOCIAL_CIRCLE_BKT
    df['DEF_30_CNT_SOCIAL_CIRCLE_BKT'] = pd.cut(
        df['DEF_30_CNT_SOCIAL_CIRCLE'],
        bins=[0, 1, float('inf')],
        labels=['0', '1 or more'],
        right=False
    )

    df['DEF_30_CNT_SOCIAL_CIRCLE_BKT'] = df['DEF_30_CNT_SOCIAL_CIRCLE_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # OBS_60_CNT_SOCIAL_CIRCLE_BKT
    df['OBS_60_CNT_SOCIAL_CIRCLE_BKT'] = pd.cut(
        df['OBS_60_CNT_SOCIAL_CIRCLE'],
        bins=[0, 1, float('inf')],
        labels=['0', '1 or more'],
        right=False
    )

    df['OBS_60_CNT_SOCIAL_CIRCLE_BKT'] = df['OBS_60_CNT_SOCIAL_CIRCLE_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # DEF_60_CNT_SOCIAL_CIRCLE_BKT
    df['DEF_60_CNT_SOCIAL_CIRCLE_BKT'] = pd.cut(
        df['DEF_60_CNT_SOCIAL_CIRCLE'],
        bins=[0, 1, float('inf')],
        labels=['0', '1 or more'],
        right=False
    )

    df['DEF_60_CNT_SOCIAL_CIRCLE_BKT'] = df['DEF_60_CNT_SOCIAL_CIRCLE_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # AMT_REQ_CREDIT_BUREAU_YEAR_BKT
    df['AMT_REQ_CREDIT_BUREAU_YEAR_BKT'] = pd.cut(
        df['AMT_REQ_CREDIT_BUREAU_YEAR'],
        bins=[0, 1, float('inf')],
        labels=['0', '1 or more'],
        right=False
    )

    df['AMT_REQ_CREDIT_BUREAU_YEAR_BKT'] = df['AMT_REQ_CREDIT_BUREAU_YEAR_BKT'].cat.add_categories('Unknown').fillna('Unknown')

    # Take the natural log of 
    df['AMT_GOODS_PRICE'] = np.log(df['AMT_GOODS_PRICE'])

    return df