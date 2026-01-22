from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
    FunctionTransformer,
)
from sklearn.impute import SimpleImputer

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def prepare_data(
    df: pd.DataFrame, cc: pd.DataFrame, target: str = "TARGET", test_prop: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Preprocesses dataframe for model fitting"""

    # clean the dataframe and add transformed features
    data_with_features = gen_features(df, cc)
    print("columns after join", data_with_features.columns)
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split_data(
        data_with_features, target, test_prop
    )

    return X_train, X_test, y_train, y_test  # Fixed typo: y_train_y_test


def get_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    ordinal_features: Optional[List[str]] = None,
    flag_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """Gets the preprocessor artifact using submitted features"""

    # categorical_transformer
    # passthrough features

    # Define transformers
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("to_str", FunctionTransformer(lambda x: x.astype(str))),
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            (
                "ordinal",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    flag_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="UNK"),
            ),
            ("passthrough", FunctionTransformer()),
        ]
    )

    transformers = []

    if numeric_features is not None and len(numeric_features) > 0:
        transformers.append(("num", numeric_transformer, numeric_features))

    if categorical_features is not None and len(categorical_features) > 0:
        transformers.append(("cat", categorical_transformer, categorical_features))

    if ordinal_features is not None and len(ordinal_features) > 0:
        transformers.append(("ord", ordinal_transformer, ordinal_features))

    if flag_features is not None and len(flag_features) > 0:
        transformers.append(("flag", flag_transformer, flag_features))

    # Create the preprocessor with only the relevant transformers
    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def train_test_split_data(
    df: pd.DataFrame, target: str = "TARGET", test_prop: float = 0.20
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
        X,
        y,
        test_size=test_prop,
        random_state=101,
        stratify=y,
        # Fixed: use test_prop parameter
    )

    return X_train, X_test, y_train, y_test


def get_mob(df):
    "Gets the months on book (MOB) from a dataframe using the 'MONTHS_BALANCE' feature"
    if "MONTHS_BALANCE" not in df.columns:
        print("MONTHS_BALANCE must be in column list")

    result = df.groupby("SK_ID_CURR")["MONTHS_BALANCE"].min().abs().reset_index()
    result.columns = ["SK_ID_CURR", "MOB"]

    return result


def get_max_utilization(df):
    "gets a dataframe for max utilization over client history"

    utilization_df = df[["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]].copy()

    # Replace 0 with NaN to avoid division by zero
    utilization_df["AMT_CREDIT_LIMIT_ACTUAL"] = utilization_df[
        "AMT_CREDIT_LIMIT_ACTUAL"
    ].replace(0, np.nan)

    # Calculate utilization (will be NaN where credit limit is 0 or NaN)
    utilization_df["UTILIZATION"] = (
        utilization_df["AMT_BALANCE"] / utilization_df["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    # Group by client and get max, ignoring NaN values
    max_utilization_df = utilization_df.groupby("SK_ID_CURR")["UTILIZATION"].max()

    max_utilization_df = max_utilization_df.reset_index()
    max_utilization_df.columns = ["SK_ID_CURR", "MAX_UTILIZATION"]

    return max_utilization_df


def get_dpd(df):
    "Gets DPD features"

    max_df = df.groupby("SK_ID_CURR")["SK_DPD"].max().reset_index().fillna(0)
    max_df["MAX_DPD_IND"] = (max_df["SK_DPD"] > 0).astype(int)
    max_df.drop("SK_DPD", axis=1, inplace=True)
    max_df.columns = ["SK_ID_CURR", "MAX_DPD_IND"]

    max_6mo_df = (
        df[df["MONTHS_BALANCE"] >= -6]
        .groupby("SK_ID_CURR")["SK_DPD"]
        .max()
        .reset_index()
        .fillna(0)
    )
    max_6mo_df["MAX_DPD_L6M_IND"] = (max_6mo_df["SK_DPD"] > 0).astype(int)
    max_6mo_df.drop("SK_DPD", axis=1, inplace=True)
    max_6mo_df.columns = ["SK_ID_CURR", "MAX_DPD_L6M_IND"]

    max_recent_df = (
        df[df["MONTHS_BALANCE"] == -1]
        .groupby("SK_ID_CURR")["SK_DPD"]
        .max()
        .reset_index()
        .fillna(0)
    )
    max_recent_df["MAX_DPD_L1M_IND"] = (max_recent_df["SK_DPD"] > 0).astype(int)
    max_recent_df.drop("SK_DPD", axis=1, inplace=True)
    max_recent_df.columns = ["SK_ID_CURR", "MAX_DPD_L1M_IND"]

    result_df = (
        max_df.merge(max_6mo_df, on="SK_ID_CURR", how="outer")
        .merge(max_recent_df, on="SK_ID_CURR", how="outer")
        .fillna(0)
    )

    return result_df


def gen_features(df: pd.DataFrame, cc: pd.DataFrame) -> pd.DataFrame:
    """Cleans the dataframe and creates feature engineering columns"""

    # OCCUPATION_TYPE
    df["OCCUPATION_TYPE"] = df.OCCUPATION_TYPE.fillna("Unknown")

    # CNT_FAM_MEMBERS_BKT
    df["CNT_FAM_MEMBERS_BKT"] = pd.cut(
        df["CNT_FAM_MEMBERS"],
        bins=[0, 1, 2, 3, 4, float("inf")],
        labels=["0", "1", "2", "3", "4 or more"],
        right=False,
    )

    df["CNT_FAM_MEMBERS_BKT"] = (
        df["CNT_FAM_MEMBERS_BKT"].cat.add_categories("Unknown").fillna("Unknown")
    )

    # EMPLOYMENT_BKT
    df["YEARS_EMPLOYED"] = abs(df["DAYS_EMPLOYED"]) / 365

    df["EMPLOYMENT_BKT"] = pd.cut(
        df["YEARS_EMPLOYED"],
        bins=[0, 5, 10, float("inf")],
        labels=["0-5 years", "5-10 years", "10+ years"],
        right=False,
    )

    df["EMPLOYMENT_BKT"] = (
        df["EMPLOYMENT_BKT"].cat.add_categories("Unknown").fillna("Unknown")
    )

    # OBS_30_CNT_SOCIAL_CIRCLE_BKT
    df["OBS_30_CNT_SOCIAL_CIRCLE_BKT"] = pd.cut(
        df["OBS_30_CNT_SOCIAL_CIRCLE"],
        bins=[0, 1, float("inf")],
        labels=["0", "1 or more"],
        right=False,
    )

    df["OBS_30_CNT_SOCIAL_CIRCLE_BKT"] = (
        df["OBS_30_CNT_SOCIAL_CIRCLE_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
    )

    # DEF_30_CNT_SOCIAL_CIRCLE_BKT
    df["DEF_30_CNT_SOCIAL_CIRCLE_BKT"] = pd.cut(
        df["DEF_30_CNT_SOCIAL_CIRCLE"],
        bins=[0, 1, float("inf")],
        labels=["0", "1 or more"],
        right=False,
    )

    df["DEF_30_CNT_SOCIAL_CIRCLE_BKT"] = (
        df["DEF_30_CNT_SOCIAL_CIRCLE_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
    )

    # OBS_60_CNT_SOCIAL_CIRCLE_BKT
    df["OBS_60_CNT_SOCIAL_CIRCLE_BKT"] = pd.cut(
        df["OBS_60_CNT_SOCIAL_CIRCLE"],
        bins=[0, 1, float("inf")],
        labels=["0", "1 or more"],
        right=False,
    )

    df["OBS_60_CNT_SOCIAL_CIRCLE_BKT"] = (
        df["OBS_60_CNT_SOCIAL_CIRCLE_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
    )

    # DEF_60_CNT_SOCIAL_CIRCLE_BKT
    df["DEF_60_CNT_SOCIAL_CIRCLE_BKT"] = pd.cut(
        df["DEF_60_CNT_SOCIAL_CIRCLE"],
        bins=[0, 1, float("inf")],
        labels=["0", "1 or more"],
        right=False,
    )

    df["DEF_60_CNT_SOCIAL_CIRCLE_BKT"] = (
        df["DEF_60_CNT_SOCIAL_CIRCLE_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
    )

    # AMT_REQ_CREDIT_BUREAU_YEAR_BKT
    df["AMT_REQ_CREDIT_BUREAU_YEAR_BKT"] = pd.cut(
        df["AMT_REQ_CREDIT_BUREAU_YEAR"],
        bins=[0, 1, float("inf")],
        labels=["0", "1 or more"],
        right=False,
    )

    df["AMT_REQ_CREDIT_BUREAU_YEAR_BKT"] = (
        df["AMT_REQ_CREDIT_BUREAU_YEAR_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
    )

    # Take the natural log of
    df["AMT_GOODS_PRICE"] = np.log(df["AMT_GOODS_PRICE"])

    # create the utilization
    balance_sum = cc.groupby("SK_ID_CURR")["AMT_BALANCE"].sum()
    credit_limit_sum = cc.groupby("SK_ID_CURR")["AMT_CREDIT_LIMIT_ACTUAL"].sum()

    # Replace 0 with NaN to avoid division by zero
    credit_limit_sum = credit_limit_sum.replace(0, np.nan)

    utilization = balance_sum / credit_limit_sum

    # Convert to DataFrame
    utilization_df = utilization.reset_index()
    utilization_df.columns = ["SK_ID_CURR", "UTILIZATION"]

    # join with the dataframe
    df = df.merge(utilization_df, on="SK_ID_CURR", how="left")

    max_utilization_df = get_max_utilization(cc)

    df = df.merge(max_utilization_df, on="SK_ID_CURR", how="left")

    # get the mob and join
    mob_df = get_mob(cc)
    df = df.merge(mob_df, on="SK_ID_CURR", how="left")

    # get the DPD and join
    dpd_df = get_dpd(cc)
    df = df.merge(dpd_df, on="SK_ID_CURR", how="left")

    return df
