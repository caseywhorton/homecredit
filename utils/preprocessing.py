"""
Data preprocessing module for loan default prediction.

This module provides functions for feature engineering, data preparation,
and preprocessing pipeline creation for machine learning models.
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


# =============================================================================
# Main Data Preparation Functions
# =============================================================================


def prepare_data(
    df: pd.DataFrame,
    cc: pd.DataFrame,
    target: str = "TARGET",
    test_prop: float = 0.20,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare data for model training by adding features and splitting.

    Args:
        df: Main application dataframe (application_train)
        cc: Credit card balance dataframe (credit_card_balance)
        target: Name of target column
        test_prop: Proportion of data to use for test set
            Current fixed to 20% (0.20)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    # Clean the dataframe and add transformed features
    data_with_features = gen_features(df, cc)
    print("columns after join", data_with_features.columns)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split_data(
        data_with_features, target, test_prop
    )

    return X_train, X_test, y_train, y_test


def get_preprocessor(
    numeric_features: Optional[List[str]] = None,
    categorical_features: Optional[List[str]] = None,
    ordinal_features: Optional[List[str]] = None,
    flag_features: Optional[List[str]] = None,
) -> ColumnTransformer:
    """
    Create sklearn preprocessing pipeline for different feature types.

    Args:
        numeric_features: List of numeric column names
        categorical_features: List of categorical column names
        ordinal_features: List of ordinal column names
        flag_features: List of binary flag column names

    Returns:
        ColumnTransformer with appropriate preprocessing steps
    """
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
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )

    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
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
        transformers.append(
            ("num", numeric_transformer, numeric_features)
        )

    if categorical_features is not None and len(categorical_features) > 0:
        transformers.append(
            ("cat", categorical_transformer, categorical_features)
        )

    if ordinal_features is not None and len(ordinal_features) > 0:
        transformers.append(
            ("ord", ordinal_transformer, ordinal_features)
        )

    if flag_features is not None and len(flag_features) > 0:
        transformers.append(
            ("flag", flag_transformer, flag_features)
        )

    # Create the preprocessor with only the relevant transformers
    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def train_test_split_data(
    df: pd.DataFrame, target: str = "TARGET", test_prop: float = 0.20
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split dataframe into training and test sets.

    Args:
        df: Input dataframe with features and target
        target: Target column name
        test_prop: Proportion for test set (0-1)

    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_prop,
        random_state=101,
        stratify=y,
    )

    return X_train, X_test, y_train, y_test


def gen_features(
    df: pd.DataFrame, cc: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate engineered features and clean dataframe.

    Args:
        df: Main application dataframe
        cc: Credit card balance dataframe

    Returns:
        Dataframe with engineered features added
    """
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
        df["CNT_FAM_MEMBERS_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
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
        df["EMPLOYMENT_BKT"]
        .cat.add_categories("Unknown")
        .fillna("Unknown")
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

    # Take the natural log of AMT_GOODS_PRICE
    df["AMT_GOODS_PRICE"] = np.log(df["AMT_GOODS_PRICE"])

    # Create the utilization features
    balance_sum = cc.groupby("SK_ID_CURR")["AMT_BALANCE"].sum()
    credit_limit_sum = (
        cc.groupby("SK_ID_CURR")["AMT_CREDIT_LIMIT_ACTUAL"].sum()
    )

    # Replace 0 with NaN to avoid division by zero
    credit_limit_sum = credit_limit_sum.replace(0, np.nan)

    utilization = balance_sum / credit_limit_sum

    # Convert to DataFrame
    utilization_df = utilization.reset_index()
    utilization_df.columns = ["SK_ID_CURR", "UTILIZATION"]

    # Join with the dataframe
    df = df.merge(utilization_df, on="SK_ID_CURR", how="left")

    # Get max utilization and join
    max_utilization_df = get_max_utilization(cc)
    df = df.merge(max_utilization_df, on="SK_ID_CURR", how="left")

    # Get the months on book and join
    mob_df = get_mob(cc)
    df = df.merge(mob_df, on="SK_ID_CURR", how="left")

    # Get the DPD features and join
    dpd_df = get_dpd(cc)
    df = df.merge(dpd_df, on="SK_ID_CURR", how="left")

    # Get the minimum payment metrics and join
    min_pay_df = get_min_pay(cc)
    df = df.merge(min_pay_df, on="SK_ID_CURR", how="left")

    return df


# =============================================================================
# Feature Engineering Helper Functions
# =============================================================================


def get_mob(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate months on book (MOB) from credit card history.

    Args:
        df: Credit card dataframe with MONTHS_BALANCE column

    Returns:
        Dataframe with SK_ID_CURR, MOB, and MAX_INSTALL columns
    """
    if "MONTHS_BALANCE" not in df.columns:
        print("MONTHS_BALANCE must be in column list")

    df_mob = (
        df.groupby("SK_ID_CURR")["MONTHS_BALANCE"]
        .min()
        .abs()
        .reset_index()
    )

    df_installment = (
        df.groupby("SK_ID_CURR")["CNT_INSTALMENT_MATURE_CUM"]
        .max()
        .abs()
        .reset_index()
    )

    result_df = df_mob.merge(
        df_installment, on="SK_ID_CURR", how="left"
    )

    result_df.columns = ["SK_ID_CURR", "MOB", "MAX_INSTALL"]

    return result_df


def get_max_utilization(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate maximum credit utilization over client history.

    Args:
        df: Credit card dataframe with balance and limit columns

    Returns:
        Dataframe with SK_ID_CURR and MAX_UTILIZATION columns
    """
    utilization_df = df[
        ["SK_ID_CURR", "AMT_BALANCE", "AMT_CREDIT_LIMIT_ACTUAL"]
    ].copy()

    # Replace 0 with NaN to avoid division by zero
    utilization_df["AMT_CREDIT_LIMIT_ACTUAL"] = utilization_df[
        "AMT_CREDIT_LIMIT_ACTUAL"
    ].replace(0, np.nan)

    # Calculate utilization (will be NaN where credit limit is 0)
    utilization_df["UTILIZATION"] = (
        utilization_df["AMT_BALANCE"]
        / utilization_df["AMT_CREDIT_LIMIT_ACTUAL"]
    )

    # Group by client and get max, ignoring NaN values
    max_utilization_df = (
        utilization_df.groupby("SK_ID_CURR")["UTILIZATION"]
        .max()
        .reset_index()
    )

    max_utilization_df.columns = ["SK_ID_CURR", "MAX_UTILIZATION"]

    return max_utilization_df


def get_dpd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate days past due (DPD) indicator features.

    Creates indicators for whether customer has ever been past due,
    past due in last 6 months, and past due in last month.

    Args:
        df: Credit card dataframe with SK_DPD and MONTHS_BALANCE

    Returns:
        Dataframe with SK_ID_CURR and DPD indicator columns
    """
    # Max DPD overall
    max_df = (
        df.groupby("SK_ID_CURR")["SK_DPD"]
        .max()
        .reset_index()
        .fillna(0)
    )
    max_df["MAX_DPD_IND"] = (max_df["SK_DPD"] > 0).astype(int)
    max_df.drop("SK_DPD", axis=1, inplace=True)
    max_df.columns = ["SK_ID_CURR", "MAX_DPD_IND"]

    # Max DPD in last 6 months
    max_6mo_df = (
        df[df["MONTHS_BALANCE"] >= -6]
        .groupby("SK_ID_CURR")["SK_DPD"]
        .max()
        .reset_index()
        .fillna(0)
    )
    max_6mo_df["MAX_DPD_L6M_IND"] = (
        max_6mo_df["SK_DPD"] > 0
    ).astype(int)
    max_6mo_df.drop("SK_DPD", axis=1, inplace=True)
    max_6mo_df.columns = ["SK_ID_CURR", "MAX_DPD_L6M_IND"]

    # Max DPD in most recent month
    max_recent_df = (
        df[df["MONTHS_BALANCE"] == -1]
        .groupby("SK_ID_CURR")["SK_DPD"]
        .max()
        .reset_index()
        .fillna(0)
    )
    max_recent_df["MAX_DPD_L1M_IND"] = (
        max_recent_df["SK_DPD"] > 0
    ).astype(int)
    max_recent_df.drop("SK_DPD", axis=1, inplace=True)
    max_recent_df.columns = ["SK_ID_CURR", "MAX_DPD_L1M_IND"]

    # Merge all DPD features
    result_df = (
        max_df.merge(max_6mo_df, on="SK_ID_CURR", how="outer")
        .merge(max_recent_df, on="SK_ID_CURR", how="outer")
        .fillna(0)
    )

    return result_df


def get_min_pay(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate minimum payment behavior indicators.

    Creates features for whether customer pays only minimum,
    pays less than 2.5% of balance, and average principal to
    balance payment ratio.

    Args:
        df: Credit card dataframe with payment columns

    Returns:
        Dataframe with SK_ID_CURR and payment behavior columns
    """
    df["MIN_PAY_IND"] = (
        (
            df["AMT_PAYMENT_TOTAL_CURRENT"]
            == df["AMT_INST_MIN_REGULARITY"]
        )
        & (df["AMT_INST_MIN_REGULARITY"] > 0)
    ).astype(int)

    df["MIN_PAY_2P5_IND"] = (
        (
            df["AMT_PAYMENT_TOTAL_CURRENT"]
            <= df["AMT_BALANCE"]
        )
        & (df["AMT_INST_MIN_REGULARITY"] > 0)
    ).astype(int)

    result_df = (
        df[["SK_ID_CURR", "MIN_PAY_IND", "MIN_PAY_2P5_IND"]]
        .groupby("SK_ID_CURR")
        .max()
        .reset_index()
    )

    # Calculate average principal to balance payment ratio
    df_prin_pay = (
        df.assign(
            AMT_PRIN_BAL_PAY_AVG=(
                df["AMT_RECEIVABLE_PRINCIPAL"]
                / df["AMT_BALANCE"].replace(0, np.nan)
            )
        )
        .groupby("SK_ID_CURR")["AMT_PRIN_BAL_PAY_AVG"]
        .mean()
        .reset_index()
    ).fillna(1)

    result_df = result_df.merge(
        df_prin_pay, on="SK_ID_CURR", how="left"
    )

    return result_df