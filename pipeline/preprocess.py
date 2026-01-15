import pandas as pd
from typing import Tuple
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
import numpy as np


CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "class"


def _bucket_rare_categories(df: pd.DataFrame, col: str, threshold: float = 0.01):
    freq = df[col].value_counts(normalize=True)
    rare = freq[freq < threshold].index
    df[col] = df[col].replace(rare, "rare")
    return df


def _feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        df["bytes_total"] = df["src_bytes"] + df["dst_bytes"]
        df["bytes_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1)
    return df


def preprocess(df: pd.DataFrame) -> Tuple:
    df = df.copy()

    # ------------------------------------------------
    # 1. Ensure ALL categoricals are strings
    # ------------------------------------------------
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # ------------------------------------------------
    # 2. Feature engineering
    # ------------------------------------------------
    df = _feature_engineering(df)

    # ------------------------------------------------
    # 3. Rare category bucketing
    # ------------------------------------------------
    for col in CATEGORICAL:
        if col in df.columns:
            df = _bucket_rare_categories(df, col)

    # ------------------------------------------------
    # 4. Separate target
    # ------------------------------------------------
    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # ------------------------------------------------
    # 5. Train/test split
    # ------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )

    # ------------------------------------------------
    # 6. Preprocessing pipeline
    # ------------------------------------------------
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_cols),
            ("cat", categorical_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3
    )

    return X_train, X_test, y_train, y_test, preproc
