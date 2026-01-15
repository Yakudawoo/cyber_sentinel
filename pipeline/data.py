import pandas as pd
from typing import List, Dict

EXPECTED_CATEGORICAL = ["protocol_type", "service", "flag"]
EXPECTED_NUMERIC = ["duration", "src_bytes", "dst_bytes"]
TARGET_COL = "class"

MAPPING = {"anomaly": 1, "normal": 0}


def load_data(path: str = "data/train_data.csv") -> pd.DataFrame:
    """
    Robust data loader for the Cyber Sentinel intrusion dataset.
    
    Improvements included:
    - Schema validation (categorical & numeric columns)
    - Standardized dtypes (strings for categorical, ints/floats for numeric)
    - Explicit handling of missing values
    - Safe target normalization to {0,1}
    """

    df = pd.read_csv(path)
    df.describe()

    # -----------------------------
    # 1. Validate expected columns
    # -----------------------------
    missing_cat = [c for c in EXPECTED_CATEGORICAL if c not in df.columns]
    missing_num = [c for c in EXPECTED_NUMERIC if c not in df.columns]

    if TARGET_COL not in df.columns:
        raise ValueError(f"CSV missing target column '{TARGET_COL}'")

    if missing_cat:
        raise ValueError(f"Missing categorical columns: {missing_cat}")

    if missing_num:
        raise ValueError(f"Missing numeric columns: {missing_num}")

    # ------------------------------------
    # 2. Standardize categorical columns
    # ------------------------------------
    for col in EXPECTED_CATEGORICAL:
        df[col] = df[col].astype("string")

    # ------------------------------------
    # 3. Standardize numeric columns
    # ------------------------------------
    for col in EXPECTED_NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ------------------------------------
    # 4. Handle missing values
    # ------------------------------------
    # Drop rows with missing target (cannot be used)
    df = df.dropna(subset=[TARGET_COL])

    # Replace NA in numeric features with 0 (safe default)
    df[EXPECTED_NUMERIC] = df[EXPECTED_NUMERIC].fillna(0)

    # Replace NA in categoricals with "unknown"
    df[EXPECTED_CATEGORICAL] = df[EXPECTED_CATEGORICAL].fillna("unknown")

    # ------------------------------------
    # 5. Normalize the target to {0,1}
    # ------------------------------------
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map(MAPPING)

    if not set(df[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must be 'normal'/'anomaly' or already 0/1.")

    return df
