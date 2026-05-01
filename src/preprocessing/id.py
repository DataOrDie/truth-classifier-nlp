"""Preprocessing utilities for the id column.

Paths:
- Option drop: remove id from the dataset
- Option hash_bucket: create a stable id_bucket feature from id


============================================================
ID QUALITY CHECK
============================================================
Total rows: 8,950
Missing/empty IDs: 0
Duplicate IDs: 0
Unique IDs: 8,950

ML-ready shape (without 'id'): (8950, 9)
Created 'id_bucket' (0-99) for optional stable split strategy.


"""

from __future__ import annotations

import hashlib

import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "drop" or "hash_bucket".
DEFAULT_ID_OPTION = "drop"
DEFAULT_ID_SOURCE_COL = "id"
DEFAULT_ID_BUCKET_COL = "id_bucket"
DEFAULT_ID_N_BUCKETS = 100
DEFAULT_ID_DROP_SOURCE_COL = True


def _normalize_id(series: pd.Series) -> pd.Series:
    """Normalize id values to trimmed pandas string dtype."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.mask(cleaned.fillna("").eq(""), pd.NA)
    return cleaned


def _id_to_bucket(value: str, n_buckets: int = 100) -> int:
    """Map an id value to a stable hash bucket in [0, n_buckets)."""
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest, 16) % n_buckets


def _stable_hash_bucket(series: pd.Series, n_buckets: int = 100) -> pd.Series:
    """Create stable hash buckets for non-missing ids."""
    buckets = series.apply(
        lambda x: pd.NA if pd.isna(x) else _id_to_bucket(str(x), n_buckets=n_buckets)
    )
    return buckets.astype("Int64")


def preprocess_id_option_drop(
    df: pd.DataFrame,
    source_col: str = DEFAULT_ID_SOURCE_COL,
) -> pd.DataFrame:
    """Drop option for id (remove id from the dataset)."""
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()
    result = result.drop(columns=[source_col])
    return result


def preprocess_id_option_hash_bucket(
    df: pd.DataFrame,
    source_col: str = DEFAULT_ID_SOURCE_COL,
    bucket_col: str = DEFAULT_ID_BUCKET_COL,
    n_buckets: int = DEFAULT_ID_N_BUCKETS,
    drop_source_col: bool = DEFAULT_ID_DROP_SOURCE_COL,
) -> pd.DataFrame:
    """Hash-bucket option for id.

    Creates a stable integer bucket feature from id values and, by default,
    drops the original id to avoid leakage.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")
    if n_buckets <= 0:
        raise ValueError("n_buckets must be greater than 0")

    result = df.copy()
    normalized_id = _normalize_id(result[source_col])
    result[source_col] = normalized_id
    result[bucket_col] = _stable_hash_bucket(normalized_id, n_buckets=n_buckets)

    if drop_source_col:
        result = result.drop(columns=[source_col])

    return result


def preprocess_id(
    df: pd.DataFrame,
    option: str = DEFAULT_ID_OPTION,
    source_col: str = DEFAULT_ID_SOURCE_COL,
    bucket_col: str = DEFAULT_ID_BUCKET_COL,
    n_buckets: int = DEFAULT_ID_N_BUCKETS,
    drop_source_col: bool = DEFAULT_ID_DROP_SOURCE_COL,
) -> pd.DataFrame:
    """Dispatch id preprocessing based on selected option.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    option : str
        "drop" to remove id, or "hash_bucket" to create id_bucket.
    source_col : str
        Column name for id.
    bucket_col : str
        Output column name for hash buckets.
    n_buckets : int
        Number of hash buckets.
    drop_source_col : bool
        When option is "hash_bucket", remove the original id column if True.
    """
    option_normalized = str(option).strip().lower()

    if option_normalized == "drop":
        return preprocess_id_option_drop(df=df, source_col=source_col)
    if option_normalized == "hash_bucket":
        return preprocess_id_option_hash_bucket(
            df=df,
            source_col=source_col,
            bucket_col=bucket_col,
            n_buckets=n_buckets,
            drop_source_col=drop_source_col,
        )

    raise ValueError("option must be 'drop' or 'hash_bucket'")


def preprocess_id_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    option: str = DEFAULT_ID_OPTION,
    source_col: str = DEFAULT_ID_SOURCE_COL,
    bucket_col: str = DEFAULT_ID_BUCKET_COL,
    n_buckets: int = DEFAULT_ID_N_BUCKETS,
    drop_source_col: bool = DEFAULT_ID_DROP_SOURCE_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply the same id preprocessing option to train and test datasets."""
    train_out = preprocess_id(
        df=train_df,
        option=option,
        source_col=source_col,
        bucket_col=bucket_col,
        n_buckets=n_buckets,
        drop_source_col=drop_source_col,
    )
    test_out = preprocess_id(
        df=test_df,
        option=option,
        source_col=source_col,
        bucket_col=bucket_col,
        n_buckets=n_buckets,
        drop_source_col=drop_source_col,
    )
    return train_out, test_out