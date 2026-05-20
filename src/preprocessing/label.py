"""Preprocessing utilities for the label column.

Paths:
- Option drop: remove the label column from the dataset
- Option skip: return the input dataframe unchanged


============================================================
LABEL PREPROCESSING
============================================================
Rows used for training target: 8,950
Unique target classes: [0, 1]

Class distribution:
  class 0 (true claim): 3,155 rows (35.25%)
  class 1 (false claim): 5,795 rows (64.75%)

Suggested class_weight for models: {0: 1.4183835182250397, 1: 0.7722174288179465}
X_ml shape: (8950, 9)
y_ml shape: (8950,)


"""

from __future__ import annotations

import pandas as pd


# Choose the default preprocessing path.
# Allowed values: "drop" or "skip".
DEFAULT_LABEL_OPTION = "skip"
DEFAULT_LABEL_SOURCE_COL = "label"


def preprocess_label_option_drop(
    df: pd.DataFrame,
    source_col: str = DEFAULT_LABEL_SOURCE_COL,
) -> pd.DataFrame:
    """Drop option for label (remove the column from the dataset)."""
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()
    result = result.drop(columns=[source_col])
    return result


def preprocess_label_option_skip(
    df: pd.DataFrame,
    source_col: str = DEFAULT_LABEL_SOURCE_COL,
) -> pd.DataFrame:
    """Skip option for label (return the same dataframe object unchanged)."""
    _ = source_col
    return df


def preprocess_label(
    df: pd.DataFrame,
    option: str = DEFAULT_LABEL_OPTION,
    source_col: str = DEFAULT_LABEL_SOURCE_COL,
) -> pd.DataFrame:
    """Dispatch label preprocessing based on the selected option.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    option : str
        "drop" to remove label, or "skip" to leave it unchanged.
    source_col : str
        Column name for the label column.
    """
    option_normalized = str(option).strip().lower()

    if option_normalized == "drop":
        return preprocess_label_option_drop(df=df, source_col=source_col)
    if option_normalized == "skip":
        return preprocess_label_option_skip(df=df, source_col=source_col)

    raise ValueError("option must be 'drop' or 'skip'")


__all__ = [
    "DEFAULT_LABEL_OPTION",
    "DEFAULT_LABEL_SOURCE_COL",
    "preprocess_label_option_drop",
    "preprocess_label_option_skip",
    "preprocess_label",
]
