"""Subject preprocessing utilities.

This module converts the raw ``subject`` column into a feature-enriched
dataframe. The main entry point is :func:`preprocess_subject`, which accepts a
dataframe plus options and returns a single dataframe with subject-derived
columns added.

The module is designed so you can compare multiple preprocessing combinations
without changing downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import re

import pandas as pd


# Public defaults for the flat-kwargs API (match statement_ds style)
DEFAULT_SUBJECT_SOURCE_COL = "subject"
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_CLEAN_TEXT = True
DEFAULT_NORMALIZE_SEPARATORS = True
DEFAULT_SPLIT_TOPICS = True
DEFAULT_PRIMARY_STRATEGY = "first"
DEFAULT_RARE_THRESHOLD = 10
DEFAULT_RARE_LABEL = "other"
DEFAULT_GROUP_RARE = True
DEFAULT_MAX_TOPICS_FOR_PRIMARY: int | None = None
DEFAULT_MULTI_TOPIC_LABEL = "multi-topic"
DEFAULT_ADD_LENGTH_FEATURES = True
DEFAULT_ADD_TOPIC_LIST = True
DEFAULT_ADD_TOPIC_COUNT = True
DEFAULT_ADD_MULTIPLE_TOPICS_FLAG = True
DEFAULT_ADD_PRIMARY = True
DEFAULT_ADD_GROUPED_PRIMARY = True
DEFAULT_ADD_SUBJECT_FREQUENCY = True
DEFAULT_ADD_SUBJECT_IS_RARE = True
DEFAULT_SUBJECT_LABEL_COL: str | None = None
DEFAULT_VERBOSE = False

DEFAULT_ADD_SUBJECT_PRIMARY_TRUE_RATE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'
# ✅ CORRECT (safe usage of add_subject_primary_true_rate):
# for fold, (train_idx, val_idx) in enumerate(cv.split(df)):
#     df_fold = df.iloc[train_idx].copy()
#     # Compute true-rate only on this fold
#     df_fold_with_rate = preprocess_subject(
#         df_fold, 
#         add_subject_primary_true_rate=True,
#         subject_label_col='label'
#     )

@dataclass
class SubjectDSOptions:
    """Options for the subject preprocessing pipeline (flat-kwargs style)."""

    source_col: str = DEFAULT_SUBJECT_SOURCE_COL
    keep_original: bool = DEFAULT_KEEP_ORIGINAL
    clean_text: bool = DEFAULT_CLEAN_TEXT
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS
    split_topics: bool = DEFAULT_SPLIT_TOPICS
    primary_strategy: str = DEFAULT_PRIMARY_STRATEGY
    rare_threshold: int = DEFAULT_RARE_THRESHOLD
    rare_label: str = DEFAULT_RARE_LABEL
    group_rare: bool = DEFAULT_GROUP_RARE
    max_topics_for_primary: int | None = DEFAULT_MAX_TOPICS_FOR_PRIMARY
    multi_topic_label: str = DEFAULT_MULTI_TOPIC_LABEL
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES
    add_topic_list: bool = DEFAULT_ADD_TOPIC_LIST
    add_topic_count: bool = DEFAULT_ADD_TOPIC_COUNT
    add_multiple_topics_flag: bool = DEFAULT_ADD_MULTIPLE_TOPICS_FLAG
    add_primary: bool = DEFAULT_ADD_PRIMARY
    add_grouped_primary: bool = DEFAULT_ADD_GROUPED_PRIMARY
    add_subject_frequency: bool = DEFAULT_ADD_SUBJECT_FREQUENCY
    add_subject_is_rare: bool = DEFAULT_ADD_SUBJECT_IS_RARE
    add_subject_primary_true_rate: bool = DEFAULT_ADD_SUBJECT_PRIMARY_TRUE_RATE
    subject_label_col: str | None = DEFAULT_SUBJECT_LABEL_COL
    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


SUBJECT_PRESETS: dict[str, SubjectDSOptions] = {
    "minimal": SubjectDSOptions(
        add_length_features=False,
        add_topic_list=False,
        add_grouped_primary=False,
    ),
    "expanded": SubjectDSOptions(),
    "multi_topic": SubjectDSOptions(
        primary_strategy="multi",
        max_topics_for_primary=3,
        multi_topic_label="multi-topic",
    ),
    "rare_safe": SubjectDSOptions(
        primary_strategy="most_frequent",
        group_rare=True,
        rare_threshold=10,
        max_topics_for_primary=3,
        multi_topic_label="multi-topic",
    ),
}


def _scale_col(series: pd.Series, method: str) -> pd.Series:
    """Apply z-score standardization or min-max normalization to a numeric series."""
    s = series.astype(float)
    if method == 'standardize':
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 0 else s - mu
    if method == 'normalize':
        mn, mx = s.min(), s.max()
        rng = mx - mn
        return (s - mn) / rng if rng > 0 else s - mn
    return s


def _clean_subject_text(text: str, normalize_separators: bool = True) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if normalize_separators:
        text = re.sub(r"[\|/;]+", ",", text)
    text = re.sub(r"[^a-z0-9,\s&-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _split_subject_topics(text: str) -> list[str]:
    if not text:
        return []
    parts = re.split(r"[,|/;]+", text)
    return [part.strip() for part in parts if part.strip()]


def _pick_primary_topic(
    topics: list[str],
    strategy: str,
    global_topic_counts: pd.Series,
) -> str:
    if not topics:
        return "unknown"

    strategy_normalized = strategy.strip().lower()
    if strategy_normalized == "first":
        return topics[0]
    if strategy_normalized == "longest":
        return max(topics, key=len)
    if strategy_normalized == "most_frequent":
        ranked_topics = sorted(
            topics,
            key=lambda topic: (
                int(global_topic_counts.get(topic, 0)),
                len(topic),
                -topics.index(topic),
            ),
            reverse=True,
        )
        return ranked_topics[0]
    if strategy_normalized == "multi":
        return "multi-topic"

    raise ValueError(
        "primary_strategy must be one of: 'first', 'longest', 'most_frequent', 'multi'"
    )


def preprocess_subject(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SUBJECT_SOURCE_COL,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    clean_text: bool = DEFAULT_CLEAN_TEXT,
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS,
    split_topics: bool = DEFAULT_SPLIT_TOPICS,
    primary_strategy: str = DEFAULT_PRIMARY_STRATEGY,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    rare_label: str = DEFAULT_RARE_LABEL,
    group_rare: bool = DEFAULT_GROUP_RARE,
    max_topics_for_primary: int | None = DEFAULT_MAX_TOPICS_FOR_PRIMARY,
    multi_topic_label: str = DEFAULT_MULTI_TOPIC_LABEL,
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES,
    add_topic_list: bool = DEFAULT_ADD_TOPIC_LIST,
    add_topic_count: bool = DEFAULT_ADD_TOPIC_COUNT,
    add_multiple_topics_flag: bool = DEFAULT_ADD_MULTIPLE_TOPICS_FLAG,
    add_primary: bool = DEFAULT_ADD_PRIMARY,
    add_grouped_primary: bool = DEFAULT_ADD_GROUPED_PRIMARY,
    add_subject_frequency: bool = DEFAULT_ADD_SUBJECT_FREQUENCY,
    add_subject_is_rare: bool = DEFAULT_ADD_SUBJECT_IS_RARE,
    add_subject_primary_true_rate: bool = DEFAULT_ADD_SUBJECT_PRIMARY_TRUE_RATE,
    subject_label_col: str | None = DEFAULT_SUBJECT_LABEL_COL,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess the subject column and return a single enriched dataframe.

    This function uses a flat keyword-argument API consistent with
    `preprocess_statement_ds` so callers can pass explicit flags.

    WARNING: If add_subject_primary_true_rate=True, the resulting feature will
    contain information from the label column. To avoid leakage during training,
    this feature MUST be computed within cross-validation folds, not on the
    full training set before splitting.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    # flat-kwargs are used directly (no config object)

    result = df.copy()
    _scalable: list[str] = []
    subject_raw = result[source_col].fillna("").astype(str)

    if clean_text:
        subject_clean = subject_raw.apply(
            lambda value: _clean_subject_text(value, normalize_separators=normalize_separators)
        )
    else:
        subject_clean = subject_raw.str.strip()

    if split_topics:
        subject_topics = subject_clean.apply(_split_subject_topics)
    else:
        subject_topics = subject_clean.apply(lambda value: [value] if value else [])

    exploded_topics = subject_topics.explode().dropna()
    global_topic_counts = (
        exploded_topics.value_counts() if not exploded_topics.empty else pd.Series(dtype=int)
    )

    topic_count = subject_topics.apply(len)

    if add_topic_list:
        result[f"{source_col}_topics"] = subject_topics

    if add_topic_count:
        result[f"{source_col}_topic_count"] = topic_count
        _scalable.append(f"{source_col}_topic_count")

    if add_multiple_topics_flag:
        result[f"{source_col}_has_multiple_topics"] = topic_count.gt(1).astype(int)

    if add_length_features:
        result[f"{source_col}_length"] = subject_clean.str.len()
        result[f"{source_col}_token_count"] = subject_clean.apply(lambda value: len(value.split()))
        _scalable += [f"{source_col}_length", f"{source_col}_token_count"]

    if add_primary:
        primary_subject = subject_topics.apply(
            lambda topics: _pick_primary_topic(topics, primary_strategy, global_topic_counts)
        )

        if max_topics_for_primary is not None:
            primary_subject = primary_subject.where(topic_count <= int(max_topics_for_primary), multi_topic_label)

        result[f"{source_col}_primary"] = primary_subject

        # Compute frequency for each primary subject
        if add_subject_frequency:
            subject_value_counts = result[f"{source_col}_primary"].value_counts()
            result[f"{source_col}_frequency"] = result[f"{source_col}_primary"].map(subject_value_counts).fillna(0).astype(int)
            _scalable.append(f"{source_col}_frequency")

        # Compute is_rare flag based on threshold
        if add_subject_is_rare:
            if add_subject_frequency:
                result[f"{source_col}_is_rare"] = (result[f"{source_col}_frequency"] < int(rare_threshold)).astype(int)
            else:
                # Fallback: compute frequency inline if not already done
                subject_value_counts = result[f"{source_col}_primary"].value_counts()
                result[f"{source_col}_is_rare"] = (
                    result[f"{source_col}_primary"].map(subject_value_counts).fillna(0) < int(rare_threshold)
                ).astype(int)

        # Compute true-rate for each primary subject
        # WARNING: This feature contains label information and must be computed within CV folds to avoid leakage
        if add_subject_primary_true_rate:
            if subject_label_col is None:
                raise ValueError(
                    "add_subject_primary_true_rate=True requires subject_label_col to be specified. "
                    "Set subject_label_col to the name of the label column (usually 'label')."
                )
            if subject_label_col not in result.columns:
                raise KeyError(
                    f"Label column '{subject_label_col}' not found in DataFrame. "
                    f"Available columns: {list(result.columns[:10])}..."
                )
            # Compute empirical true-rate (mean of label) for each subject_primary
            subject_true_rates = result.groupby(f"{source_col}_primary")[subject_label_col].mean()
            result[f"{source_col}_primary_true_rate"] = result[f"{source_col}_primary"].map(subject_true_rates).fillna(0.5)

        if group_rare and add_grouped_primary:
            primary_value_counts = result[f"{source_col}_primary"].value_counts()
            rare_subjects = primary_value_counts[primary_value_counts < int(rare_threshold)].index
            result[f"{source_col}_primary_grouped"] = result[f"{source_col}_primary"].where(
                ~result[f"{source_col}_primary"].isin(rare_subjects), rare_label
            )
    elif group_rare:
        raise ValueError("group_rare=True requires add_primary=True so a primary subject can be grouped")

    if scale != 'none':
        for col in _scalable:
            result[col] = _scale_col(result[col], scale)

    result[f"{source_col}_clean"] = subject_clean

    if keep_original:
        result[source_col] = subject_raw
    else:
        result = result.drop(columns=[source_col])

    return result


def preprocess_subject_option_minimal(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SUBJECT_SOURCE_COL,
) -> pd.DataFrame:
    opts = SUBJECT_PRESETS["minimal"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_subject(df=df, **args)


def preprocess_subject_option_expanded(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SUBJECT_SOURCE_COL,
) -> pd.DataFrame:
    opts = SUBJECT_PRESETS["expanded"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_subject(df=df, **args)


def preprocess_subject_option_multi_topic(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SUBJECT_SOURCE_COL,
) -> pd.DataFrame:
    opts = SUBJECT_PRESETS["multi_topic"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_subject(df=df, **args)


def preprocess_subject_option_rare_safe(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SUBJECT_SOURCE_COL,
) -> pd.DataFrame:
    opts = SUBJECT_PRESETS["rare_safe"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_subject(df=df, **args)


__all__ = [
    "DEFAULT_SUBJECT_SOURCE_COL",
    "DEFAULT_SCALE",
    "SubjectDSOptions",
    "SUBJECT_PRESETS",
    "preprocess_subject",
    "preprocess_subject_option_expanded",
    "preprocess_subject_option_minimal",
    "preprocess_subject_option_multi_topic",
    "preprocess_subject_option_rare_safe",
]