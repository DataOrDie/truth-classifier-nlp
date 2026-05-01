"""Speaker-job preprocessing utilities.

This module mirrors the API and behavior of ``src.preprocessing.speaker`` so
callers can toggle preprocessing options via flat kwargs or presets. The main
entry point is :func:`preprocess_speaker_job` which returns a dataframe with
additional speaker-job features.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import pandas as pd


# Public defaults for the flat-kwargs API.
DEFAULT_SOURCE_COL = "speaker_job"
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_CLEAN_TEXT = True
DEFAULT_NORMALIZE_SEPARATORS = True
DEFAULT_GROUP_RARE = True
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_RARE_LABEL = "other"
DEFAULT_ADD_LENGTH_FEATURES = True
DEFAULT_ADD_FREQUENCY = True
DEFAULT_ADD_IS_RARE = True
DEFAULT_ADD_GROUPED_JOB = True
DEFAULT_ADD_TITLE_FLAG = True
DEFAULT_ADD_COMMA_FLAG = True
DEFAULT_ADD_SLASH_FLAG = True
DEFAULT_ADD_AMPERSAND_FLAG = True
DEFAULT_ADD_TOKEN_COUNT = True
DEFAULT_ADD_JOB_PRIMARY_TRUE_RATE = False
DEFAULT_JOB_LABEL_COL: Optional[str] = None
DEFAULT_VERBOSE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'


@dataclass
class SpeakerJobDSOptions:
    source_col: str = DEFAULT_SOURCE_COL
    keep_original: bool = DEFAULT_KEEP_ORIGINAL
    clean_text: bool = DEFAULT_CLEAN_TEXT
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS
    group_rare: bool = DEFAULT_GROUP_RARE
    rare_threshold: int = DEFAULT_RARE_THRESHOLD
    rare_label: str = DEFAULT_RARE_LABEL
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES
    add_frequency: bool = DEFAULT_ADD_FREQUENCY
    add_is_rare: bool = DEFAULT_ADD_IS_RARE
    add_grouped_job: bool = DEFAULT_ADD_GROUPED_JOB
    add_title_flag: bool = DEFAULT_ADD_TITLE_FLAG
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG
    add_slash_flag: bool = DEFAULT_ADD_SLASH_FLAG
    add_ampersand_flag: bool = DEFAULT_ADD_AMPERSAND_FLAG
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT
    add_job_primary_true_rate: bool = DEFAULT_ADD_JOB_PRIMARY_TRUE_RATE
    job_label_col: Optional[str] = DEFAULT_JOB_LABEL_COL
    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


PRESETS: dict[str, SpeakerJobDSOptions] = {
    "minimal": SpeakerJobDSOptions(
        add_grouped_job=False,
        add_job_primary_true_rate=False,
        add_title_flag=False,
        add_comma_flag=False,
        add_slash_flag=False,
        add_ampersand_flag=False,
    ),
    "expanded": SpeakerJobDSOptions(),
    "rare_safe": SpeakerJobDSOptions(group_rare=True, rare_threshold=5, add_job_primary_true_rate=False),
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


def _clean_speaker_job(text: str, normalize_separators: bool = True) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if normalize_separators:
        text = re.sub(r"[\|/;,&]+", " ", text)
    text = re.sub(r"[^a-z0-9\s'.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_speaker_job(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SOURCE_COL,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    clean_text: bool = DEFAULT_CLEAN_TEXT,
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS,
    group_rare: bool = DEFAULT_GROUP_RARE,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    rare_label: str = DEFAULT_RARE_LABEL,
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES,
    add_frequency: bool = DEFAULT_ADD_FREQUENCY,
    add_is_rare: bool = DEFAULT_ADD_IS_RARE,
    add_grouped_job: bool = DEFAULT_ADD_GROUPED_JOB,
    add_title_flag: bool = DEFAULT_ADD_TITLE_FLAG,
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG,
    add_slash_flag: bool = DEFAULT_ADD_SLASH_FLAG,
    add_ampersand_flag: bool = DEFAULT_ADD_AMPERSAND_FLAG,
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT,
    add_job_primary_true_rate: bool = DEFAULT_ADD_JOB_PRIMARY_TRUE_RATE,
    job_label_col: Optional[str] = DEFAULT_JOB_LABEL_COL,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess the `speaker_job` column and return an enriched dataframe.

    Notes:
    - If ``add_job_primary_true_rate`` is True, this must be computed inside
      cross-validation folds to avoid leakage.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()
    job_raw = result[source_col].fillna("").astype(str)
    _scalable: list[str] = []

    if clean_text:
        job_clean = job_raw.apply(lambda v: _clean_speaker_job(v, normalize_separators=normalize_separators))
    else:
        job_clean = job_raw.str.strip().str.lower()

    job_clean = job_clean.replace("", "unknown")
    job_counts = job_clean.value_counts()
    job_frequency = job_clean.map(job_counts).fillna(0).astype(int)

    if add_frequency:
        result[f"{source_col}_frequency"] = job_frequency
        result[f"{source_col}_frequency_pct"] = job_frequency / max(len(result), 1)
        _scalable += [f"{source_col}_frequency", f"{source_col}_frequency_pct"]

    if add_is_rare:
        result[f"{source_col}_is_rare"] = job_frequency.lt(int(rare_threshold)).astype(int)

    if add_grouped_job:
        rare_jobs = job_counts[job_counts < int(rare_threshold)].index
        result[f"{source_col}_grouped"] = job_clean.where(~job_clean.isin(rare_jobs), rare_label)

    if add_length_features:
        result[f"{source_col}_char_len"] = job_clean.str.len()
        result[f"{source_col}_token_count"] = job_clean.apply(lambda value: len(value.split()))
        _scalable += [f"{source_col}_char_len", f"{source_col}_token_count"]

    if add_title_flag:
        result[f"{source_col}_has_title"] = job_raw.str.contains(
            r"\b(ceo|cfo|cto|professor|doctor|dr|senator|judge|mayor|governor|attorney|lawyer|president|executive|director)\b",
            case=False,
            regex=True,
        ).fillna(False).astype(int)

    if add_comma_flag:
        result[f"{source_col}_has_comma"] = job_raw.str.contains(",", regex=False).fillna(False).astype(int)

    if add_slash_flag:
        result[f"{source_col}_has_slash"] = job_raw.str.contains("/", regex=False).fillna(False).astype(int)

    if add_ampersand_flag:
        result[f"{source_col}_has_ampersand"] = job_raw.str.contains("&", regex=False).fillna(False).astype(int)

    if add_job_primary_true_rate:
        if job_label_col is None:
            raise ValueError(
                "add_job_primary_true_rate=True requires job_label_col to be specified. "
                "Set job_label_col to the name of the label column (usually 'label')."
            )
        if job_label_col not in result.columns:
            raise KeyError(f"Label column '{job_label_col}' not found in DataFrame.")
        job_true_rates = result.groupby(job_clean)[job_label_col].mean()
        result[f"{source_col}_primary_true_rate"] = job_clean.map(job_true_rates).fillna(0.5)

    if scale != 'none':
        for col in _scalable:
            result[col] = _scale_col(result[col], scale)

    result[f"{source_col}_clean"] = job_clean

    if keep_original:
        result[source_col] = job_raw
    else:
        result = result.drop(columns=[source_col])

    if group_rare and not add_grouped_job:
        raise ValueError("group_rare=True requires add_grouped_job=True so rare jobs can be grouped")

    if verbose:
        unknown_jobs = (job_clean == "unknown").sum()
        print("Total rows:", len(result))
        print(f"Unknown/empty {source_col} values: {unknown_jobs:,}")
        print(f"Unique {source_col} values: {job_clean.nunique():,}")
        print(f"Rare job threshold: < {rare_threshold} occurrences -> grouped as '{rare_label}'")
        print("\nTop 10 jobs by frequency:")
        print(job_counts.head(10).to_string())
        print("\nSample before/after:")
        sample_job = pd.DataFrame({
            f"{source_col}_raw": job_raw.head(5),
            f"{source_col}_clean": job_clean.head(5),
            f"{source_col}_grouped": result.get(f"{source_col}_grouped", pd.Series(dtype=str)).head(5),
            f"{source_col}_frequency": job_frequency.head(5),
            f"{source_col}_is_rare": result.get(f"{source_col}_is_rare", pd.Series(dtype=int)).head(5),
        })
        print(sample_job.to_string(index=False))

    return result


def preprocess_speaker_job_option_minimal(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["minimal"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker_job(df=df, **args)


def preprocess_speaker_job_option_expanded(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["expanded"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker_job(df=df, **args)


def preprocess_speaker_job_option_rare_safe(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["rare_safe"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker_job(df=df, **args)


__all__ = [
    "DEFAULT_SOURCE_COL",
    "DEFAULT_SCALE",
    "SpeakerJobDSOptions",
    "PRESETS",
    "preprocess_speaker_job",
    "preprocess_speaker_job_option_minimal",
    "preprocess_speaker_job_option_expanded",
    "preprocess_speaker_job_option_rare_safe",
]