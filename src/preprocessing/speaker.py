"""Speaker preprocessing utilities.

This module converts the raw ``speaker`` column into a feature-enriched
dataframe. The main entry point is :func:`preprocess_speaker`, which accepts a
dataframe plus options and returns a single dataframe with speaker-derived
columns added.

The module mirrors the flat-kwargs style used by ``subject.py`` so callers can
toggle individual features without changing downstream code.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


# Public defaults for the flat-kwargs API.
DEFAULT_SPEAKER_SOURCE_COL = "speaker"
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_CLEAN_TEXT = True
DEFAULT_NORMALIZE_SEPARATORS = True
DEFAULT_GROUP_RARE = True
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_RARE_LABEL = "other"
DEFAULT_ADD_LENGTH_FEATURES = True
DEFAULT_ADD_FREQUENCY = True
DEFAULT_ADD_IS_RARE = True
DEFAULT_ADD_GROUPED_SPEAKER = True
DEFAULT_ADD_TITLE_FLAG = True
DEFAULT_ADD_COMMA_FLAG = True
DEFAULT_ADD_PERIOD_FLAG = True
DEFAULT_ADD_TOKEN_COUNT = True
DEFAULT_ADD_SPEAKER_PRIMARY_TRUE_RATE = False
DEFAULT_SPEAKER_LABEL_COL: str | None = None
DEFAULT_VERBOSE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'


@dataclass
class SpeakerDSOptions:
    """Options for the speaker preprocessing pipeline (flat-kwargs style)."""

    source_col: str = DEFAULT_SPEAKER_SOURCE_COL
    keep_original: bool = DEFAULT_KEEP_ORIGINAL
    clean_text: bool = DEFAULT_CLEAN_TEXT
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS
    group_rare: bool = DEFAULT_GROUP_RARE
    rare_threshold: int = DEFAULT_RARE_THRESHOLD
    rare_label: str = DEFAULT_RARE_LABEL
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES
    add_frequency: bool = DEFAULT_ADD_FREQUENCY
    add_is_rare: bool = DEFAULT_ADD_IS_RARE
    add_grouped_speaker: bool = DEFAULT_ADD_GROUPED_SPEAKER
    add_title_flag: bool = DEFAULT_ADD_TITLE_FLAG
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG
    add_period_flag: bool = DEFAULT_ADD_PERIOD_FLAG
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT
    add_speaker_primary_true_rate: bool = DEFAULT_ADD_SPEAKER_PRIMARY_TRUE_RATE
    speaker_label_col: str | None = DEFAULT_SPEAKER_LABEL_COL
    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


SPEAKER_PRESETS: dict[str, SpeakerDSOptions] = {
    "minimal": SpeakerDSOptions(
        add_grouped_speaker=False,
        add_speaker_primary_true_rate=False,
        add_title_flag=False,
        add_comma_flag=False,
        add_period_flag=False,
    ),
    "expanded": SpeakerDSOptions(),
    "rare_safe": SpeakerDSOptions(
        group_rare=True,
        rare_threshold=5,
        add_speaker_primary_true_rate=False,
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


def _clean_speaker_text(text: str, normalize_separators: bool = True) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    if normalize_separators:
        text = re.sub(r"[\|/_;,-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s'.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_speaker(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SPEAKER_SOURCE_COL,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    clean_text: bool = DEFAULT_CLEAN_TEXT,
    normalize_separators: bool = DEFAULT_NORMALIZE_SEPARATORS,
    group_rare: bool = DEFAULT_GROUP_RARE,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    rare_label: str = DEFAULT_RARE_LABEL,
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES,
    add_frequency: bool = DEFAULT_ADD_FREQUENCY,
    add_is_rare: bool = DEFAULT_ADD_IS_RARE,
    add_grouped_speaker: bool = DEFAULT_ADD_GROUPED_SPEAKER,
    add_title_flag: bool = DEFAULT_ADD_TITLE_FLAG,
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG,
    add_period_flag: bool = DEFAULT_ADD_PERIOD_FLAG,
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT,
    add_speaker_primary_true_rate: bool = DEFAULT_ADD_SPEAKER_PRIMARY_TRUE_RATE,
    speaker_label_col: str | None = DEFAULT_SPEAKER_LABEL_COL,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess the speaker column and return a single enriched dataframe.

    WARNING: If add_speaker_primary_true_rate=True, the resulting feature will
    contain label information and must be computed within cross-validation
    folds to avoid leakage.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()
    speaker_raw = result[source_col].fillna("").astype(str)
    _scalable: list[str] = []

    if clean_text:
        speaker_clean = speaker_raw.apply(
            lambda value: _clean_speaker_text(value, normalize_separators=normalize_separators)
        )
    else:
        speaker_clean = speaker_raw.str.strip().str.lower()

    speaker_clean = speaker_clean.replace("", "unknown")
    speaker_counts = speaker_clean.value_counts()
    speaker_frequency = speaker_clean.map(speaker_counts).fillna(0).astype(int)

    if add_frequency:
        result[f"{source_col}_frequency"] = speaker_frequency
        result[f"{source_col}_frequency_pct"] = speaker_frequency / max(len(result), 1)
        _scalable += [f"{source_col}_frequency", f"{source_col}_frequency_pct"]

    if add_is_rare:
        result[f"{source_col}_is_rare"] = speaker_frequency.lt(int(rare_threshold)).astype(int)

    if add_grouped_speaker:
        rare_speakers = speaker_counts[speaker_counts < int(rare_threshold)].index
        result[f"{source_col}_grouped"] = speaker_clean.where(~speaker_clean.isin(rare_speakers), rare_label)

    if add_length_features:
        result[f"{source_col}_char_len"] = speaker_clean.str.len()
        result[f"{source_col}_token_count"] = speaker_clean.apply(lambda value: len(value.split()))
        _scalable += [f"{source_col}_char_len", f"{source_col}_token_count"]

    if add_title_flag:
        result[f"{source_col}_has_title"] = speaker_raw.str.contains(
            r"\b(?:mr|mrs|ms|dr|gov|sen|rep|pres|prof)\b",
            case=False,
            regex=True,
        ).fillna(False).astype(int)

    if add_comma_flag:
        result[f"{source_col}_has_comma"] = speaker_raw.str.contains(",", regex=False).fillna(False).astype(int)

    if add_period_flag:
        result[f"{source_col}_has_period"] = speaker_raw.str.contains(".", regex=False).fillna(False).astype(int)

    if add_speaker_primary_true_rate:
        if speaker_label_col is None:
            raise ValueError(
                "add_speaker_primary_true_rate=True requires speaker_label_col to be specified. "
                "Set speaker_label_col to the name of the label column (usually 'label')."
            )
        if speaker_label_col not in result.columns:
            raise KeyError(
                f"Label column '{speaker_label_col}' not found in DataFrame. "
                f"Available columns: {list(result.columns[:10])}..."
            )
        speaker_true_rates = result.groupby(speaker_clean)[speaker_label_col].mean()
        result[f"{source_col}_primary_true_rate"] = speaker_clean.map(speaker_true_rates).fillna(0.5)

    if scale != 'none':
        for col in _scalable:
            result[col] = _scale_col(result[col], scale)

    result[f"{source_col}_clean"] = speaker_clean

    if keep_original:
        result[source_col] = speaker_raw
    else:
        result = result.drop(columns=[source_col])

    if group_rare and not add_grouped_speaker:
        raise ValueError("group_rare=True requires add_grouped_speaker=True so rare speakers can be grouped")

    if verbose:
        empty_speakers = (speaker_clean == "unknown").sum()
        print(f"Total rows: {len(result):,}")
        print(f"Unknown/empty speakers: {empty_speakers:,}")
        print(f"Unique speakers: {speaker_clean.nunique():,}")
        print(f"Rare speaker threshold: < {rare_threshold} occurrences -> grouped as '{rare_label}'")
        print("\nTop 10 speakers by frequency:")
        print(speaker_counts.head(10).to_string())
        print("\nSample before/after:")
        sample_speaker = pd.DataFrame({
            "speaker_raw": speaker_raw.head(5),
            "speaker_clean": speaker_clean.head(5),
            f"{source_col}_grouped": result.get(f"{source_col}_grouped", pd.Series(dtype=str)).head(5),
            f"{source_col}_frequency": speaker_frequency.head(5),
            f"{source_col}_is_rare": result.get(f"{source_col}_is_rare", pd.Series(dtype=int)).head(5),
        })
        print(sample_speaker.to_string(index=False))

    return result


def preprocess_speaker_option_minimal(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SPEAKER_SOURCE_COL,
) -> pd.DataFrame:
    opts = SPEAKER_PRESETS["minimal"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker(df=df, **args)


def preprocess_speaker_option_expanded(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SPEAKER_SOURCE_COL,
) -> pd.DataFrame:
    opts = SPEAKER_PRESETS["expanded"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker(df=df, **args)


def preprocess_speaker_option_rare_safe(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SPEAKER_SOURCE_COL,
) -> pd.DataFrame:
    opts = SPEAKER_PRESETS["rare_safe"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_speaker(df=df, **args)


__all__ = [
    "DEFAULT_SPEAKER_SOURCE_COL",
    "DEFAULT_SCALE",
    "SpeakerDSOptions",
    "SPEAKER_PRESETS",
    "preprocess_speaker",
    "preprocess_speaker_option_expanded",
    "preprocess_speaker_option_minimal",
    "preprocess_speaker_option_rare_safe",
]