"""State-info preprocessing utilities.

This module mirrors the API and behavior of ``src.preprocessing.speaker_job`` so
callers can toggle preprocessing options via flat kwargs or presets. The main
entry point is :func:`preprocess_state_info` which returns a dataframe with
additional geographic features.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Public defaults for the flat-kwargs API.
# ---------------------------------------------------------------------------
DEFAULT_SOURCE_COL = "state_info"
DEFAULT_DROP = False
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_CLEAN_TEXT = True
DEFAULT_NORMALIZE_STATE = True
DEFAULT_GROUP_RARE = True
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_RARE_LABEL = "other"
DEFAULT_ADD_IS_US_STATE = True
DEFAULT_ADD_FREQUENCY = True
DEFAULT_ADD_IS_RARE = True
DEFAULT_ADD_GROUPED_STATE = True
DEFAULT_ADD_LENGTH_FEATURES = True
DEFAULT_ADD_TOKEN_COUNT = True
DEFAULT_ADD_HAS_US_WORDS = True
DEFAULT_ADD_US_REGION = False
DEFAULT_VERBOSE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'


# ---------------------------------------------------------------------------
# U.S. geography reference tables
# ---------------------------------------------------------------------------
_US_STATES: dict[str, str] = {
    "alabama": "al", "alaska": "ak", "arizona": "az", "arkansas": "ar",
    "california": "ca", "colorado": "co", "connecticut": "ct", "delaware": "de",
    "district of columbia": "dc", "florida": "fl", "georgia": "ga", "hawaii": "hi",
    "idaho": "id", "illinois": "il", "indiana": "in", "iowa": "ia",
    "kansas": "ks", "kentucky": "ky", "louisiana": "la", "maine": "me",
    "maryland": "md", "massachusetts": "ma", "michigan": "mi", "minnesota": "mn",
    "mississippi": "ms", "missouri": "mo", "montana": "mt", "nebraska": "ne",
    "nevada": "nv", "new hampshire": "nh", "new jersey": "nj", "new mexico": "nm",
    "new york": "ny", "north carolina": "nc", "north dakota": "nd", "ohio": "oh",
    "oklahoma": "ok", "oregon": "or", "pennsylvania": "pa", "rhode island": "ri",
    "south carolina": "sc", "south dakota": "sd", "tennessee": "tn", "texas": "tx",
    "utah": "ut", "vermont": "vt", "virginia": "va", "washington": "wa",
    "west virginia": "wv", "wisconsin": "wi", "wyoming": "wy",
}
_CODE_TO_NAME: dict[str, str] = {v: k for k, v in _US_STATES.items()}
_US_STATE_NAMES: frozenset[str] = frozenset(_US_STATES.keys())

# Maps full state name -> Census region.
_STATE_REGIONS: dict[str, str] = {
    "connecticut": "northeast", "maine": "northeast", "massachusetts": "northeast",
    "new hampshire": "northeast", "new jersey": "northeast", "new york": "northeast",
    "pennsylvania": "northeast", "rhode island": "northeast", "vermont": "northeast",
    "alabama": "south", "arkansas": "south", "delaware": "south",
    "district of columbia": "south", "florida": "south", "georgia": "south",
    "kentucky": "south", "louisiana": "south", "maryland": "south",
    "mississippi": "south", "north carolina": "south", "oklahoma": "south",
    "south carolina": "south", "tennessee": "south", "texas": "south",
    "virginia": "south", "west virginia": "south",
    "illinois": "midwest", "indiana": "midwest", "iowa": "midwest",
    "kansas": "midwest", "michigan": "midwest", "minnesota": "midwest",
    "missouri": "midwest", "nebraska": "midwest", "north dakota": "midwest",
    "ohio": "midwest", "south dakota": "midwest", "wisconsin": "midwest",
    "alaska": "west", "arizona": "west", "california": "west", "colorado": "west",
    "hawaii": "west", "idaho": "west", "montana": "west", "nevada": "west",
    "new mexico": "west", "oregon": "west", "utah": "west", "washington": "west",
    "wyoming": "west",
}


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------
@dataclass
class StateInfoDSOptions:
    source_col: str = DEFAULT_SOURCE_COL
    drop: bool = DEFAULT_DROP
    keep_original: bool = DEFAULT_KEEP_ORIGINAL
    clean_text: bool = DEFAULT_CLEAN_TEXT
    normalize_state: bool = DEFAULT_NORMALIZE_STATE
    group_rare: bool = DEFAULT_GROUP_RARE
    rare_threshold: int = DEFAULT_RARE_THRESHOLD
    rare_label: str = DEFAULT_RARE_LABEL
    add_is_us_state: bool = DEFAULT_ADD_IS_US_STATE
    add_frequency: bool = DEFAULT_ADD_FREQUENCY
    add_is_rare: bool = DEFAULT_ADD_IS_RARE
    add_grouped_state: bool = DEFAULT_ADD_GROUPED_STATE
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT
    add_has_us_words: bool = DEFAULT_ADD_HAS_US_WORDS
    add_us_region: bool = DEFAULT_ADD_US_REGION
    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, StateInfoDSOptions] = {
    "drop": StateInfoDSOptions(drop=True),
    "minimal": StateInfoDSOptions(
        add_grouped_state=False,
        add_length_features=False,
        add_token_count=False,
        add_has_us_words=False,
        add_us_region=False,
    ),
    "expanded": StateInfoDSOptions(add_us_region=True),
    "rare_safe": StateInfoDSOptions(group_rare=True, rare_threshold=5, add_us_region=False),
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
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


def _clean_state_info(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_state(text: str) -> str:
    """Return full state name if text is a known name or 2-letter code."""
    if text in _US_STATE_NAMES:
        return text
    if text in _CODE_TO_NAME:
        return _CODE_TO_NAME[text]
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preprocess_state_info(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SOURCE_COL,
    drop: bool = DEFAULT_DROP,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    clean_text: bool = DEFAULT_CLEAN_TEXT,
    normalize_state: bool = DEFAULT_NORMALIZE_STATE,
    group_rare: bool = DEFAULT_GROUP_RARE,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    rare_label: str = DEFAULT_RARE_LABEL,
    add_is_us_state: bool = DEFAULT_ADD_IS_US_STATE,
    add_frequency: bool = DEFAULT_ADD_FREQUENCY,
    add_is_rare: bool = DEFAULT_ADD_IS_RARE,
    add_grouped_state: bool = DEFAULT_ADD_GROUPED_STATE,
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES,
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT,
    add_has_us_words: bool = DEFAULT_ADD_HAS_US_WORDS,
    add_us_region: bool = DEFAULT_ADD_US_REGION,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess the ``state_info`` column and return an enriched dataframe.

    When ``drop=True`` the source column is removed and no features are added.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()

    if drop:
        return result.drop(columns=[source_col])

    state_raw = result[source_col].fillna("").astype(str)
    _scalable: list[str] = []

    if clean_text:
        state_clean = state_raw.apply(_clean_state_info)
    else:
        state_clean = state_raw.str.strip().str.lower()

    state_clean = state_clean.replace("", "unknown")

    if normalize_state:
        state_normalized = state_clean.apply(_normalize_state)
    else:
        state_normalized = state_clean

    state_counts = state_normalized.value_counts()
    state_frequency = state_normalized.map(state_counts).fillna(0).astype(int)

    if add_frequency:
        result[f"{source_col}_frequency"] = state_frequency
        result[f"{source_col}_frequency_pct"] = state_frequency / max(len(result), 1)
        _scalable += [f"{source_col}_frequency", f"{source_col}_frequency_pct"]

    if add_is_rare:
        result[f"{source_col}_is_rare"] = state_frequency.lt(int(rare_threshold)).astype(int)

    if add_grouped_state:
        rare_states = state_counts[state_counts < int(rare_threshold)].index
        result[f"{source_col}_grouped"] = state_normalized.where(
            ~state_normalized.isin(rare_states), rare_label
        )

    if add_is_us_state:
        result[f"{source_col}_is_us_state"] = state_normalized.isin(_US_STATE_NAMES).astype(int)

    if add_length_features:
        result[f"{source_col}_char_len"] = state_normalized.str.len()
        _scalable.append(f"{source_col}_char_len")

    if add_token_count:
        result[f"{source_col}_token_count"] = state_normalized.apply(lambda x: len(x.split()))
        _scalable.append(f"{source_col}_token_count")

    if add_has_us_words:
        result[f"{source_col}_has_us_words"] = state_normalized.str.contains(
            r"\bus\b|\bunited states\b|\busa\b", regex=True
        ).fillna(False).astype(int)

    if add_us_region:
        result[f"{source_col}_us_region"] = state_normalized.map(_STATE_REGIONS).fillna("unknown")

    if scale != 'none':
        for col in _scalable:
            result[col] = _scale_col(result[col], scale)

    result[f"{source_col}_clean"] = state_normalized

    if keep_original:
        result[source_col] = state_raw
    else:
        result = result.drop(columns=[source_col])

    if group_rare and not add_grouped_state:
        raise ValueError("group_rare=True requires add_grouped_state=True so rare values can be grouped")

    if verbose:
        unknown_count = (state_normalized == "unknown").sum()
        print("Total rows:", len(result))
        print(f"Unknown/empty {source_col} values: {unknown_count:,}")
        print(f"Unique {source_col} values: {state_normalized.nunique():,}")
        print(f"U.S. state matches: {state_normalized.isin(_US_STATE_NAMES).sum():,}")
        print(f"Rare value threshold: < {rare_threshold} occurrences -> grouped as '{rare_label}'")
        print(f"\nTop 10 {source_col} values:")
        print(state_counts.head(10).to_string())
        print("\nSample before/after:")
        sample = pd.DataFrame({
            f"{source_col}_raw": state_raw.head(5),
            f"{source_col}_clean": state_normalized.head(5),
            f"{source_col}_is_us_state": state_normalized.isin(_US_STATE_NAMES).head(5).astype(int),
        })
        print(sample.to_string(index=False))

    return result


# ---------------------------------------------------------------------------
# Preset convenience functions
# ---------------------------------------------------------------------------
def preprocess_state_info_option_drop(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["drop"]
    return preprocess_state_info(df=df, **{**opts.__dict__, "source_col": source_col})


def preprocess_state_info_option_minimal(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["minimal"]
    return preprocess_state_info(df=df, **{**opts.__dict__, "source_col": source_col})


def preprocess_state_info_option_expanded(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["expanded"]
    return preprocess_state_info(df=df, **{**opts.__dict__, "source_col": source_col})


def preprocess_state_info_option_rare_safe(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    opts = PRESETS["rare_safe"]
    return preprocess_state_info(df=df, **{**opts.__dict__, "source_col": source_col})


__all__ = [
    "DEFAULT_SOURCE_COL",
    "DEFAULT_SCALE",
    "StateInfoDSOptions",
    "PRESETS",
    "preprocess_state_info",
    "preprocess_state_info_option_drop",
    "preprocess_state_info_option_minimal",
    "preprocess_state_info_option_expanded",
    "preprocess_state_info_option_rare_safe",
]
