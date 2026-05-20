"""Party affiliation preprocessing utilities.

This module mirrors the API and behavior of ``src.preprocessing.speaker_job`` so
callers can toggle preprocessing options via flat kwargs or presets. The main
entry point is :func:`preprocess_party_affiliation` which returns a dataframe with
additional party-affiliation features.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import pandas as pd


# Public defaults for the flat-kwargs API.
DEFAULT_SOURCE_COL = "party_affiliation"
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_CLEAN_TEXT = True
DEFAULT_GROUP_RARE = True
DEFAULT_RARE_THRESHOLD = 5
DEFAULT_RARE_LABEL = "other"
DEFAULT_ADD_LENGTH_FEATURES = True
DEFAULT_ADD_FREQUENCY = True
DEFAULT_ADD_IS_RARE = True
DEFAULT_ADD_GROUPED_PARTY = True
DEFAULT_ADD_SLASH_FLAG = True
DEFAULT_ADD_AMPERSAND_FLAG = True
DEFAULT_ADD_COMMA_FLAG = True
DEFAULT_ADD_PARENTHESES_FLAG = True
DEFAULT_ADD_TOKEN_COUNT = True
DEFAULT_ADD_IS_MAJOR_PARTY = True
DEFAULT_ADD_IS_INSTITUTIONAL = True
DEFAULT_ADD_PARTY_PRIMARY_TRUE_RATE = False
DEFAULT_PARTY_LABEL_COL: Optional[str] = None
DEFAULT_VERBOSE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'


@dataclass
class PartyAffiliationDSOptions:
    source_col: str = DEFAULT_SOURCE_COL
    keep_original: bool = DEFAULT_KEEP_ORIGINAL
    clean_text: bool = DEFAULT_CLEAN_TEXT
    group_rare: bool = DEFAULT_GROUP_RARE
    rare_threshold: int = DEFAULT_RARE_THRESHOLD
    rare_label: str = DEFAULT_RARE_LABEL
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES
    add_frequency: bool = DEFAULT_ADD_FREQUENCY
    add_is_rare: bool = DEFAULT_ADD_IS_RARE
    add_grouped_party: bool = DEFAULT_ADD_GROUPED_PARTY
    add_slash_flag: bool = DEFAULT_ADD_SLASH_FLAG
    add_ampersand_flag: bool = DEFAULT_ADD_AMPERSAND_FLAG
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG
    add_parentheses_flag: bool = DEFAULT_ADD_PARENTHESES_FLAG
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT
    add_is_major_party: bool = DEFAULT_ADD_IS_MAJOR_PARTY
    add_is_institutional: bool = DEFAULT_ADD_IS_INSTITUTIONAL
    add_party_primary_true_rate: bool = DEFAULT_ADD_PARTY_PRIMARY_TRUE_RATE
    party_label_col: Optional[str] = DEFAULT_PARTY_LABEL_COL
    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


PRESETS: dict[str, PartyAffiliationDSOptions] = {
    "minimal": PartyAffiliationDSOptions(
        add_grouped_party=False,
        add_party_primary_true_rate=False,
        add_is_major_party=False,
        add_is_institutional=False,
        add_slash_flag=False,
        add_ampersand_flag=False,
        add_comma_flag=False,
        add_parentheses_flag=False,
    ),
    "expanded": PartyAffiliationDSOptions(),
    "rare_safe": PartyAffiliationDSOptions(
        group_rare=True, 
        rare_threshold=5, 
        add_party_primary_true_rate=False
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


def _clean_party_affiliation(text: str) -> str:
    """Clean and normalize party affiliation text.
    
    Lowercases, strips whitespace, removes HTML tags, and collapses
    internal whitespace.
    """
    text = str(text).lower().strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_party_affiliation(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SOURCE_COL,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    clean_text: bool = DEFAULT_CLEAN_TEXT,
    group_rare: bool = DEFAULT_GROUP_RARE,
    rare_threshold: int = DEFAULT_RARE_THRESHOLD,
    rare_label: str = DEFAULT_RARE_LABEL,
    add_length_features: bool = DEFAULT_ADD_LENGTH_FEATURES,
    add_frequency: bool = DEFAULT_ADD_FREQUENCY,
    add_is_rare: bool = DEFAULT_ADD_IS_RARE,
    add_grouped_party: bool = DEFAULT_ADD_GROUPED_PARTY,
    add_slash_flag: bool = DEFAULT_ADD_SLASH_FLAG,
    add_ampersand_flag: bool = DEFAULT_ADD_AMPERSAND_FLAG,
    add_comma_flag: bool = DEFAULT_ADD_COMMA_FLAG,
    add_parentheses_flag: bool = DEFAULT_ADD_PARENTHESES_FLAG,
    add_token_count: bool = DEFAULT_ADD_TOKEN_COUNT,
    add_is_major_party: bool = DEFAULT_ADD_IS_MAJOR_PARTY,
    add_is_institutional: bool = DEFAULT_ADD_IS_INSTITUTIONAL,
    add_party_primary_true_rate: bool = DEFAULT_ADD_PARTY_PRIMARY_TRUE_RATE,
    party_label_col: Optional[str] = DEFAULT_PARTY_LABEL_COL,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Preprocess the `party_affiliation` column and return an enriched dataframe.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe containing the party_affiliation column.
    source_col : str
        Name of the column to preprocess (default: "party_affiliation").
    keep_original : bool
        Whether to keep the original column in the output (default: True).
    clean_text : bool
        Whether to clean and normalize party affiliation text (default: True).
    group_rare : bool
        Whether to group rare party affiliations (default: True).
    rare_threshold : int
        Frequency threshold below which parties are considered rare (default: 5).
    rare_label : str
        Label to assign to rare parties (default: "other").
    add_length_features : bool
        Whether to add character length and token count features (default: True).
    add_frequency : bool
        Whether to add frequency encoding (default: True).
    add_is_rare : bool
        Whether to add is_rare binary flag (default: True).
    add_grouped_party : bool
        Whether to add grouped version with rare parties merged (default: True).
    add_slash_flag : bool
        Whether to add flag for "/" in raw text (default: True).
    add_ampersand_flag : bool
        Whether to add flag for "&" in raw text (default: True).
    add_comma_flag : bool
        Whether to add flag for "," in raw text (default: True).
    add_parentheses_flag : bool
        Whether to add flag for parentheses in raw text (default: True).
    add_token_count : bool
        Whether to add token count feature (default: True).
    add_is_major_party : bool
        Whether to add binary flag for major parties (democrat/republican) (default: True).
    add_is_institutional : bool
        Whether to add flag for institutional roles (state-official, business-leader, etc.) (default: True).
    add_party_primary_true_rate : bool
        Whether to add true-claim rate by party (default: False). 
        **Warning: Must be computed inside CV folds to avoid leakage**.
    party_label_col : Optional[str]
        Name of the label column (required if add_party_primary_true_rate=True).
    verbose : bool
        Whether to print detailed statistics (default: False).

    Returns:
    --------
    pd.DataFrame
        A copy of the input dataframe with additional party affiliation features.

    Notes:
    ------
    - If ``add_party_primary_true_rate`` is True, this must be computed inside
      cross-validation folds to avoid leakage.
    """
    if source_col not in df.columns:
        raise KeyError(f"Column '{source_col}' not found in DataFrame")

    result = df.copy()
    party_raw = result[source_col].fillna("").astype(str)
    _scalable: list[str] = []

    if clean_text:
        party_clean = party_raw.apply(_clean_party_affiliation)
    else:
        party_clean = party_raw.str.strip().str.lower()

    party_clean = party_clean.replace("", "unknown")
    party_counts = party_clean.value_counts()
    party_frequency = party_clean.map(party_counts).fillna(0).astype(int)

    if add_frequency:
        result[f"{source_col}_frequency"] = party_frequency
        result[f"{source_col}_frequency_pct"] = party_frequency / max(len(result), 1)
        _scalable += [f"{source_col}_frequency", f"{source_col}_frequency_pct"]

    if add_is_rare:
        result[f"{source_col}_is_rare"] = party_frequency.lt(int(rare_threshold)).astype(int)

    if add_grouped_party:
        rare_parties = party_counts[party_counts < int(rare_threshold)].index
        result[f"{source_col}_grouped"] = party_clean.where(~party_clean.isin(rare_parties), rare_label)

    if add_length_features:
        result[f"{source_col}_char_len"] = party_clean.str.len()
        result[f"{source_col}_token_count"] = party_clean.apply(lambda value: len(value.split()))
        _scalable += [f"{source_col}_char_len", f"{source_col}_token_count"]

    if add_slash_flag:
        result[f"{source_col}_has_slash"] = party_raw.str.contains("/", regex=False).fillna(False).astype(int)

    if add_ampersand_flag:
        result[f"{source_col}_has_ampersand"] = party_raw.str.contains("&", regex=False).fillna(False).astype(int)

    if add_comma_flag:
        result[f"{source_col}_has_comma"] = party_raw.str.contains(",", regex=False).fillna(False).astype(int)

    if add_parentheses_flag:
        result[f"{source_col}_has_parentheses"] = party_raw.str.contains(
            r"[\(\)]", regex=True
        ).fillna(False).astype(int)

    if add_is_major_party:
        major_parties = ["democrat", "republican"]
        result[f"{source_col}_is_major_party"] = party_clean.isin(major_parties).astype(int)

    if add_is_institutional:
        institutional_roles = [
            "state-official",
            "business-leader",
            "journalist",
            "newsmaker",
            "columnist",
            "county-commissioner",
            "education-official",
            "labor-leader",
            "activist",
        ]
        result[f"{source_col}_is_institutional"] = party_clean.isin(institutional_roles).astype(int)

    if add_party_primary_true_rate:
        if party_label_col is None:
            raise ValueError(
                "add_party_primary_true_rate=True requires party_label_col to be specified. "
                "Set party_label_col to the name of the label column (usually 'label')."
            )
        if party_label_col not in result.columns:
            raise KeyError(f"Label column '{party_label_col}' not found in DataFrame.")
        party_true_rates = result.groupby(party_clean)[party_label_col].mean()
        result[f"{source_col}_primary_true_rate"] = party_clean.map(party_true_rates).fillna(0.5)

    if scale != 'none':
        for col in _scalable:
            result[col] = _scale_col(result[col], scale)

    result[f"{source_col}_clean"] = party_clean

    if keep_original:
        result[source_col] = party_raw
    else:
        result = result.drop(columns=[source_col])

    if group_rare and not add_grouped_party:
        raise ValueError("group_rare=True requires add_grouped_party=True so rare parties can be grouped")

    if verbose:
        print("=" * 70)
        print("PARTY AFFILIATION PREPROCESSING")
        print("=" * 70)
        unknown_parties = (party_clean == "unknown").sum()
        print(f"\nTotal rows: {len(result):,}")
        print(f"Unknown/empty {source_col} values: {unknown_parties:,}")
        print(f"Unique {source_col} values: {party_clean.nunique():,}")
        print(f"Rare party threshold: < {rare_threshold} occurrences -> grouped as '{rare_label}'")
        
        rare_count = party_frequency.lt(rare_threshold).sum()
        rare_rows = result[f"{source_col}_is_rare"].sum() if add_is_rare else 0
        print(f"Parties below threshold: {rare_count:,}")
        print(f"Rows belonging to rare parties: {rare_rows:,}")
        
        if add_is_major_party:
            major_count = result[f"{source_col}_is_major_party"].sum()
            print(f"Major party (democrat/republican) rows: {major_count:,}")
        
        if add_is_institutional:
            institutional_count = result[f"{source_col}_is_institutional"].sum()
            print(f"Institutional role rows: {institutional_count:,}")

        print(f"\nTop 15 parties by frequency:")
        print(party_counts.head(15).to_string())
        
        print(f"\nSample before/after:")
        sample_cols = [f"{source_col}_raw", f"{source_col}_clean"]
        if add_grouped_party:
            sample_cols.append(f"{source_col}_grouped")
        if add_frequency:
            sample_cols.append(f"{source_col}_frequency")
        if add_is_rare:
            sample_cols.append(f"{source_col}_is_rare")
        
        sample_party = pd.DataFrame({
            f"{source_col}_raw": party_raw.head(10),
            f"{source_col}_clean": party_clean.head(10),
        })
        if add_grouped_party:
            sample_party[f"{source_col}_grouped"] = result[f"{source_col}_grouped"].head(10)
        if add_frequency:
            sample_party[f"{source_col}_frequency"] = party_frequency.head(10)
        if add_is_rare:
            sample_party[f"{source_col}_is_rare"] = result.get(f"{source_col}_is_rare", pd.Series(dtype=int)).head(10)
        
        print(sample_party.to_string(index=False))

    return result


def preprocess_party_affiliation_option_minimal(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    """Minimal preprocessing: clean text, frequency, and basic flags only."""
    opts = PRESETS["minimal"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_party_affiliation(df=df, **args)


def preprocess_party_affiliation_option_expanded(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    """Expanded preprocessing: all features including major party and institutional flags."""
    opts = PRESETS["expanded"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_party_affiliation(df=df, **args)


def preprocess_party_affiliation_option_rare_safe(df: pd.DataFrame, source_col: str = DEFAULT_SOURCE_COL) -> pd.DataFrame:
    """Rare-safe preprocessing: groups rare parties without label leakage."""
    opts = PRESETS["rare_safe"]
    args = {**opts.__dict__, "source_col": source_col}
    return preprocess_party_affiliation(df=df, **args)


__all__ = [
    "DEFAULT_SOURCE_COL",
    "DEFAULT_SCALE",
    "PartyAffiliationDSOptions",
    "PRESETS",
    "preprocess_party_affiliation",
    "preprocess_party_affiliation_option_minimal",
    "preprocess_party_affiliation_option_expanded",
    "preprocess_party_affiliation_option_rare_safe",
]
