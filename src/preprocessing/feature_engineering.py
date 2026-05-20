"""Cross-column feature engineering utilities.

Computes interaction features, aggregate statistics, and text-style signals
that span multiple preprocessing module outputs. Run after all column-level
modules have finished (i.e., at the end of preprocess_one_step).

Three feature families are implemented:

  1. Interaction features  — concatenate two cleaned columns into a joint
     categorical key. Capture patterns invisible in either column alone.

  2. Aggregate features    — compute per-group statistics (mean label, mean
     length, etc.). These are leakage-prone: compute ONLY inside CV folds.

  3. Text-style features   — derive linguistic signals directly from the
     cleaned statement text (negations, hedges, absolutist language, etc.).

Column name conventions
-----------------------
All output columns are prefixed ``fe_`` so they can be identified and
dropped or included as a group without ambiguity.

Source column dependencies
--------------------------
All features default to the ``*_clean`` columns that every preprocessing
module always produces, so no optional flags need to be enabled just to
run this module. Individual features note if a non-default source is used.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Source-column defaults
# ---------------------------------------------------------------------------
DEFAULT_STATEMENT_COL = "statement_clean"
DEFAULT_STATEMENT_ORIGINAL_COL = "statement_original"  # used for proper-noun heuristic
DEFAULT_SPEAKER_COL = "speaker_clean"
DEFAULT_SUBJECT_COL = "subject_clean"
DEFAULT_PARTY_COL = "party_affiliation_clean"
DEFAULT_SPEAKER_JOB_COL = "speaker_job_clean"
DEFAULT_STATE_COL = "state_info_clean"
DEFAULT_LABEL_COL: Optional[str] = None  # required for aggregate/target-stat features

# ---------------------------------------------------------------------------
# Interaction feature defaults
# ---------------------------------------------------------------------------
DEFAULT_ADD_SPEAKER_SUBJECT = False
DEFAULT_ADD_SPEAKER_PARTY = False
DEFAULT_ADD_SUBJECT_PARTY = False
DEFAULT_ADD_SPEAKER_JOB_SUBJECT = False
DEFAULT_ADD_STATE_PARTY = False
DEFAULT_ADD_SPEAKER_STATEMENT_LEN_BUCKET = False
# Word count thresholds for short / medium / long buckets.
DEFAULT_STATEMENT_LEN_BINS: tuple = (50, 150)

# ---------------------------------------------------------------------------
# Aggregate feature defaults  (leakage risk — CV folds only)
# ---------------------------------------------------------------------------
DEFAULT_ADD_SPEAKER_TRUE_RATE = False
DEFAULT_ADD_SUBJECT_TRUE_RATE = False
DEFAULT_ADD_PARTY_TRUE_RATE = False
DEFAULT_ADD_SPEAKER_AVG_STATEMENT_LEN = False
DEFAULT_ADD_SUBJECT_AVG_STATEMENT_LEN = False
DEFAULT_ADD_SPEAKER_AVG_PUNCTUATION = False
DEFAULT_ADD_SPEAKER_AVG_NUMBER_RATIO = False

# ---------------------------------------------------------------------------
# Text-style feature defaults
# ---------------------------------------------------------------------------
DEFAULT_ADD_NEGATION_COUNT = False
DEFAULT_ADD_HEDGE_COUNT = False
DEFAULT_ADD_ABSOLUTIST_COUNT = False
DEFAULT_ADD_NUMERAL_COUNT = False
DEFAULT_ADD_PROPER_NOUN_COUNT = False
DEFAULT_ADD_READABILITY = False
DEFAULT_ADD_SENTIMENT = False  # requires: pip install textblob

DEFAULT_VERBOSE = False
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'


# ---------------------------------------------------------------------------
# Word lists for text-style features
# ---------------------------------------------------------------------------
_NEGATION_WORDS: frozenset[str] = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere",
    "neither", "nor", "cannot", "cant", "wont", "wouldnt", "shouldnt",
    "couldnt", "didnt", "doesnt", "isnt", "arent", "wasnt", "werent",
    "havent", "hasnt", "hadnt", "without",
})

_HEDGE_WORDS: frozenset[str] = frozenset({
    "maybe", "perhaps", "possibly", "probably", "likely", "unlikely",
    "might", "may", "could", "seem", "seems", "seemed", "appears",
    "appear", "appeared", "suggests", "suggest", "indicates", "indicate",
    "approximately", "around", "about", "roughly", "generally",
    "sometimes", "often", "occasionally", "usually", "typically",
    "tend", "tends", "tended", "supposedly", "allegedly", "reportedly",
    "apparently", "presumably", "ostensibly",
})

_ABSOLUTIST_WORDS: frozenset[str] = frozenset({
    "always", "never", "everyone", "everybody", "nobody", "everything",
    "nothing", "everywhere", "nowhere", "all", "none", "only",
    "absolute", "absolutely", "completely", "entirely", "totally",
    "perfectly", "impossible", "certain", "certainly", "guaranteed",
    "undeniably", "unquestionably", "definitively", "proven", "fact",
    "truth", "lie", "lied", "lies", "false", "hoax",
})


# ---------------------------------------------------------------------------
# Options dataclass
# ---------------------------------------------------------------------------
@dataclass
class FeatureEngineeringOptions:
    """Options for the cross-column feature engineering module.

    All add_* flags default to False. Enable them individually to grow the
    feature set. Aggregate features require label_col and carry leakage risk.
    """

    # Source column names (match the *_clean outputs of the preprocessing modules)
    statement_col: str = DEFAULT_STATEMENT_COL
    statement_original_col: str = DEFAULT_STATEMENT_ORIGINAL_COL  # for proper-noun heuristic
    speaker_col: str = DEFAULT_SPEAKER_COL
    subject_col: str = DEFAULT_SUBJECT_COL
    party_col: str = DEFAULT_PARTY_COL
    speaker_job_col: str = DEFAULT_SPEAKER_JOB_COL
    state_col: str = DEFAULT_STATE_COL
    label_col: Optional[str] = DEFAULT_LABEL_COL  # required for aggregate features

    # --- Interaction features ---
    # Concatenate two cleaned columns into a joint categorical key.
    # Output columns: fe_<name> with values like "barack obama__health care"
    add_speaker_subject: bool = DEFAULT_ADD_SPEAKER_SUBJECT
    add_speaker_party: bool = DEFAULT_ADD_SPEAKER_PARTY
    add_subject_party: bool = DEFAULT_ADD_SUBJECT_PARTY
    add_speaker_job_subject: bool = DEFAULT_ADD_SPEAKER_JOB_SUBJECT
    add_state_party: bool = DEFAULT_ADD_STATE_PARTY
    add_speaker_statement_len_bucket: bool = DEFAULT_ADD_SPEAKER_STATEMENT_LEN_BUCKET
    statement_len_bins: tuple = DEFAULT_STATEMENT_LEN_BINS  # (short_max, medium_max) word counts

    # --- Aggregate features (leakage risk — CV folds only) ---
    # Empirical mean of label_col per group. Set label_col to use these.
    # WARNING: compute ONLY on training folds, then map to val/test rows.
    add_speaker_true_rate: bool = DEFAULT_ADD_SPEAKER_TRUE_RATE
    add_subject_true_rate: bool = DEFAULT_ADD_SUBJECT_TRUE_RATE
    add_party_true_rate: bool = DEFAULT_ADD_PARTY_TRUE_RATE
    # Non-label aggregates (no leakage risk)
    add_speaker_avg_statement_len: bool = DEFAULT_ADD_SPEAKER_AVG_STATEMENT_LEN
    add_subject_avg_statement_len: bool = DEFAULT_ADD_SUBJECT_AVG_STATEMENT_LEN
    add_speaker_avg_punctuation: bool = DEFAULT_ADD_SPEAKER_AVG_PUNCTUATION
    add_speaker_avg_number_ratio: bool = DEFAULT_ADD_SPEAKER_AVG_NUMBER_RATIO

    # --- Text-style features ---
    # All derived from statement_col (statement_clean by default).
    add_negation_count: bool = DEFAULT_ADD_NEGATION_COUNT
    add_hedge_count: bool = DEFAULT_ADD_HEDGE_COUNT
    add_absolutist_count: bool = DEFAULT_ADD_ABSOLUTIST_COUNT
    add_numeral_count: bool = DEFAULT_ADD_NUMERAL_COUNT
    # Heuristic: capitalized non-sentence-start words in statement_original_col.
    add_proper_noun_count: bool = DEFAULT_ADD_PROPER_NOUN_COUNT
    # Flesch Reading Ease approximation (no external library required).
    add_readability: bool = DEFAULT_ADD_READABILITY
    # TextBlob polarity and subjectivity. Requires: pip install textblob
    add_sentiment: bool = DEFAULT_ADD_SENTIMENT

    scale: str = DEFAULT_SCALE
    verbose: bool = DEFAULT_VERBOSE


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
PRESETS: dict[str, FeatureEngineeringOptions] = {
    "none": FeatureEngineeringOptions(),
    "interactions": FeatureEngineeringOptions(
        add_speaker_subject=True,
        add_speaker_party=True,
        add_subject_party=True,
        add_speaker_job_subject=True,
        add_state_party=True,
        add_speaker_statement_len_bucket=True,
    ),
    "text": FeatureEngineeringOptions(
        add_negation_count=True,
        add_hedge_count=True,
        add_absolutist_count=True,
        add_numeral_count=True,
        add_proper_noun_count=True,
        add_readability=True,
    ),
    # All non-leakage features. Aggregates excluded because they need CV folds.
    "expanded": FeatureEngineeringOptions(
        add_speaker_subject=True,
        add_speaker_party=True,
        add_subject_party=True,
        add_speaker_job_subject=True,
        add_state_party=True,
        add_speaker_statement_len_bucket=True,
        add_speaker_avg_statement_len=True,
        add_subject_avg_statement_len=True,
        add_speaker_avg_punctuation=True,
        add_speaker_avg_number_ratio=True,
        add_negation_count=True,
        add_hedge_count=True,
        add_absolutist_count=True,
        add_numeral_count=True,
        add_proper_noun_count=True,
        add_readability=True,
    ),
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


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z']+", str(text).lower())


def _count_from_set(tokens: list[str], word_set: frozenset[str]) -> int:
    return sum(1 for t in tokens if t in word_set)


def _count_syllables(word: str) -> int:
    """Heuristic syllable count via vowel-group detection."""
    word = re.sub(r"[^a-z]", "", word.lower())
    if not word:
        return 0
    count = len(re.findall(r"[aeiou]+", word))
    if word.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def _flesch_reading_ease(text: str) -> float:
    """Approximate Flesch Reading Ease score. Higher = easier to read."""
    sentences = max(1, len(re.split(r"[.!?]+", str(text).strip())))
    words = str(text).split()
    if not words:
        return 0.0
    syllables = sum(_count_syllables(w) for w in words)
    asl = len(words) / sentences
    asw = syllables / len(words)
    return 206.835 - 1.015 * asl - 84.6 * asw


def _proper_noun_heuristic(text: str) -> int:
    """Count capitalized words that are not at the start of a sentence."""
    # Split into sentences, then count capitalized non-first words.
    sentences = re.split(r"(?<=[.!?])\s+", str(text).strip())
    count = 0
    for sentence in sentences:
        words = sentence.split()
        if len(words) > 1:
            count += sum(1 for w in words[1:] if w and w[0].isupper() and w.isalpha())
    return count


def _len_bucket(word_count: int, bins: tuple) -> str:
    low, high = bins[0], bins[1]
    if word_count < low:
        return "short"
    if word_count <= high:
        return "medium"
    return "long"


def _check_col(df: pd.DataFrame, col: str, feature_name: str) -> bool:
    """Return True if col exists; warn and return False otherwise."""
    if col not in df.columns:
        warnings.warn(
            f"feature_engineering: cannot compute '{feature_name}' — "
            f"column '{col}' not found in dataframe. "
            f"Check that the upstream preprocessing module ran successfully."
        )
        return False
    return True


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def preprocess_feature_engineering(
    df: pd.DataFrame,
    statement_col: str = DEFAULT_STATEMENT_COL,
    statement_original_col: str = DEFAULT_STATEMENT_ORIGINAL_COL,
    speaker_col: str = DEFAULT_SPEAKER_COL,
    subject_col: str = DEFAULT_SUBJECT_COL,
    party_col: str = DEFAULT_PARTY_COL,
    speaker_job_col: str = DEFAULT_SPEAKER_JOB_COL,
    state_col: str = DEFAULT_STATE_COL,
    label_col: Optional[str] = DEFAULT_LABEL_COL,
    # Interaction features
    add_speaker_subject: bool = DEFAULT_ADD_SPEAKER_SUBJECT,
    add_speaker_party: bool = DEFAULT_ADD_SPEAKER_PARTY,
    add_subject_party: bool = DEFAULT_ADD_SUBJECT_PARTY,
    add_speaker_job_subject: bool = DEFAULT_ADD_SPEAKER_JOB_SUBJECT,
    add_state_party: bool = DEFAULT_ADD_STATE_PARTY,
    add_speaker_statement_len_bucket: bool = DEFAULT_ADD_SPEAKER_STATEMENT_LEN_BUCKET,
    statement_len_bins: tuple = DEFAULT_STATEMENT_LEN_BINS,
    # Aggregate features (leakage risk)
    add_speaker_true_rate: bool = DEFAULT_ADD_SPEAKER_TRUE_RATE,
    add_subject_true_rate: bool = DEFAULT_ADD_SUBJECT_TRUE_RATE,
    add_party_true_rate: bool = DEFAULT_ADD_PARTY_TRUE_RATE,
    add_speaker_avg_statement_len: bool = DEFAULT_ADD_SPEAKER_AVG_STATEMENT_LEN,
    add_subject_avg_statement_len: bool = DEFAULT_ADD_SUBJECT_AVG_STATEMENT_LEN,
    add_speaker_avg_punctuation: bool = DEFAULT_ADD_SPEAKER_AVG_PUNCTUATION,
    add_speaker_avg_number_ratio: bool = DEFAULT_ADD_SPEAKER_AVG_NUMBER_RATIO,
    # Text-style features
    add_negation_count: bool = DEFAULT_ADD_NEGATION_COUNT,
    add_hedge_count: bool = DEFAULT_ADD_HEDGE_COUNT,
    add_absolutist_count: bool = DEFAULT_ADD_ABSOLUTIST_COUNT,
    add_numeral_count: bool = DEFAULT_ADD_NUMERAL_COUNT,
    add_proper_noun_count: bool = DEFAULT_ADD_PROPER_NOUN_COUNT,
    add_readability: bool = DEFAULT_ADD_READABILITY,
    add_sentiment: bool = DEFAULT_ADD_SENTIMENT,
    scale: str = DEFAULT_SCALE,
    verbose: bool = DEFAULT_VERBOSE,
) -> pd.DataFrame:
    """Compute cross-column features and append them to the dataframe.

    All output columns are prefixed ``fe_``.

    Leakage warning
    ---------------
    ``add_speaker_true_rate``, ``add_subject_true_rate``, and
    ``add_party_true_rate`` use the label column to compute per-group means.
    These MUST be computed inside cross-validation folds — call this function
    separately on each training fold, then map to validation/test rows using
    the per-group means computed on the training fold only.
    """
    result = df.copy()
    _scalable: list[str] = []

    # -----------------------------------------------------------------------
    # INTERACTION FEATURES
    # Each concatenates two cleaned category columns into a joint key.
    # Source columns are the *_clean outputs of preprocessing modules and are
    # always present when running through preprocess_one_step.
    # -----------------------------------------------------------------------

    # fe_speaker_subject: speaker × subject
    # Captures whether certain speakers are more deceptive on specific topics.
    # Depends on: speaker_col (speaker_clean) and subject_col (subject_clean)
    if add_speaker_subject:
        if _check_col(result, speaker_col, "fe_speaker_subject") and \
           _check_col(result, subject_col, "fe_speaker_subject"):
            result["fe_speaker_subject"] = (
                result[speaker_col].astype(str) + "__" + result[subject_col].astype(str)
            )

    # fe_speaker_party: speaker × party_affiliation
    # Checks if a speaker's false-claim patterns vary depending on their party.
    # Depends on: speaker_col (speaker_clean) and party_col (party_affiliation_clean)
    if add_speaker_party:
        if _check_col(result, speaker_col, "fe_speaker_party") and \
           _check_col(result, party_col, "fe_speaker_party"):
            result["fe_speaker_party"] = (
                result[speaker_col].astype(str) + "__" + result[party_col].astype(str)
            )

    # fe_subject_party: subject × party_affiliation
    # Captures whether topics have different credibility patterns by party.
    # Depends on: subject_col (subject_clean) and party_col (party_affiliation_clean)
    if add_subject_party:
        if _check_col(result, subject_col, "fe_subject_party") and \
           _check_col(result, party_col, "fe_subject_party"):
            result["fe_subject_party"] = (
                result[subject_col].astype(str) + "__" + result[party_col].astype(str)
            )

    # fe_speaker_job_subject: speaker_job × subject
    # Role-topic interaction: does a speaker's occupation affect truthfulness by topic?
    # Depends on: speaker_job_col (speaker_job_clean) and subject_col (subject_clean)
    if add_speaker_job_subject:
        if _check_col(result, speaker_job_col, "fe_speaker_job_subject") and \
           _check_col(result, subject_col, "fe_speaker_job_subject"):
            result["fe_speaker_job_subject"] = (
                result[speaker_job_col].astype(str) + "__" + result[subject_col].astype(str)
            )

    # fe_state_party: state_info × party_affiliation
    # Captures regional political credibility patterns.
    # Depends on: state_col (state_info_clean) and party_col (party_affiliation_clean)
    if add_state_party:
        if _check_col(result, state_col, "fe_state_party") and \
           _check_col(result, party_col, "fe_state_party"):
            result["fe_state_party"] = (
                result[state_col].astype(str) + "__" + result[party_col].astype(str)
            )

    # fe_speaker_len_bucket: speaker × statement length bucket
    # Checks if certain speakers tend to make long vs. short claims differently.
    # Word count bins controlled by statement_len_bins (default: short<50, medium≤150, long>150).
    # Depends on: speaker_col (speaker_clean) and statement_col (statement_clean)
    if add_speaker_statement_len_bucket:
        if _check_col(result, speaker_col, "fe_speaker_len_bucket") and \
           _check_col(result, statement_col, "fe_speaker_len_bucket"):
            word_counts = result[statement_col].astype(str).apply(lambda t: len(t.split()))
            buckets = word_counts.apply(lambda n: _len_bucket(n, statement_len_bins))
            result["fe_speaker_len_bucket"] = (
                result[speaker_col].astype(str) + "__" + buckets
            )

    # -----------------------------------------------------------------------
    # AGGREGATE FEATURES
    # Per-group statistics. Non-label aggregates carry no leakage risk.
    # Label-based aggregates MUST be computed inside CV folds.
    # -----------------------------------------------------------------------

    # fe_speaker_true_rate: empirical false-claim rate per speaker.
    # WARNING: leakage risk — compute ONLY on the training fold, then map values
    # to the validation/test fold using the training-fold means.
    # Depends on: speaker_col (speaker_clean) and label_col (set label_col='label')
    if add_speaker_true_rate:
        if label_col is None:
            raise ValueError(
                "add_speaker_true_rate=True requires label_col to be set "
                "(e.g. label_col='label'). Compute this ONLY inside CV folds."
            )
        if _check_col(result, speaker_col, "fe_speaker_true_rate") and \
           _check_col(result, label_col, "fe_speaker_true_rate"):
            rates = result.groupby(result[speaker_col])[label_col].mean()
            result["fe_speaker_true_rate"] = result[speaker_col].map(rates).fillna(0.5)
            _scalable.append("fe_speaker_true_rate")

    # fe_subject_true_rate: empirical false-claim rate per subject topic.
    # WARNING: leakage risk — same CV-fold constraint as fe_speaker_true_rate.
    # Depends on: subject_col (subject_clean) and label_col (set label_col='label')
    if add_subject_true_rate:
        if label_col is None:
            raise ValueError(
                "add_subject_true_rate=True requires label_col to be set. "
                "Compute this ONLY inside CV folds."
            )
        if _check_col(result, subject_col, "fe_subject_true_rate") and \
           _check_col(result, label_col, "fe_subject_true_rate"):
            rates = result.groupby(result[subject_col])[label_col].mean()
            result["fe_subject_true_rate"] = result[subject_col].map(rates).fillna(0.5)
            _scalable.append("fe_subject_true_rate")

    # fe_party_true_rate: empirical false-claim rate per party affiliation.
    # WARNING: leakage risk — same CV-fold constraint as fe_speaker_true_rate.
    # Depends on: party_col (party_affiliation_clean) and label_col (set label_col='label')
    if add_party_true_rate:
        if label_col is None:
            raise ValueError(
                "add_party_true_rate=True requires label_col to be set. "
                "Compute this ONLY inside CV folds."
            )
        if _check_col(result, party_col, "fe_party_true_rate") and \
           _check_col(result, label_col, "fe_party_true_rate"):
            rates = result.groupby(result[party_col])[label_col].mean()
            result["fe_party_true_rate"] = result[party_col].map(rates).fillna(0.5)
            _scalable.append("fe_party_true_rate")

    # fe_speaker_avg_statement_len: mean word count of statements per speaker.
    # No leakage risk (does not use the label).
    # Depends on: speaker_col (speaker_clean) and statement_col (statement_clean)
    if add_speaker_avg_statement_len:
        if _check_col(result, speaker_col, "fe_speaker_avg_statement_len") and \
           _check_col(result, statement_col, "fe_speaker_avg_statement_len"):
            word_counts = result[statement_col].astype(str).apply(lambda t: len(t.split()))
            avg_lens = result[speaker_col].map(
                result.assign(_wc=word_counts).groupby(speaker_col)["_wc"].mean()
            )
            result["fe_speaker_avg_statement_len"] = avg_lens.fillna(word_counts.mean())
            _scalable.append("fe_speaker_avg_statement_len")

    # fe_subject_avg_statement_len: mean word count of statements per subject.
    # No leakage risk.
    # Depends on: subject_col (subject_clean) and statement_col (statement_clean)
    if add_subject_avg_statement_len:
        if _check_col(result, subject_col, "fe_subject_avg_statement_len") and \
           _check_col(result, statement_col, "fe_subject_avg_statement_len"):
            word_counts = result[statement_col].astype(str).apply(lambda t: len(t.split()))
            avg_lens = result[subject_col].map(
                result.assign(_wc=word_counts).groupby(subject_col)["_wc"].mean()
            )
            result["fe_subject_avg_statement_len"] = avg_lens.fillna(word_counts.mean())
            _scalable.append("fe_subject_avg_statement_len")

    # fe_speaker_avg_punctuation: mean punctuation-character density per speaker.
    # No leakage risk. High punctuation may indicate excited or informal style.
    # Depends on: speaker_col (speaker_clean) and statement_col (statement_clean)
    if add_speaker_avg_punctuation:
        if _check_col(result, speaker_col, "fe_speaker_avg_punctuation") and \
           _check_col(result, statement_col, "fe_speaker_avg_punctuation"):
            def _punct_density(text: str) -> float:
                text = str(text)
                punct = sum(1 for c in text if c in '.,!?;:"\'-()[]{}')
                return punct / max(len(text), 1)
            densities = result[statement_col].apply(_punct_density)
            result["fe_speaker_avg_punctuation"] = result[speaker_col].map(
                result.assign(_pd=densities).groupby(speaker_col)["_pd"].mean()
            ).fillna(densities.mean())
            _scalable.append("fe_speaker_avg_punctuation")

    # fe_speaker_avg_number_ratio: mean digit-character ratio per speaker.
    # No leakage risk. Speakers who cite many numbers may be more specific.
    # Depends on: speaker_col (speaker_clean) and statement_col (statement_clean)
    if add_speaker_avg_number_ratio:
        if _check_col(result, speaker_col, "fe_speaker_avg_number_ratio") and \
           _check_col(result, statement_col, "fe_speaker_avg_number_ratio"):
            def _digit_ratio(text: str) -> float:
                text = str(text)
                return sum(c.isdigit() for c in text) / max(len(text), 1)
            ratios = result[statement_col].apply(_digit_ratio)
            result["fe_speaker_avg_number_ratio"] = result[speaker_col].map(
                result.assign(_dr=ratios).groupby(speaker_col)["_dr"].mean()
            ).fillna(ratios.mean())
            _scalable.append("fe_speaker_avg_number_ratio")

    # -----------------------------------------------------------------------
    # TEXT-STYLE FEATURES
    # Derived from the cleaned statement text. No leakage risk.
    # All depend on: statement_col (statement_clean)
    # -----------------------------------------------------------------------

    if any([add_negation_count, add_hedge_count, add_absolutist_count,
            add_numeral_count, add_readability, add_sentiment]):
        stmt_ok = _check_col(result, statement_col, "text-style features")
    else:
        stmt_ok = False

    # fe_negation_count: number of negation words in the statement.
    # Fake claims sometimes use more negations to create doubt.
    # Depends on: statement_col (statement_clean)
    if add_negation_count and stmt_ok:
        result["fe_negation_count"] = result[statement_col].apply(
            lambda t: _count_from_set(_word_tokens(t), _NEGATION_WORDS)
        )
        _scalable.append("fe_negation_count")

    # fe_hedge_count: number of hedge/uncertainty words in the statement.
    # Hedging language may indicate vague or unverifiable claims.
    # Depends on: statement_col (statement_clean)
    if add_hedge_count and stmt_ok:
        result["fe_hedge_count"] = result[statement_col].apply(
            lambda t: _count_from_set(_word_tokens(t), _HEDGE_WORDS)
        )
        _scalable.append("fe_hedge_count")

    # fe_absolutist_count: number of absolutist/extreme words in the statement.
    # Absolutist language ("everyone knows", "always", "never") correlates with
    # misleading or exaggerated claims.
    # Depends on: statement_col (statement_clean)
    if add_absolutist_count and stmt_ok:
        result["fe_absolutist_count"] = result[statement_col].apply(
            lambda t: _count_from_set(_word_tokens(t), _ABSOLUTIST_WORDS)
        )
        _scalable.append("fe_absolutist_count")

    # fe_numeral_count: number of digit sequences in the statement.
    # Claims with specific numbers are more verifiable (and sometimes more false).
    # Depends on: statement_col (statement_clean)
    if add_numeral_count and stmt_ok:
        result["fe_numeral_count"] = result[statement_col].apply(
            lambda t: len(re.findall(r"\d+", str(t)))
        )
        _scalable.append("fe_numeral_count")

    # fe_proper_noun_count: heuristic count of capitalized non-sentence-start words.
    # Uses statement_original_col (original case) for accuracy.
    # Depends on: statement_original_col (statement_original — always produced by statement_ds)
    if add_proper_noun_count:
        if _check_col(result, statement_original_col, "fe_proper_noun_count"):
            result["fe_proper_noun_count"] = result[statement_original_col].apply(
                _proper_noun_heuristic
            )
            _scalable.append("fe_proper_noun_count")

    # fe_readability: Flesch Reading Ease approximation (no external library).
    # Higher score = easier to read. Complex statements may signal more nuanced claims.
    # Score range: roughly 0 (very hard) to 100 (very easy).
    # Depends on: statement_col (statement_clean)
    if add_readability and stmt_ok:
        result["fe_readability"] = result[statement_col].apply(_flesch_reading_ease)
        _scalable.append("fe_readability")

    # fe_sentiment_polarity / fe_sentiment_subjectivity: TextBlob-based sentiment.
    # Polarity: -1 (very negative) to +1 (very positive).
    # Subjectivity: 0 (objective) to 1 (subjective).
    # Subjective language may correlate with opinion-based or misleading claims.
    # Depends on: statement_col (statement_clean)
    # Requires: pip install textblob
    if add_sentiment and stmt_ok:
        try:
            from textblob import TextBlob  # type: ignore
            result["fe_sentiment_polarity"] = result[statement_col].apply(
                lambda t: TextBlob(str(t)).sentiment.polarity
            )
            result["fe_sentiment_subjectivity"] = result[statement_col].apply(
                lambda t: TextBlob(str(t)).sentiment.subjectivity
            )
            _scalable += ["fe_sentiment_polarity", "fe_sentiment_subjectivity"]
        except ImportError:
            warnings.warn(
                "add_sentiment=True requires TextBlob: pip install textblob. "
                "Skipping fe_sentiment_polarity and fe_sentiment_subjectivity."
            )

    if scale != 'none':
        for col in _scalable:
            if col in result.columns:
                result[col] = _scale_col(result[col], scale)

    if verbose:
        fe_cols = [c for c in result.columns if c.startswith("fe_")]
        print(f"feature_engineering: added {len(fe_cols)} column(s): {fe_cols}")

    return result


# ---------------------------------------------------------------------------
# Preset convenience functions
# ---------------------------------------------------------------------------
def preprocess_feature_engineering_none(df: pd.DataFrame) -> pd.DataFrame:
    """Return df unchanged (no features added)."""
    return df.copy()


def preprocess_feature_engineering_interactions(df: pd.DataFrame) -> pd.DataFrame:
    opts = PRESETS["interactions"]
    return preprocess_feature_engineering(df=df, **{
        k: v for k, v in opts.__dict__.items()
    })


def preprocess_feature_engineering_text(df: pd.DataFrame) -> pd.DataFrame:
    opts = PRESETS["text"]
    return preprocess_feature_engineering(df=df, **{
        k: v for k, v in opts.__dict__.items()
    })


def preprocess_feature_engineering_expanded(df: pd.DataFrame) -> pd.DataFrame:
    opts = PRESETS["expanded"]
    return preprocess_feature_engineering(df=df, **{
        k: v for k, v in opts.__dict__.items()
    })


__all__ = [
    "DEFAULT_STATEMENT_COL",
    "DEFAULT_STATEMENT_ORIGINAL_COL",
    "DEFAULT_SPEAKER_COL",
    "DEFAULT_SUBJECT_COL",
    "DEFAULT_PARTY_COL",
    "DEFAULT_SPEAKER_JOB_COL",
    "DEFAULT_STATE_COL",
    "DEFAULT_SCALE",
    "FeatureEngineeringOptions",
    "PRESETS",
    "preprocess_feature_engineering",
    "preprocess_feature_engineering_none",
    "preprocess_feature_engineering_interactions",
    "preprocess_feature_engineering_text",
    "preprocess_feature_engineering_expanded",
]
