from datetime import datetime
import importlib.util
from pathlib import Path
import sys
from time import time

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'data' / 'train.csv').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError('Could not locate the project root from the current working directory.')

project_root = find_project_root(Path.cwd())
src_path = project_root / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import OneStepOptions, preprocess_one_step
from submit.save_model import save_model

print(f'Project root: {project_root}')

df = None

if 'df' not in globals() or not isinstance(df, pd.DataFrame):
    df = pd.read_csv(project_root / 'data' / 'train.csv')

# print(type(df))
# print(df.shape)
# df.head()


# ============================================================
# LABEL OPTIONS  (src/preprocessing/label.py)
# Always produces: label (binary 0/1 column)
# ============================================================

# 'skip' -- keep the label column as-is (for training)
# 'drop' -- remove the label column (for inference / test set)
label_option = 'skip'
label_source_col = 'label'


# ============================================================
# ID OPTIONS  (src/preprocessing/id.py)
# ============================================================

# 'drop'        -- remove the ID column entirely
# 'hash_bucket' -- replace IDs with a hash-bucket integer
id_option = 'drop'


# ============================================================
# SUBJECT OPTIONS  (src/preprocessing/subject.py)
# Always produces: subject_clean
# ============================================================

subject_source_col = 'subject'
subject_keep_original = False               # keep raw subject column
subject_clean_text = True                   # lowercase + strip + collapse whitespace
subject_normalize_separators = True         # normalize | ; , & to single space
subject_split_topics = True                 # split multi-topic strings into a list
subject_primary_strategy = 'most_frequent'  # 'first' | 'most_frequent' -- pick primary topic
subject_rare_threshold = 10                 # subjects with < N occurrences are 'rare'
subject_rare_label = 'other'               # label applied to grouped rare subjects
subject_max_topics_for_primary = None       # cap number of topics considered
subject_multi_topic_label = 'multi-topic'  # label when > 1 topic and not capped

# Additional subject feature columns (all False by default)
subject_add_length_features = False         # subject_length, subject_token_count
subject_add_topic_list = False              # subject_topics -- list of individual topics
subject_add_topic_count = True              # subject_topic_count -- number of topics
subject_add_multiple_topics_flag = False    # subject_has_multiple_topics -- 0/1 flag
subject_add_primary = True                  # subject_primary -- single primary topic string
subject_add_grouped_primary = True          # subject_grouped -- primary with rares collapsed; requires add_primary
subject_group_rare = True                   # collapse rare subjects; requires add_grouped_primary
subject_add_subject_frequency = False       # subject_frequency -- count of primary topic; requires add_primary
subject_add_subject_is_rare = True          # subject_is_rare -- 0/1 flag; requires add_primary
subject_add_subject_primary_true_rate = False  # subject_primary_true_rate -- leakage risk -- CV folds only
subject_label_col = None                    # label column name; required for primary_true_rate

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: topic_count, length, token_count, frequency. NOT binary flags.
subject_scale = 'standardize'
subject_verbose = False


# ============================================================
# STATEMENT OPTIONS  (src/preprocessing/statement_ds.py)
# Always produces: statement_original, statement_clean
# ============================================================

statement_source_col = 'statement'
statement_original_output_col = 'statement_original'  # always produced; holds pre-clean text
statement_output_col = 'statement_clean'
statement_keep_original = False             # keep the raw source column

# Text normalization
statement_lower = True                      # convert to lowercase
statement_remove_html = True                # strip HTML tags
statement_remove_urls = True                # remove http:// and www. links
statement_replace_numbers = False           # replace numeric tokens with statement_number_token
statement_number_token = '<NUM>'            # token used when replace_numbers=True
statement_stopword_removal = False          # remove common English stopwords
statement_keep_negations = True             # preserve negation words even when removing stopwords
statement_remove_punctuation = False        # strip all punctuation characters
statement_stemmer = 'none'                # 'none' | 'porter' | 'snowball' -- requires NLTK
statement_lemmatizer = 'wordnet'               # 'none' | 'wordnet' -- requires NLTK
statement_repair_polluted_statements = True # fix malformed/polluted statement text

# Optional feature columns (all False by default)
statement_add_rare_token_features = False   # rare_token_count, avg_token_freq
statement_rare_token_threshold = 1          # tokens with <= N occurrences are 'rare'
statement_token_freqs = None                # pre-computed token frequency dict
statement_add_spelling_errors = False       # spelling_err_count (approximate)
statement_add_lexical_features = True       # char_len, word_count, upper_ratio, counts, digit_ratio
statement_add_pollution_features = False    # tab_count, newline_count, row_spillover_flag
statement_add_ner_features = False          # entity counts (PERSON, ORG, GPE, DATE, NUM, OTHER)
statement_ner_model = 'en_core_web_sm'      # spaCy model; requires: pip install spacy

# Vectorizer ('none' | 'tfidf' | 'bigram' | 'binary' | 'embeddings')
statement_vectorizer_type = 'bigram'
statement_vectorizer_max_features = 10000    # None = no vocab limit
statement_vectorizer_min_df = 2             # minimum document frequency
statement_vectorizer_max_df = 0.9           # maximum document frequency ratio
statement_embedding_model = 'all-MiniLM-L6-v2'  # sentence-transformers model
statement_fitted_vectorizer = None          # pre-fitted vectorizer for test-set

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: char_len, word_count, ratios, counts (lexical, rare token, spelling, NER).
# NOT binary flags (row_spillover_flag) or vec_* columns (already normalized).
statement_scale = 'standardize'
statement_verbose = False


# ============================================================
# SPEAKER OPTIONS  (src/preprocessing/speaker.py)
# Always produces: speaker_clean
# ============================================================

speaker_source_col = 'speaker'
speaker_keep_original = False               # keep raw speaker column
speaker_clean_text = True                   # lowercase + strip whitespace
speaker_normalize_separators = True         # normalize separator characters

# Rare-grouping
speaker_group_rare = True                   # requires add_grouped_speaker
speaker_rare_threshold = 5                  # speakers with < N occurrences are 'rare'
speaker_rare_label = 'other'               # label for rare speakers

# Additional speaker feature columns (all False by default)
speaker_add_length_features = False         # speaker_char_len, speaker_token_count
speaker_add_frequency = True                # speaker_frequency, speaker_frequency_pct
speaker_add_is_rare = True                  # speaker_is_rare -- 0/1 flag
speaker_add_grouped_speaker = True          # speaker_grouped -- rare names collapsed; requires group_rare
speaker_add_title_flag = True               # speaker_has_title -- 0/1 (senator, doctor, CEO, etc.)
speaker_add_comma_flag = True               # speaker_has_comma -- 0/1
speaker_add_period_flag = False             # speaker_has_period -- 0/1 (e.g. 'J. Biden')
speaker_add_token_count = False             # speaker_token_count
speaker_add_speaker_primary_true_rate = False  # speaker_primary_true_rate -- leakage risk -- CV folds only
speaker_label_col = None                    # label column; required for primary_true_rate

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
speaker_scale = 'standardize'
speaker_verbose = False


# ============================================================
# SPEAKER JOB OPTIONS  (src/preprocessing/speaker_job.py)
# Always produces: speaker_job_clean
# ============================================================

speaker_job_source_col = 'speaker_job'
speaker_job_keep_original = False           # keep raw speaker_job column
speaker_job_clean_text = True               # lowercase + strip + remove HTML/URLs
speaker_job_normalize_separators = True     # normalize | ; , / & to space

# Rare-grouping
speaker_job_group_rare = False              # requires add_grouped_job
speaker_job_rare_threshold = 5              # jobs with < N occurrences are 'rare'
speaker_job_rare_label = 'other'           # label for rare jobs

# Additional speaker-job feature columns (all False by default)
speaker_job_add_length_features = True      # speaker_job_char_len, speaker_job_token_count
speaker_job_add_frequency = False           # speaker_job_frequency, frequency_pct
speaker_job_add_is_rare = False             # speaker_job_is_rare -- 0/1 flag
speaker_job_add_grouped_job = False         # speaker_job_grouped -- rare jobs collapsed; requires group_rare
speaker_job_add_title_flag = True           # speaker_job_has_title -- 0/1 (senator, CEO, professor)
speaker_job_add_comma_flag = True           # speaker_job_has_comma -- 0/1
speaker_job_add_slash_flag = False          # speaker_job_has_slash -- 0/1
speaker_job_add_ampersand_flag = False      # speaker_job_has_ampersand -- 0/1
speaker_job_add_token_count = False         # speaker_job_token_count
speaker_job_add_job_primary_true_rate = False  # speaker_job_primary_true_rate -- leakage risk -- CV folds only
speaker_job_job_label_col = None            # label column; required for primary_true_rate

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: char_len, token_count, frequency, frequency_pct. NOT binary flags.
speaker_job_scale = 'none'
speaker_job_verbose = False


# ============================================================
# PARTY AFFILIATION OPTIONS  (src/preprocessing/party_affiliation.py)
# Always produces: party_affiliation_clean
# ============================================================

party_affiliation_source_col = 'party_affiliation'
party_affiliation_keep_original = False     # keep raw party_affiliation column
party_affiliation_clean_text = True         # lowercase + strip + remove HTML

# Rare-grouping
party_affiliation_group_rare = True         # requires add_grouped_party
party_affiliation_rare_threshold = 5        # parties with < N occurrences are 'rare'
party_affiliation_rare_label = 'other'     # label for rare parties

# Additional party feature columns (all False by default)
party_affiliation_add_length_features = True        # party_affiliation_char_len
party_affiliation_add_frequency = True              # party_affiliation_frequency, frequency_pct
party_affiliation_add_is_rare = True                # party_affiliation_is_rare -- 0/1 flag
party_affiliation_add_grouped_party = True          # party_affiliation_grouped -- rare parties collapsed; requires group_rare
party_affiliation_add_slash_flag = True             # party_affiliation_has_slash -- 0/1
party_affiliation_add_ampersand_flag = True         # party_affiliation_has_ampersand -- 0/1
party_affiliation_add_comma_flag = False            # party_affiliation_has_comma -- 0/1
party_affiliation_add_parentheses_flag = True       # party_affiliation_has_parentheses -- 0/1
party_affiliation_add_token_count = True            # party_affiliation_token_count
party_affiliation_add_is_major_party = True         # party_affiliation_is_major_party -- 0/1 (democrat/republican)
party_affiliation_add_is_institutional = True       # party_affiliation_is_institutional -- 0/1 (govt bodies)
party_affiliation_add_party_primary_true_rate = False  # leakage risk -- CV folds only
party_affiliation_party_label_col = None            # label column; required for primary_true_rate

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
party_affiliation_scale = 'standardize'
party_affiliation_verbose = False


# ============================================================
# STATE INFO OPTIONS  (src/preprocessing/state.py)
# Always produces: state_info_clean
# ============================================================

state_source_col = 'state_info'
state_drop = False                          # True -- drop state_info entirely, add no features
state_keep_original = False                 # keep raw state_info column
state_clean_text = True                     # lowercase + strip + remove HTML
state_normalize_state = True                # expand 2-letter state codes to full names

# Rare-grouping
state_group_rare = True                     # requires add_grouped_state
state_rare_threshold = 5                    # states with < N occurrences are 'rare'
state_rare_label = 'other'                 # label for rare states

# Additional state feature columns (all False by default)
state_add_is_us_state = True                # state_info_is_us_state -- 0/1 flag
state_add_frequency = False                 # state_info_frequency, state_info_frequency_pct
state_add_is_rare = False                   # state_info_is_rare -- 0/1 flag
state_add_grouped_state = True              # state_info_grouped -- rare states collapsed; requires group_rare
state_add_length_features = False           # state_info_char_len
state_add_token_count = True                # state_info_token_count
state_add_has_us_words = False              # state_info_has_us_words -- 0/1 ('us', 'usa', 'united states')
state_add_us_region = True                  # state_info_us_region -- 'northeast'|'south'|'midwest'|'west'

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
state_scale = 'none'
state_verbose = False



# ============================================================
# FEATURE ENGINEERING OPTIONS  (src/preprocessing/feature_engineering.py)
#
# Cross-column features computed after all per-column modules finish.
# All flags default to False. All output columns are prefixed fe_.
#
# Three families:
#   1. Interaction â€” combine two cleaned columns into a joint category key
#   2. Aggregate   â€” per-group statistics (mean label, mean length, etc.)
#   3. Text-style  â€” linguistic signals from the statement text
# ============================================================

# Source column names. These match the *_clean outputs of the preprocessing
# modules and are always present after preprocess_one_step runs.
# Only change these if you renamed the upstream output columns.
fe_statement_col = 'statement_clean'
fe_statement_original_col = 'statement_original'   # used for proper-noun heuristic
fe_speaker_col = 'speaker_clean'
fe_subject_col = 'subject_clean'
fe_party_col = 'party_affiliation_clean'
fe_speaker_job_col = 'speaker_job_clean'
fe_state_col = 'state_info_clean'

# Label column â€” required only for the *_true_rate aggregate features.
# Set to 'label' when enabling those features inside CV folds.
fe_label_col = None


# ------------------------------------------------------------
# 1. INTERACTION FEATURES
# Concatenate two cleaned columns into a joint categorical key.
# Output format: "<value_a>__<value_b>" (double underscore separator).
# Example: "barack obama__health care" for fe_speaker_subject.
# No leakage risk. All depend only on *_clean columns.
# ------------------------------------------------------------

# fe_speaker_subject: do certain speakers make false claims on specific topics?
# Depends on: fe_speaker_col and fe_subject_col
fe_add_speaker_subject = False

# fe_speaker_party: does a speaker's false-claim rate vary by party affiliation?
# Depends on: fe_speaker_col and fe_party_col
fe_add_speaker_party = False

# fe_subject_party: do topics have different credibility patterns by party?
# Depends on: fe_subject_col and fe_party_col
fe_add_subject_party = False

# fe_speaker_job_subject: does a speaker's role interact with topic credibility?
# Depends on: fe_speaker_job_col and fe_subject_col
fe_add_speaker_job_subject = False

# fe_state_party: captures regional political credibility patterns.
# Depends on: fe_state_col and fe_party_col
fe_add_state_party = False

# fe_speaker_len_bucket: speaker Ã— statement length bucket (short/medium/long).
# Checks if certain speakers tend to make longer or shorter claims differently.
# Depends on: fe_speaker_col and fe_statement_col
fe_add_speaker_statement_len_bucket = False
# Word-count boundaries for short/medium/long buckets: short < bins[0], long > bins[1].
fe_statement_len_bins = (50, 150)


# ------------------------------------------------------------
# 2. AGGREGATE FEATURES
#
# *_true_rate features: WARNING â€” leakage risk.
# These compute the mean label per group on the CURRENT dataframe.
# Only enable them when calling preprocess_one_step on a training fold,
# then map the computed means to validation/test rows manually.
# Set fe_label_col='label' when enabling these.
#
# Non-label aggregates (avg_statement_len, avg_punctuation, avg_number_ratio)
# have no leakage risk.
# ------------------------------------------------------------

# fe_speaker_true_rate: mean label (false-claim rate) per speaker.
# WARNING: leakage risk â€” compute ONLY inside CV training folds!
# Requires fe_label_col to be set.
fe_add_speaker_true_rate = False

# fe_subject_true_rate: mean label per subject topic.
# WARNING: leakage risk â€” compute ONLY inside CV training folds!
# Requires fe_label_col to be set.
fe_add_subject_true_rate = False

# fe_party_true_rate: mean label per party affiliation.
# WARNING: leakage risk â€” compute ONLY inside CV training folds!
# Requires fe_label_col to be set.
fe_add_party_true_rate = False

# fe_speaker_avg_statement_len: mean word count of statements per speaker.
# No leakage risk.
# Depends on: fe_speaker_col and fe_statement_col
fe_add_speaker_avg_statement_len = False

# fe_subject_avg_statement_len: mean word count of statements per subject.
# No leakage risk.
# Depends on: fe_subject_col and fe_statement_col
fe_add_subject_avg_statement_len = False

# fe_speaker_avg_punctuation: mean punctuation-character density per speaker.
# High punctuation may indicate excited or informal speaking style.
# No leakage risk.
# Depends on: fe_speaker_col and fe_statement_col
fe_add_speaker_avg_punctuation = False

# fe_speaker_avg_number_ratio: mean digit-character ratio per speaker.
# Speakers who cite many numbers may be more specific (or more misleading).
# No leakage risk.
# Depends on: fe_speaker_col and fe_statement_col
fe_add_speaker_avg_number_ratio = False


# ------------------------------------------------------------
# 3. TEXT-STYLE FEATURES
# All derived from fe_statement_col (statement_clean by default).
# No leakage risk.
# ------------------------------------------------------------

# fe_negation_count: count of negation words (not, never, no, cannot, etc.).
# Fake claims sometimes use more negations to create doubt.
# Depends on: fe_statement_col
fe_add_negation_count = True

# fe_hedge_count: count of hedge/uncertainty words (maybe, perhaps, might, etc.).
# Hedging language may indicate vague or unverifiable claims.
# Depends on: fe_statement_col
fe_add_hedge_count = True

# fe_absolutist_count: count of absolutist/extreme words (always, everyone, etc.).
# Absolutist language correlates with exaggerated or misleading claims.
# Depends on: fe_statement_col
fe_add_absolutist_count = True

# fe_numeral_count: count of digit sequences in the statement.
# Claims with specific numbers are more verifiable (and sometimes more false).
# Depends on: fe_statement_col
fe_add_numeral_count = True

# fe_proper_noun_count: heuristic count of capitalized non-sentence-start words.
# Uses fe_statement_original_col (statement_original) to preserve casing.
# Depends on: fe_statement_original_col
fe_add_proper_noun_count = False

# fe_readability: Flesch Reading Ease approximation (no external library needed).
# Higher = easier to read (~0â€“100 scale).
# Complex statements may signal more nuanced or technical claims.
# Depends on: fe_statement_col
fe_add_readability = True

# fe_sentiment_polarity + fe_sentiment_subjectivity: TextBlob-based sentiment.
#   Polarity:     -1 (very negative) to +1 (very positive)
#   Subjectivity:  0 (objective)     to  1 (subjective)
# Subjective language may correlate with opinion-based or misleading claims.
# Requires: pip install textblob
# Depends on: fe_statement_col
fe_add_sentiment = True

# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
# Scales: avg_statement_len, avg_punctuation, avg_number_ratio, true_rate features,
#   negation/hedge/absolutist/numeral/proper_noun counts, readability, sentiment.
# NOT: interaction string columns (fe_speaker_subject etc.) or fe_speaker_len_bucket.
fe_scale = 'standardize'

fe_verbose = False

# ============================================================
# Build FeatureEngineeringOptions inline inside OneStepOptions
# (these fe_* variables are passed as fe_* kwargs below)
# ============================================================

# ============================================================
# BUILD OneStepOptions FROM THE VARIABLES ABOVE
# ============================================================

options = OneStepOptions(
    # --- Label ---
    label_option=label_option,
    label_source_col=label_source_col,

    # --- ID ---
    id_option=id_option,

    # --- Subject ---
    subject_source_col=subject_source_col,
    subject_keep_original=subject_keep_original,
    subject_clean_text=subject_clean_text,
    subject_normalize_separators=subject_normalize_separators,
    subject_split_topics=subject_split_topics,
    subject_primary_strategy=subject_primary_strategy,
    subject_max_topics_for_primary=subject_max_topics_for_primary,
    subject_multi_topic_label=subject_multi_topic_label,
    subject_add_length_features=subject_add_length_features,
    subject_add_topic_list=subject_add_topic_list,
    subject_add_topic_count=subject_add_topic_count,
    subject_add_multiple_topics_flag=subject_add_multiple_topics_flag,
    subject_add_primary=subject_add_primary,
    subject_add_grouped_primary=subject_add_grouped_primary,
    subject_group_rare=subject_group_rare,
    subject_rare_threshold=subject_rare_threshold,
    subject_rare_label=subject_rare_label,
    subject_add_subject_frequency=subject_add_subject_frequency,
    subject_add_subject_is_rare=subject_add_subject_is_rare,
    subject_add_subject_primary_true_rate=subject_add_subject_primary_true_rate,
    subject_label_col=subject_label_col,
    subject_scale=subject_scale,
    subject_verbose=subject_verbose,

    # --- Statement ---
    statement_source_col=statement_source_col,
    statement_original_output_col=statement_original_output_col,
    statement_output_col=statement_output_col,
    statement_keep_original=statement_keep_original,
    statement_lower=statement_lower,
    statement_remove_html=statement_remove_html,
    statement_remove_urls=statement_remove_urls,
    statement_replace_numbers=statement_replace_numbers,
    statement_number_token=statement_number_token,
    statement_stopword_removal=statement_stopword_removal,
    statement_keep_negations=statement_keep_negations,
    statement_stemmer=statement_stemmer,
    statement_lemmatizer=statement_lemmatizer,
    statement_remove_punctuation=statement_remove_punctuation,
    statement_repair_polluted_statements=statement_repair_polluted_statements,
    statement_add_rare_token_features=statement_add_rare_token_features,
    statement_rare_token_threshold=statement_rare_token_threshold,
    statement_token_freqs=statement_token_freqs,
    statement_add_spelling_errors=statement_add_spelling_errors,
    statement_add_lexical_features=statement_add_lexical_features,
    statement_add_pollution_features=statement_add_pollution_features,
    statement_verbose=statement_verbose,
    statement_add_ner_features=statement_add_ner_features,
    statement_ner_model=statement_ner_model,
    statement_vectorizer_type=statement_vectorizer_type,
    statement_vectorizer_max_features=statement_vectorizer_max_features,
    statement_vectorizer_min_df=statement_vectorizer_min_df,
    statement_vectorizer_max_df=statement_vectorizer_max_df,
    statement_embedding_model=statement_embedding_model,
    statement_fitted_vectorizer=statement_fitted_vectorizer,
    statement_scale=statement_scale,

    # --- Speaker ---
    speaker_source_col=speaker_source_col,
    speaker_keep_original=speaker_keep_original,
    speaker_clean_text=speaker_clean_text,
    speaker_normalize_separators=speaker_normalize_separators,
    speaker_group_rare=speaker_group_rare,
    speaker_rare_threshold=speaker_rare_threshold,
    speaker_rare_label=speaker_rare_label,
    speaker_add_length_features=speaker_add_length_features,
    speaker_add_frequency=speaker_add_frequency,
    speaker_add_is_rare=speaker_add_is_rare,
    speaker_add_grouped_speaker=speaker_add_grouped_speaker,
    speaker_add_title_flag=speaker_add_title_flag,
    speaker_add_comma_flag=speaker_add_comma_flag,
    speaker_add_period_flag=speaker_add_period_flag,
    speaker_add_token_count=speaker_add_token_count,
    speaker_add_speaker_primary_true_rate=speaker_add_speaker_primary_true_rate,
    speaker_label_col=speaker_label_col,
    speaker_scale=speaker_scale,
    speaker_verbose=speaker_verbose,

    # --- Speaker job ---
    speaker_job_source_col=speaker_job_source_col,
    speaker_job_keep_original=speaker_job_keep_original,
    speaker_job_clean_text=speaker_job_clean_text,
    speaker_job_normalize_separators=speaker_job_normalize_separators,
    speaker_job_group_rare=speaker_job_group_rare,
    speaker_job_rare_threshold=speaker_job_rare_threshold,
    speaker_job_rare_label=speaker_job_rare_label,
    speaker_job_add_length_features=speaker_job_add_length_features,
    speaker_job_add_frequency=speaker_job_add_frequency,
    speaker_job_add_is_rare=speaker_job_add_is_rare,
    speaker_job_add_grouped_job=speaker_job_add_grouped_job,
    speaker_job_add_title_flag=speaker_job_add_title_flag,
    speaker_job_add_comma_flag=speaker_job_add_comma_flag,
    speaker_job_add_slash_flag=speaker_job_add_slash_flag,
    speaker_job_add_ampersand_flag=speaker_job_add_ampersand_flag,
    speaker_job_add_token_count=speaker_job_add_token_count,
    speaker_job_add_job_primary_true_rate=speaker_job_add_job_primary_true_rate,
    speaker_job_job_label_col=speaker_job_job_label_col,
    speaker_job_scale=speaker_job_scale,
    speaker_job_verbose=speaker_job_verbose,

    # --- Party affiliation ---
    party_affiliation_source_col=party_affiliation_source_col,
    party_affiliation_keep_original=party_affiliation_keep_original,
    party_affiliation_clean_text=party_affiliation_clean_text,
    party_affiliation_group_rare=party_affiliation_group_rare,
    party_affiliation_rare_threshold=party_affiliation_rare_threshold,
    party_affiliation_rare_label=party_affiliation_rare_label,
    party_affiliation_add_length_features=party_affiliation_add_length_features,
    party_affiliation_add_frequency=party_affiliation_add_frequency,
    party_affiliation_add_is_rare=party_affiliation_add_is_rare,
    party_affiliation_add_grouped_party=party_affiliation_add_grouped_party,
    party_affiliation_add_slash_flag=party_affiliation_add_slash_flag,
    party_affiliation_add_ampersand_flag=party_affiliation_add_ampersand_flag,
    party_affiliation_add_comma_flag=party_affiliation_add_comma_flag,
    party_affiliation_add_parentheses_flag=party_affiliation_add_parentheses_flag,
    party_affiliation_add_token_count=party_affiliation_add_token_count,
    party_affiliation_add_is_major_party=party_affiliation_add_is_major_party,
    party_affiliation_add_is_institutional=party_affiliation_add_is_institutional,
    party_affiliation_add_party_primary_true_rate=party_affiliation_add_party_primary_true_rate,
    party_affiliation_party_label_col=party_affiliation_party_label_col,
    party_affiliation_scale=party_affiliation_scale,
    party_affiliation_verbose=party_affiliation_verbose,

    # --- State info ---
    state_source_col=state_source_col,
    state_drop=state_drop,
    state_keep_original=state_keep_original,
    state_clean_text=state_clean_text,
    state_normalize_state=state_normalize_state,
    state_group_rare=state_group_rare,
    state_rare_threshold=state_rare_threshold,
    state_rare_label=state_rare_label,
    state_add_is_us_state=state_add_is_us_state,
    state_add_frequency=state_add_frequency,
    state_add_is_rare=state_add_is_rare,
    state_add_grouped_state=state_add_grouped_state,
    state_add_length_features=state_add_length_features,
    state_add_token_count=state_add_token_count,
    state_add_has_us_words=state_add_has_us_words,
    state_add_us_region=state_add_us_region,
    state_scale=state_scale,
    state_verbose=state_verbose,

    # --- Feature engineering ---
    fe_statement_col=fe_statement_col,
    fe_statement_original_col=fe_statement_original_col,
    fe_speaker_col=fe_speaker_col,
    fe_subject_col=fe_subject_col,
    fe_party_col=fe_party_col,
    fe_speaker_job_col=fe_speaker_job_col,
    fe_state_col=fe_state_col,
    fe_label_col=fe_label_col,
    fe_add_speaker_subject=fe_add_speaker_subject,
    fe_add_speaker_party=fe_add_speaker_party,
    fe_add_subject_party=fe_add_subject_party,
    fe_add_speaker_job_subject=fe_add_speaker_job_subject,
    fe_add_state_party=fe_add_state_party,
    fe_add_speaker_statement_len_bucket=fe_add_speaker_statement_len_bucket,
    fe_statement_len_bins=fe_statement_len_bins,
    fe_add_speaker_true_rate=fe_add_speaker_true_rate,
    fe_add_subject_true_rate=fe_add_subject_true_rate,
    fe_add_party_true_rate=fe_add_party_true_rate,
    fe_add_speaker_avg_statement_len=fe_add_speaker_avg_statement_len,
    fe_add_subject_avg_statement_len=fe_add_subject_avg_statement_len,
    fe_add_speaker_avg_punctuation=fe_add_speaker_avg_punctuation,
    fe_add_speaker_avg_number_ratio=fe_add_speaker_avg_number_ratio,
    fe_add_negation_count=fe_add_negation_count,
    fe_add_hedge_count=fe_add_hedge_count,
    fe_add_absolutist_count=fe_add_absolutist_count,
    fe_add_numeral_count=fe_add_numeral_count,
    fe_add_proper_noun_count=fe_add_proper_noun_count,
    fe_add_readability=fe_add_readability,
    fe_add_sentiment=fe_add_sentiment,
    fe_scale=fe_scale,
    fe_verbose=fe_verbose,
)

# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------
_script_start = time()
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
print(f"[SECTION] Running preprocessing  [{_now()}]")
_t0 = time()
df_processed = preprocess_one_step(df, options=options)
print(f"  Rows: {len(df_processed):,}  |  Total columns (all dtypes): {df_processed.shape[1]}  |  {time()-_t0:.1f}s")


# -----------------------------------------------------------------------------
# Feature matrix
# -----------------------------------------------------------------------------
print("[SECTION] Building feature matrix")
X = df_processed.drop(columns=["label"]).select_dtypes(exclude="object")
y = df_processed["label"]

tfidf_cols = [c for c in X.columns if c.startswith("vec_")]
other_cols  = [c for c in X.columns if not c.startswith("vec_")]
print(f"  TF-IDF features : {len(tfidf_cols)}")
print(f"  Other features  : {len(other_cols)}  →  {other_cols}")
print(f"  Total numeric   : {X.shape[1]}")
print(f"Target distribution:\n{y.value_counts(normalize=True).round(4)}\n")


# -----------------------------------------------------------------------------
# Train / holdout split
# -----------------------------------------------------------------------------
print(f"[SECTION] Building train/holdout split  [{_now()}]")
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Train/val: {X_trainval.shape[0]:,}   Holdout: {X_holdout.shape[0]:,}   CV folds: {skf.get_n_splits()}")


# -----------------------------------------------------------------------------
# Model hyperparameters
# -----------------------------------------------------------------------------
CLASS_WEIGHT      = {0: 1.42, 1: 0.77}
C_VALUE           = 1.0           # overwritten by HP search when enable_hp_search=True
PENALTY           = "l2"          # overwritten by HP search when enable_hp_search=True
MAX_ITER          = 1000
model_name        = "lr"
create_kaggle_csv = True

# "class_weight"      → pass CLASS_WEIGHT to LogisticRegression (no resampling)
# "oversample_reject" → upsample class 0 (true statements, minority) to match class 1
balance_strategy  = "class_weight"  # "none" | "class_weight" | "oversample_reject"

# Hyperparameter search — GridSearchCV over C × penalty before the main CV loop.
# Finds the best combo, then the main CV evaluates it in full detail.
enable_hp_search  = True
C_GRID            = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
PENALTY_GRID      = ["l1", "l2"]

# Threshold tuning — searches for the probability cutoff that maximises THRESHOLD_METRIC
# on out-of-fold (OOF) predictions from the CV loop (no holdout leakage).
THRESHOLD              = 0.5    # default; overwritten when tuning runs + overwrite_threshold=True
enable_threshold_tuning = True
overwrite_threshold     = True
THRESHOLD_METRIC        = "macro_f1"  # metric to maximise: "macro_f1" | "mcc" | "balanced_acc"


def rebalance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    if strategy in ("none", "class_weight"):
        return X_train, y_train

    frame = X_train.copy()
    frame["_label"] = y_train.values
    minority = frame[frame["_label"] == 0]  # true statements — minority (35%)
    majority = frame[frame["_label"] == 1]  # false statements — majority (65%)

    if strategy == "oversample_reject":
        minority = minority.sample(n=len(majority), replace=True, random_state=random_state)
    else:
        raise ValueError(f"Unknown balance_strategy: {strategy!r}")

    balanced = (
        pd.concat([minority, majority], ignore_index=True)
        .sample(frac=1.0, random_state=random_state)
        .reset_index(drop=True)
    )
    return balanced.drop(columns=["_label"]), balanced["_label"]


# class_weight arg passed to LR — None when resampling handles balance instead
_lr_class_weight = CLASS_WEIGHT if balance_strategy == "class_weight" else None


# -----------------------------------------------------------------------------
# W&B init
# -----------------------------------------------------------------------------
# Hyperparameter search (GridSearchCV)
# -----------------------------------------------------------------------------
if enable_hp_search:
    print(f"[SECTION] Hyperparameter search: {len(C_GRID)} C values × {len(PENALTY_GRID)} penalties  [{_now()}]")
    _t0 = time()

    _search = GridSearchCV(
        estimator=LogisticRegression(
            solver="liblinear",
            class_weight=_lr_class_weight,
            max_iter=MAX_ITER,
            random_state=42,
        ),
        param_grid={"C": C_GRID, "penalty": PENALTY_GRID},
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        n_jobs=-1,
        refit=False,
        verbose=0,
    )
    _search.fit(X_trainval, y_trainval)

    _results = (
        pd.DataFrame(_search.cv_results_)[["param_C", "param_penalty", "mean_test_score", "std_test_score"]]
        .sort_values("mean_test_score", ascending=False)
        .reset_index(drop=True)
    )
    print(f"\n  {'C':>8}  {'penalty':>8}  {'macro_f1':>10}  {'±':>8}")
    for _, row in _results.iterrows():
        best = (row["param_C"] == _search.best_params_["C"] and row["param_penalty"] == _search.best_params_["penalty"])
        print(f"  {row['param_C']:>8}  {row['param_penalty']:>8}  {row['mean_test_score']:>10.4f}  {row['std_test_score']:>8.4f}{'  ←' if best else ''}")

    C_VALUE = _search.best_params_["C"]
    PENALTY = _search.best_params_["penalty"]
    print(f"\n  Best: C={C_VALUE}, penalty={PENALTY}  (cv macro_f1={_search.best_score_:.4f})  [{time()-_t0:.1f}s]")
    _lr_class_weight = CLASS_WEIGHT if balance_strategy == "class_weight" else None


# -----------------------------------------------------------------------------
print("[SECTION] Initializing W&B run")
wandb.login()
run = wandb.init(
    project="truth-classifier-logistic-regression",
    config={
        "model":                   "LogisticRegression",
        "solver":                  "liblinear",
        "C":                       C_VALUE,
        "penalty":                 PENALTY,
        "max_iter":                MAX_ITER,
        "balance_strategy":        balance_strategy,
        "class_weight":            str(_lr_class_weight),
        "hp_search_enabled":       enable_hp_search,
        "C_grid":                  str(C_GRID) if enable_hp_search else "n/a",
        "penalty_grid":            str(PENALTY_GRID) if enable_hp_search else "n/a",
        "cv_folds":                skf.get_n_splits(),
        "n_trainval":              int(X_trainval.shape[0]),
        "n_holdout":               int(X_holdout.shape[0]),
        "n_features":              int(X.shape[1]),
        "n_tfidf":                 len(tfidf_cols),
        "n_other":                 len(other_cols),
        "vectorizer_type":         statement_vectorizer_type,
        "vectorizer_max_features": statement_vectorizer_max_features,
        "statement_stemmer":       statement_stemmer,
        "statement_add_lexical":   statement_add_lexical_features,
        "statement_scale":         statement_scale,
        "subject_primary_strategy": subject_primary_strategy,
        "subject_scale":           subject_scale,
        "speaker_scale":           speaker_scale,
        "party_scale":             party_affiliation_scale,
        "state_scale":             state_scale,
        "fe_readability":          fe_add_readability,
        "fe_sentiment":            fe_add_sentiment,
        "fe_negation":             fe_add_negation_count,
        "fe_hedge":                fe_add_hedge_count,
        "fe_absolutist":           fe_add_absolutist_count,
        "fe_numeral":              fe_add_numeral_count,
        "fe_scale":                fe_scale,
    },
)


# -----------------------------------------------------------------------------
# Cross-validation
# -----------------------------------------------------------------------------
print(f"[SECTION] Running cross-validation  [{_now()}]")
_cv_start = time()
cv_fold_metrics = []
oof_proba = np.zeros(len(X_trainval))
oof_true  = np.zeros(len(X_trainval), dtype=int)

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
    _fold_t = time()
    X_fold_train, y_fold_train = rebalance_training_data(
        X_trainval.iloc[train_idx], y_trainval.iloc[train_idx], balance_strategy
    )
    X_fold_val = X_trainval.iloc[val_idx]
    y_fold_val  = y_trainval.iloc[val_idx]

    fold_model = LogisticRegression(
        solver="liblinear",
        C=C_VALUE,
        penalty=PENALTY,
        class_weight=_lr_class_weight,
        max_iter=MAX_ITER,
        random_state=42,
    )
    fold_model.fit(X_fold_train, y_fold_train)

    y_fold_pred  = fold_model.predict(X_fold_val)
    y_fold_proba = fold_model.predict_proba(X_fold_val)[:, 1]

    oof_proba[val_idx] = y_fold_proba
    oof_true[val_idx]  = y_fold_val.values

    fold_metrics = {
        "fold":         fold_idx,
        "roc_auc":      roc_auc_score(y_fold_val, y_fold_proba),
        "pr_auc":       average_precision_score(y_fold_val, y_fold_proba),
        "macro_f1":     f1_score(y_fold_val, y_fold_pred, average="macro", zero_division=0),
        "f1":           f1_score(y_fold_val, y_fold_pred, zero_division=0),
        "precision":    precision_score(y_fold_val, y_fold_pred, zero_division=0),
        "recall":       recall_score(y_fold_val, y_fold_pred, zero_division=0),
        "accuracy":     accuracy_score(y_fold_val, y_fold_pred),
        "mcc":          matthews_corrcoef(y_fold_val, y_fold_pred),
        "balanced_acc": balanced_accuracy_score(y_fold_val, y_fold_pred),
    }
    cv_fold_metrics.append(fold_metrics)

    print(
        f"  Fold {fold_idx} | "
        f"ROC-AUC={fold_metrics['roc_auc']:.4f}  "
        f"Macro-F1={fold_metrics['macro_f1']:.4f}  "
        f"MCC={fold_metrics['mcc']:.4f}  "
        f"Bal-Acc={fold_metrics['balanced_acc']:.4f}  "
        f"({time()-_fold_t:.1f}s)"
    )
    wandb.log({
        "cv/fold":         fold_idx,
        "cv/roc_auc":      fold_metrics["roc_auc"],
        "cv/pr_auc":       fold_metrics["pr_auc"],
        "cv/macro_f1":     fold_metrics["macro_f1"],
        "cv/f1":           fold_metrics["f1"],
        "cv/precision":    fold_metrics["precision"],
        "cv/recall":       fold_metrics["recall"],
        "cv/accuracy":     fold_metrics["accuracy"],
        "cv/mcc":          fold_metrics["mcc"],
        "cv/balanced_acc": fold_metrics["balanced_acc"],
    })

_cv_keys = ["roc_auc", "pr_auc", "macro_f1", "f1", "precision", "recall", "accuracy", "mcc", "balanced_acc"]
cv_summary = {}
for k in _cv_keys:
    cv_summary[f"cv_mean_{k}"] = float(np.mean([m[k] for m in cv_fold_metrics]))
    cv_summary[f"cv_std_{k}"]  = float(np.std([m[k] for m in cv_fold_metrics]))

print(f"\n[SECTION] Cross-validation summary  [total CV: {time()-_cv_start:.1f}s]")
for k in _cv_keys:
    print(f"  {k}: {cv_summary[f'cv_mean_{k}']:.4f} ± {cv_summary[f'cv_std_{k}']:.4f}")

cv_table = wandb.Table(
    columns=["fold"] + _cv_keys,
    data=[[int(m["fold"])] + [float(m[k]) for k in _cv_keys] for m in cv_fold_metrics],
)
wandb.log({"cv/folds_table": cv_table, **cv_summary})


# -----------------------------------------------------------------------------
# Threshold tuning (OOF)
# -----------------------------------------------------------------------------
if enable_threshold_tuning:
    print(f"\n[SECTION] Threshold tuning on OOF predictions  [{_now()}]")

    _metric_fn = {
        "macro_f1":     lambda t, yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "mcc":          lambda t, yt, yp: matthews_corrcoef(yt, yp),
        "balanced_acc": lambda t, yt, yp: balanced_accuracy_score(yt, yp),
    }[THRESHOLD_METRIC]

    threshold_grid   = np.arange(0.20, 0.76, 0.02)
    threshold_scores = {}
    for t in threshold_grid:
        preds = (oof_proba >= t).astype(int)
        threshold_scores[round(float(t), 2)] = _metric_fn(t, oof_true, preds)

    best_threshold = max(threshold_scores, key=threshold_scores.get)
    best_score     = threshold_scores[best_threshold]

    print(f"  {'threshold':>10}   {THRESHOLD_METRIC}")
    for t, s in threshold_scores.items():
        marker = "  ←" if t == best_threshold else ""
        print(f"  {t:>10.2f}   {s:.4f}{marker}")
    print(f"\n  Best threshold: {best_threshold:.2f}  (OOF {THRESHOLD_METRIC}={best_score:.4f})")

    wandb.log({
        "threshold/best":       best_threshold,
        f"threshold/oof_{THRESHOLD_METRIC}": best_score,
        "threshold/grid": wandb.Table(
            columns=["threshold", THRESHOLD_METRIC],
            data=[[t, s] for t, s in threshold_scores.items()],
        ),
    })

    if overwrite_threshold:
        THRESHOLD = best_threshold
        print(f"  THRESHOLD updated: 0.50 → {THRESHOLD:.2f}")
    else:
        print(f"  THRESHOLD kept at {THRESHOLD:.2f} (overwrite_threshold=False)")


# -----------------------------------------------------------------------------
# Final fit on full trainval
# -----------------------------------------------------------------------------
print(f"[SECTION] Fitting final model on full train/val set  [{_now()}]")
_t0 = time()
X_fit, y_fit = rebalance_training_data(X_trainval, y_trainval, balance_strategy)
model = LogisticRegression(
    solver="liblinear",
    C=C_VALUE,
    penalty=PENALTY,
    class_weight=_lr_class_weight,
    max_iter=MAX_ITER,
    random_state=42,
)
model.fit(X_fit, y_fit)
print(f"  Done in {time()-_t0:.1f}s")


# -----------------------------------------------------------------------------
# Holdout evaluation
# -----------------------------------------------------------------------------
print(f"[SECTION] Evaluating on holdout set  [{_now()}]")
print(f"  Using threshold: {THRESHOLD:.2f}")
y_proba = model.predict_proba(X_holdout)[:, 1]
y_pred  = (y_proba >= THRESHOLD).astype(int)

holdout_metrics = {
    "roc_auc":      roc_auc_score(y_holdout, y_proba),
    "pr_auc":       average_precision_score(y_holdout, y_proba),
    "macro_f1":     f1_score(y_holdout, y_pred, average="macro", zero_division=0),
    "f1":           f1_score(y_holdout, y_pred, zero_division=0),
    "precision":    precision_score(y_holdout, y_pred, zero_division=0),
    "recall":       recall_score(y_holdout, y_pred, zero_division=0),
    "accuracy":     accuracy_score(y_holdout, y_pred),
    "mcc":          matthews_corrcoef(y_holdout, y_pred),
    "balanced_acc": balanced_accuracy_score(y_holdout, y_pred),
}
cm     = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True)

print("\nHoldout results:")
for name, value in holdout_metrics.items():
    print(f"  {name}: {value:.4f}")
print(f"\n{classification_report(y_holdout, y_pred)}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(y_holdout, y_proba)
prec_c, rec_c, _ = precision_recall_curve(y_holdout, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(fpr, tpr, label=f"ROC-AUC = {holdout_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set_title("ROC Curve (holdout)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(rec_c, prec_c, label=f"PR-AUC = {holdout_metrics['pr_auc']:.4f}")
axes[1].set_title("Precision-Recall Curve (holdout)")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()

im = axes[2].imshow(cm, interpolation="nearest", cmap="Blues")
axes[2].set_title("Confusion Matrix (holdout)")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["True (0)", "False (1)"])
axes[2].set_yticklabels(["True (0)", "False (1)"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
fig.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.show()


# -----------------------------------------------------------------------------
# W&B logging — holdout
# -----------------------------------------------------------------------------
print("[SECTION] Logging to W&B")
wandb.log({
    "holdout/roc_auc":      holdout_metrics["roc_auc"],
    "holdout/pr_auc":       holdout_metrics["pr_auc"],
    "holdout/macro_f1":     holdout_metrics["macro_f1"],
    "holdout/f1":           holdout_metrics["f1"],
    "holdout/precision":    holdout_metrics["precision"],
    "holdout/recall":       holdout_metrics["recall"],
    "holdout/accuracy":     holdout_metrics["accuracy"],
    "holdout/mcc":          holdout_metrics["mcc"],
    "holdout/balanced_acc": holdout_metrics["balanced_acc"],
    "holdout/tn":           int(cm[0, 0]),
    "holdout/fp":           int(cm[0, 1]),
    "holdout/fn":           int(cm[1, 0]),
    "holdout/tp":           int(cm[1, 1]),
    "roc_pr_curves":        wandb.Image(fig),
    "confusion_matrix":     wandb.plot.confusion_matrix(
        y_true=y_holdout.tolist(),
        preds=y_pred.tolist(),
        class_names=["True (0)", "False (1)"],
    ),
})

run.summary["holdout/macro_f1"] = holdout_metrics["macro_f1"]
run.summary["holdout/roc_auc"]  = holdout_metrics["roc_auc"]
run.summary["cv_mean_macro_f1"] = cv_summary["cv_mean_macro_f1"]

report_rows = [
    [label, float(v.get("precision", 0)), float(v.get("recall", 0)),
     float(v.get("f1-score", 0)), float(v.get("support", 0))]
    for label, v in report.items() if isinstance(v, dict)
]
wandb.log({"classification_report": wandb.Table(
    columns=["label", "precision", "recall", "f1_score", "support"],
    data=report_rows,
)})

print("[SECTION] Finishing W&B run")
run.finish()


# -----------------------------------------------------------------------------
# Save model
# -----------------------------------------------------------------------------
print("[SECTION] Saving model")
saved_paths = save_model(
    model_pipeline=model,
    preprocessing_options=options,
    feature_names=X_trainval.columns.tolist(),
    project_root=project_root,
    model_name=model_name,
)
print(f"Saved model artifacts: {saved_paths}")

_model_dir = project_root / "models" / model_name

# Save fitted vectorizer so kaggle-modulo.py can apply it to test data.
if statement_vectorizer_type != "none":
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    if statement_vectorizer_type == "tfidf":
        _vec = TfidfVectorizer(
            max_features=statement_vectorizer_max_features,
            min_df=statement_vectorizer_min_df,
            max_df=statement_vectorizer_max_df,
        )
    elif statement_vectorizer_type == "bigram":
        _vec = TfidfVectorizer(
            max_features=statement_vectorizer_max_features,
            min_df=statement_vectorizer_min_df,
            max_df=statement_vectorizer_max_df,
            ngram_range=(1, 2),
        )
    elif statement_vectorizer_type == "binary":
        _vec = CountVectorizer(
            max_features=statement_vectorizer_max_features,
            min_df=statement_vectorizer_min_df,
            max_df=statement_vectorizer_max_df,
            binary=True,
        )
    else:
        _vec = None

    if _vec is not None:
        _vec.fit(df_processed[statement_output_col])
        _vec_path = _model_dir / f"{model_name}-vectorizer.joblib"
        joblib.dump(_vec, _vec_path)
        print(f"  Vectorizer saved: {_vec_path}  (vocab size: {len(_vec.vocabulary_):,})")

# Save decision threshold.
_threshold_path = _model_dir / f"{model_name}-threshold.joblib"
joblib.dump(THRESHOLD, _threshold_path)
print(f"  Threshold saved: {_threshold_path}  (value: {THRESHOLD:.2f})")

print(f"\n[DONE] Total script time: {time()-_script_start:.1f}s  [{_now()}]")


# -----------------------------------------------------------------------------
# Kaggle submission CSV
# -----------------------------------------------------------------------------
if create_kaggle_csv:
    print(f"[SECTION] Creating Kaggle submission CSV from saved model  [{_now()}]")
    kaggle_module_path = project_root / "src" / "submit" / "kaggle-modulo.py"
    spec = importlib.util.spec_from_file_location("kaggle_modulo", kaggle_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kaggle module from: {kaggle_module_path}")

    kaggle_modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_modulo)

    submission_path = kaggle_modulo.generate_submission_csv(
        model_name=model_name,
        project_root=project_root,
        is_tree_model=False,
        verbose=False,
    )
    print(f"Kaggle submission generated: {submission_path}")

