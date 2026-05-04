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
from sklearn.ensemble import RandomForestClassifier
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
from scipy.stats import randint
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder


# ============================================================
# Project root + sys.path
# ============================================================

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'data' / 'train.csv').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError('Could not locate the project root.')

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

# 'drop' -- remove ID entirely; no sequential-ID proxy needed for this run
id_option = 'drop'


# ============================================================
# SUBJECT OPTIONS  (src/preprocessing/subject.py)
# Always produces: subject_clean
# ============================================================

subject_source_col = 'subject'
subject_keep_original = False
subject_clean_text = True
subject_normalize_separators = True
subject_split_topics = True
# most_frequent: picks the most common topic per row -- most stable across CV splits
subject_primary_strategy = 'most_frequent'
subject_rare_threshold = 10
subject_rare_label = 'other'
subject_max_topics_for_primary = None
subject_multi_topic_label = 'multi-topic'

subject_add_primary = True                      # subject_primary (string) -- prereq for grouped
subject_add_topic_count = True                  # subject_topic_count -- more topics = vaguer claim
subject_add_multiple_topics_flag = True         # subject_has_multiple_topics -- 0/1
subject_add_topic_list = False                  # list column -- not usable by sklearn trees
subject_add_length_features = True              # subject_length, subject_token_count
subject_add_grouped_primary = True              # subject_primary_grouped (string) -- lower cardinality
subject_group_rare = True                       # collapse rare subjects to rare_label
subject_add_subject_frequency = True            # subject_frequency (int) -- frequency encoding
subject_add_subject_is_rare = True              # subject_is_rare -- 0/1
subject_add_subject_primary_true_rate = False   # computed fold-safe in CV loop -- not here
subject_label_col = None

# Trees are invariant to monotone transforms -- scaling adds no value
subject_scale = 'none'
subject_verbose = False


# ============================================================
# STATEMENT OPTIONS  (src/preprocessing/statement_ds.py)
# Always produces: statement_original, statement_clean
# ============================================================

statement_source_col = 'statement'
statement_original_output_col = 'statement_original'
statement_output_col = 'statement_clean'
statement_keep_original = False

statement_lower = True
statement_remove_html = True
statement_remove_urls = True
statement_replace_numbers = False
statement_number_token = '<NUM>'
# Stopword removal is enabled here to clean the TF-IDF vocabulary — the initial
# run showed "the", "in", "of" in the top-30 features, consuming importance from
# real signals. keep_negations=True preserves "not", "never", "no", "n't" so the
# fe_negation_count and absolutist features computed from statement_clean are unaffected.
statement_stopword_removal = True
statement_keep_negations = True
statement_remove_punctuation = False
# Stemming and lemmatization compress vocabulary to help linear models;
# trees gain nothing and may lose useful token distinctions.
statement_stemmer = 'none'
statement_lemmatizer = 'none'
statement_repair_polluted_statements = True

statement_add_rare_token_features = True        # rare_token_count, avg_token_freq
statement_rare_token_threshold = 1
statement_token_freqs = None
statement_add_spelling_errors = True            # spelling_err_count -- proxy for informal writing
statement_add_lexical_features = True           # char_len, word_count, upper_ratio, punct counts
statement_add_pollution_features = True         # tab_count, newline_count, row_spillover_flag
statement_add_ner_features = False              # slow; set True to add PERSON/ORG/GPE/etc counts
statement_ner_model = 'en_core_web_sm'

# TF-IDF capped at 500 terms.
# Large sparse TF-IDF matrices are expensive for tree ensembles and add
# diminishing returns beyond ~500 top terms. Switch vectorizer_type to
# 'embeddings' for better semantic coverage (5-15x slower preprocessing).
statement_vectorizer_type = 'tfidf'
statement_vectorizer_max_features = 500
statement_vectorizer_min_df = 2
statement_vectorizer_max_df = 0.7   # initial run: 0.9 let "the","in","of" survive; 0.7 cuts them
statement_embedding_model = 'all-MiniLM-L6-v2'
statement_fitted_vectorizer = None

statement_scale = 'none'
statement_verbose = False


# ============================================================
# SPEAKER OPTIONS  (src/preprocessing/speaker.py)
# Always produces: speaker_clean
# ============================================================

speaker_source_col = 'speaker'
speaker_keep_original = False
speaker_clean_text = True
speaker_normalize_separators = True

speaker_group_rare = True
speaker_rare_threshold = 5
speaker_rare_label = 'other'

speaker_add_frequency = True                    # speaker_frequency, speaker_frequency_pct
speaker_add_is_rare = True                      # speaker_is_rare -- 0/1
speaker_add_grouped_speaker = True              # speaker_grouped (string) -- rare names collapsed
speaker_add_length_features = True              # speaker_char_len, speaker_token_count
speaker_add_title_flag = True                   # speaker_has_title -- 0/1
speaker_add_comma_flag = True                   # speaker_has_comma -- 0/1 (Last, First format)
speaker_add_period_flag = True                  # speaker_has_period -- 0/1 (J. Biden style)
speaker_add_token_count = False                 # already included via speaker_add_length_features
speaker_add_speaker_primary_true_rate = False   # computed fold-safe in CV loop
speaker_label_col = None

speaker_scale = 'none'
speaker_verbose = False


# ============================================================
# SPEAKER JOB OPTIONS  (src/preprocessing/speaker_job.py)
# Always produces: speaker_job_clean
# ============================================================

speaker_job_source_col = 'speaker_job'
speaker_job_keep_original = False
speaker_job_clean_text = True
speaker_job_normalize_separators = True

speaker_job_group_rare = True
speaker_job_rare_threshold = 5
speaker_job_rare_label = 'other'

speaker_job_add_frequency = True
speaker_job_add_is_rare = True
speaker_job_add_grouped_job = True              # speaker_job_grouped (string)
speaker_job_add_length_features = True
speaker_job_add_title_flag = True               # speaker_job_has_title -- strong institutional signal
speaker_job_add_comma_flag = True
speaker_job_add_slash_flag = True
speaker_job_add_ampersand_flag = True
speaker_job_add_token_count = False             # already included via speaker_job_add_length_features
speaker_job_add_job_primary_true_rate = False   # computed fold-safe in CV loop
speaker_job_job_label_col = None

speaker_job_scale = 'none'
speaker_job_verbose = False


# ============================================================
# PARTY AFFILIATION OPTIONS  (src/preprocessing/party_affiliation.py)
# Always produces: party_affiliation_clean
# ============================================================

party_affiliation_source_col = 'party_affiliation'
party_affiliation_keep_original = False
party_affiliation_clean_text = True

party_affiliation_group_rare = True
party_affiliation_rare_threshold = 5
party_affiliation_rare_label = 'other'

party_affiliation_add_frequency = True
party_affiliation_add_is_rare = True
party_affiliation_add_grouped_party = True      # party_affiliation_grouped (string)
party_affiliation_add_length_features = True
party_affiliation_add_token_count = True
party_affiliation_add_is_major_party = True     # 0/1 -- democrat or republican
party_affiliation_add_is_institutional = True   # 0/1 -- non-party institutional roles
party_affiliation_add_slash_flag = True
party_affiliation_add_ampersand_flag = True
party_affiliation_add_comma_flag = False
party_affiliation_add_parentheses_flag = True
party_affiliation_add_party_primary_true_rate = False  # computed fold-safe in CV loop
party_affiliation_party_label_col = None

party_affiliation_scale = 'none'
party_affiliation_verbose = False


# ============================================================
# STATE INFO OPTIONS  (src/preprocessing/state.py)
# Always produces: state_info_clean
# ============================================================

state_source_col = 'state_info'
state_drop = False
state_keep_original = False
state_clean_text = True
state_normalize_state = True

state_group_rare = True
state_rare_threshold = 5
state_rare_label = 'other'

state_add_is_us_state = True                    # state_info_is_us_state -- 0/1
state_add_frequency = True                      # state_info_frequency, frequency_pct
state_add_is_rare = True                        # state_info_is_rare -- 0/1
state_add_grouped_state = True                  # state_info_grouped (string)
state_add_has_us_words = True                   # state_info_has_us_words -- 0/1
state_add_us_region = True                      # state_info_us_region -- 4-value categorical string
state_add_length_features = False
state_add_token_count = True

state_scale = 'none'
state_verbose = False


# ============================================================
# FEATURE ENGINEERING OPTIONS  (src/preprocessing/feature_engineering.py)
#
# Three families:
#   1. Interaction -- joint categorical keys for direct tree splitting
#   2. Aggregate   -- per-group statistics; *_true_rate handled in CV loop
#   3. Text-style  -- linguistic signals from the statement
# ============================================================

fe_statement_col = 'statement_clean'
fe_statement_original_col = 'statement_original'
fe_speaker_col = 'speaker_clean'
fe_subject_col = 'subject_clean'
fe_party_col = 'party_affiliation_clean'
fe_speaker_job_col = 'speaker_job_clean'
fe_state_col = 'state_info_clean'
# True-rate aggregates are computed fold-safe in the CV loop; fe_label_col stays None here
fe_label_col = None

# 1. Interaction features -- each produces a joint string key, OrdinalEncoded below.
# Trees split on these directly: "does speaker X tend to lie on topic Y?"
fe_add_speaker_subject = True
fe_add_speaker_party = True
fe_add_subject_party = True
fe_add_speaker_job_subject = True
fe_add_state_party = True
fe_add_speaker_statement_len_bucket = True
fe_statement_len_bins = (50, 150)

# 2a. Target-rate aggregates: set False; computed fold-safe inside the CV loop
fe_add_speaker_true_rate = False
fe_add_subject_true_rate = False
fe_add_party_true_rate = False

# 2b. Non-leakage aggregates: safe to compute on the full dataset
fe_add_speaker_avg_statement_len = True
fe_add_subject_avg_statement_len = True
fe_add_speaker_avg_punctuation = True
fe_add_speaker_avg_number_ratio = True

# 3. Text-style features: cheap linguistic signals orthogonal to structured metadata
fe_add_negation_count = True
fe_add_hedge_count = True
fe_add_absolutist_count = True
fe_add_numeral_count = True
fe_add_proper_noun_count = True
fe_add_readability = True
fe_add_sentiment = True

fe_scale = 'none'
fe_verbose = False


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
    subject_add_primary=subject_add_primary,
    subject_add_topic_count=subject_add_topic_count,
    subject_add_multiple_topics_flag=subject_add_multiple_topics_flag,
    subject_add_topic_list=subject_add_topic_list,
    subject_add_length_features=subject_add_length_features,
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
    statement_add_ner_features=statement_add_ner_features,
    statement_ner_model=statement_ner_model,
    statement_vectorizer_type=statement_vectorizer_type,
    statement_vectorizer_max_features=statement_vectorizer_max_features,
    statement_vectorizer_min_df=statement_vectorizer_min_df,
    statement_vectorizer_max_df=statement_vectorizer_max_df,
    statement_embedding_model=statement_embedding_model,
    statement_fitted_vectorizer=statement_fitted_vectorizer,
    statement_scale=statement_scale,
    statement_verbose=statement_verbose,

    # --- Speaker ---
    speaker_source_col=speaker_source_col,
    speaker_keep_original=speaker_keep_original,
    speaker_clean_text=speaker_clean_text,
    speaker_normalize_separators=speaker_normalize_separators,
    speaker_group_rare=speaker_group_rare,
    speaker_rare_threshold=speaker_rare_threshold,
    speaker_rare_label=speaker_rare_label,
    speaker_add_frequency=speaker_add_frequency,
    speaker_add_is_rare=speaker_add_is_rare,
    speaker_add_grouped_speaker=speaker_add_grouped_speaker,
    speaker_add_length_features=speaker_add_length_features,
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
    speaker_job_add_frequency=speaker_job_add_frequency,
    speaker_job_add_is_rare=speaker_job_add_is_rare,
    speaker_job_add_grouped_job=speaker_job_add_grouped_job,
    speaker_job_add_length_features=speaker_job_add_length_features,
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
    party_affiliation_add_frequency=party_affiliation_add_frequency,
    party_affiliation_add_is_rare=party_affiliation_add_is_rare,
    party_affiliation_add_grouped_party=party_affiliation_add_grouped_party,
    party_affiliation_add_length_features=party_affiliation_add_length_features,
    party_affiliation_add_token_count=party_affiliation_add_token_count,
    party_affiliation_add_is_major_party=party_affiliation_add_is_major_party,
    party_affiliation_add_is_institutional=party_affiliation_add_is_institutional,
    party_affiliation_add_slash_flag=party_affiliation_add_slash_flag,
    party_affiliation_add_ampersand_flag=party_affiliation_add_ampersand_flag,
    party_affiliation_add_comma_flag=party_affiliation_add_comma_flag,
    party_affiliation_add_parentheses_flag=party_affiliation_add_parentheses_flag,
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
print(f"  Rows: {len(df_processed):,}  |  Total columns: {df_processed.shape[1]}  |  {time()-_t0:.1f}s")


# -----------------------------------------------------------------------------
# Categorical encoding
# -----------------------------------------------------------------------------
# After preprocessing, string columns fall into two groups:
#   - *_clean / *_original: raw processed text strings → dropped (not useful for trees)
#   - grouped categories, regions, interaction keys → OrdinalEncoded
#
# OrdinalEncoder assigns each string category an integer. Trees split on
# integer thresholds so this is the correct encoding for categoricals.
# OneHotEncoder is wrong here: interaction-key cardinality would be 1000+.
#
# The encoder is fit on the full dataset (all 8950 rows) before the
# train/holdout split — this is the same policy as TF-IDF in lr.py and
# causes only negligible leakage since no label information is used.
# -----------------------------------------------------------------------------
print("[SECTION] Categorical encoding")

_all_obj_cols = df_processed.select_dtypes(include="object").columns.tolist()
# Raw text columns: never useful as tree features.
# Drop anything ending in _clean or _original, PLUS any original source columns
# that survived preprocessing (e.g. 'statement' when keep_original=False but the
# column is not removed by the module). In the initial run 'statement' leaked into
# the encoder and was assigned near-unique integers — effectively a row ID.
_source_cols = {
    statement_source_col, speaker_source_col, subject_source_col,
    speaker_job_source_col, party_affiliation_source_col, state_source_col,
}
_text_cols = (
    {c for c in _all_obj_cols if c.endswith(("_clean", "_original"))}
    | (_source_cols & set(_all_obj_cols))
)
# Encodable categoricals: grouped values, regions, interaction keys
_cat_cols = [c for c in _all_obj_cols if c not in _text_cols and c != label_source_col]

print(f"  Text columns dropped    : {sorted(_text_cols)}")
print(f"  Categorical cols encoded: {_cat_cols}")

ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
_cat_encoded = pd.DataFrame(
    ordinal_enc.fit_transform(df_processed[_cat_cols]),
    columns=_cat_cols,
    index=df_processed.index,
)

# Full feature dataframe: numeric output columns + ordinal-encoded categoricals
df_features = pd.concat(
    [df_processed.select_dtypes(exclude="object"), _cat_encoded],
    axis=1,
)


# -----------------------------------------------------------------------------
# Feature matrix
# -----------------------------------------------------------------------------
print("[SECTION] Building feature matrix")
X = df_features.drop(columns=[label_source_col])
y = df_processed[label_source_col]

vec_cols   = [c for c in X.columns if "_vec_" in c or c.startswith("vec_")]
cat_cols_X = [c for c in X.columns if c in _cat_cols]
other_cols = [c for c in X.columns if c not in vec_cols and c not in cat_cols_X]

print(f"  Vectorizer features     : {len(vec_cols)}")
print(f"  Encoded cat features    : {len(cat_cols_X)}  →  {cat_cols_X}")
print(f"  Other numeric features  : {len(other_cols)}  →  {other_cols}")
print(f"  Total features          : {X.shape[1]}")
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
# Model configuration
# -----------------------------------------------------------------------------
# Dataset is imbalanced (35% true / 65% false); class_weight corrects for this
# without resampling by scaling the impurity criterion during tree growth.
CLASS_WEIGHT  = {0: 1.42, 1: 0.77}
N_ESTIMATORS  = 300
MAX_DEPTH     = None    # grow full trees; depth is controlled implicitly by min_samples_leaf
MIN_SAMPLES_LEAF = 1
THRESHOLD     = 0.5     # starting point; overwritten by threshold tuning when enabled
model_name    = "rfc"
create_kaggle_csv = True   # run kaggle_module_forTrees.py after saving to produce submission CSV

# Threshold tuning — searches the OOF probability predictions from the CV loop for
# the cutoff that maximises THRESHOLD_METRIC. No retraining needed; the OOF probas
# already collected give an unbiased estimate of out-of-sample performance per threshold.
enable_threshold_tuning = True
overwrite_threshold     = True    # set False to inspect the grid without updating THRESHOLD
THRESHOLD_METRIC        = "macro_f1"  # "macro_f1" | "mcc" | "balanced_acc"

# HP search — nested CV: each outer fold runs an inner RandomizedSearchCV to find the
# best max_features and min_samples_leaf. After all outer folds, the best params are
# aggregated (mode/median across folds) and used for the final fit on full trainval.
enable_hp_search  = True
N_ITER_SEARCH     = 20      # number of parameter combinations to try per outer fold
N_CV_INNER        = 3       # inner CV folds inside each RandomizedSearchCV
HP_SEARCH_METRIC  = "f1_macro"   # sklearn scoring string
param_dist = {
    "max_features":     [0.2, 0.3, 0.5, "sqrt", "log2"],
    "min_samples_leaf": randint(1, 8),   # 1–7 inclusive
    "n_estimators":     [200, 300, 500],
}

# Whether to compute true-rate features (speaker/subject/party historical false-claim rates)
# fold-safe inside the CV loop. Most predictive single signal in this dataset.
enable_true_rate_features = True
true_rate_fallback = 0.5    # assigned to groups unseen in the training fold or test set


# -----------------------------------------------------------------------------
# True-rate feature setup
# -----------------------------------------------------------------------------
# Maps feat_name → source column in df_processed (string categories for groupby).
# Placeholder values (true_rate_fallback) fill X_trainval / X_holdout now;
# real per-fold values are written inside the CV loop and the final fit.
_tr_group_cols: dict[str, str] = {}
if enable_true_rate_features:
    _candidates = {
        "fe_speaker_true_rate": ["speaker_grouped", "speaker_clean"],
        "fe_subject_true_rate": ["subject_primary_grouped", "subject_primary", "subject_clean"],
        "fe_party_true_rate":   ["party_affiliation_grouped", "party_affiliation_clean"],
    }
    for _feat, _src_cols in _candidates.items():
        for _col in _src_cols:
            if _col in df_processed.columns:
                _tr_group_cols[_feat] = _col
                break

    X_trainval = X_trainval.copy()
    X_holdout  = X_holdout.copy()
    for _feat in _tr_group_cols:
        X_trainval[_feat] = true_rate_fallback
        X_holdout[_feat]  = true_rate_fallback

    # String-category lookup tables aligned positionally to X_trainval / X_holdout.
    # We keep these as strings (not encoded ints) so the groupby mean is computed
    # over the correct groups.
    _grp_trainval = pd.DataFrame(
        {col: df_processed[col].loc[X_trainval.index].values for col in _tr_group_cols.values()}
    )
    _grp_trainval["_label"] = y_trainval.values

    _grp_holdout = pd.DataFrame(
        {col: df_processed[col].loc[X_holdout.index].values for col in _tr_group_cols.values()}
    )

    print(f"  True-rate features  : {list(_tr_group_cols.keys())}")
    print(f"  Source columns      : {list(_tr_group_cols.values())}")


# -----------------------------------------------------------------------------
# W&B init
# -----------------------------------------------------------------------------
print("[SECTION] Initializing W&B run")
wandb.login()
run = wandb.init(
    project="truth-classifier-random-forest",
    config={
        "model":                    "RandomForestClassifier",
        "n_estimators":             N_ESTIMATORS,
        "max_depth":                str(MAX_DEPTH),
        "min_samples_leaf":         MIN_SAMPLES_LEAF,
        "class_weight":             str(CLASS_WEIGHT),
        "threshold":                THRESHOLD,
        "cv_folds":                 skf.get_n_splits(),
        "n_trainval":               int(X_trainval.shape[0]),
        "n_holdout":                int(X_holdout.shape[0]),
        "n_features_total":         int(X.shape[1]),
        "n_vec_features":           len(vec_cols),
        "n_cat_features":           len(cat_cols_X),
        "n_other_features":         len(other_cols),
        "vectorizer_type":          statement_vectorizer_type,
        "vectorizer_max_features":  statement_vectorizer_max_features,
        "statement_add_lexical":    statement_add_lexical_features,
        "statement_add_pollution":  statement_add_pollution_features,
        "statement_add_spelling":   statement_add_spelling_errors,
        "statement_add_ner":        statement_add_ner_features,
        "subject_primary_strategy": subject_primary_strategy,
        "fe_interaction_keys":      fe_add_speaker_subject,
        "fe_avg_aggregates":        fe_add_speaker_avg_statement_len,
        "fe_text_style":            fe_add_negation_count,
        "fe_sentiment":             fe_add_sentiment,
        "true_rate_features":       enable_true_rate_features,
        "true_rate_cols":           list(_tr_group_cols.values()) if enable_true_rate_features else [],
        "all_scales":               "none",
        "enable_hp_search":         enable_hp_search,
        "n_iter_search":            N_ITER_SEARCH if enable_hp_search else 0,
        "n_cv_inner":               N_CV_INNER if enable_hp_search else 0,
        "hp_search_metric":         HP_SEARCH_METRIC if enable_hp_search else "n/a",
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

    X_fold_train = X_trainval.iloc[train_idx].copy()
    X_fold_val   = X_trainval.iloc[val_idx].copy()
    y_fold_train = y_trainval.iloc[train_idx]
    y_fold_val   = y_trainval.iloc[val_idx]

    # Compute true-rate features on the training split and map to validation split.
    # Computing on the full fold (train+val) would leak label information.
    if enable_true_rate_features and _tr_group_cols:
        _grp_tr = _grp_trainval.iloc[train_idx]
        _grp_vl = _grp_trainval.iloc[val_idx]
        for _feat, _src_col in _tr_group_cols.items():
            _rate_map = _grp_tr.groupby(_src_col)["_label"].mean()
            X_fold_train[_feat] = _grp_tr[_src_col].map(_rate_map).fillna(true_rate_fallback).values
            X_fold_val[_feat]   = _grp_vl[_src_col].map(_rate_map).fillna(true_rate_fallback).values

    if enable_hp_search:
        # n_jobs=1 on the inner RF to avoid nested parallelism; RandomizedSearchCV
        # parallelises across param combinations instead.
        _base_rf = RandomForestClassifier(
            max_depth=MAX_DEPTH,
            class_weight=CLASS_WEIGHT,
            n_jobs=1,
            random_state=42,
        )
        _inner_cv = StratifiedKFold(n_splits=N_CV_INNER, shuffle=True, random_state=42)
        _inner_search = RandomizedSearchCV(
            _base_rf,
            param_distributions=param_dist,
            n_iter=N_ITER_SEARCH,
            scoring=HP_SEARCH_METRIC,
            cv=_inner_cv,
            refit=True,
            random_state=42,
            n_jobs=-1,
            error_score="raise",
        )
        _inner_search.fit(X_fold_train, y_fold_train)
        fold_model       = _inner_search.best_estimator_
        fold_best_params = _inner_search.best_params_
        print(f"    HP best: {fold_best_params}  (inner CV {HP_SEARCH_METRIC}={_inner_search.best_score_:.4f})")
    else:
        fold_model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight=CLASS_WEIGHT,
            n_jobs=-1,
            random_state=42,
        )
        fold_model.fit(X_fold_train, y_fold_train)
        fold_best_params = {}

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
        "best_params":  fold_best_params,
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
    _wandb_fold = {
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
    }
    if enable_hp_search and fold_best_params:
        _wandb_fold["hp/fold_max_features"]     = str(fold_best_params.get("max_features"))
        _wandb_fold["hp/fold_min_samples_leaf"] = fold_best_params.get("min_samples_leaf")
        _wandb_fold["hp/fold_n_estimators"]     = fold_best_params.get("n_estimators")
    wandb.log(_wandb_fold)


# -----------------------------------------------------------------------------
# CV summary
# -----------------------------------------------------------------------------
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
    data=[
        [int(m["fold"])] + [float(m[k]) for k in _cv_keys]
        for m in cv_fold_metrics
    ],
)
wandb.log({"cv/folds_table": cv_table, **cv_summary})


# -----------------------------------------------------------------------------
# HP aggregation — choose final hyperparameters from nested CV results
# -----------------------------------------------------------------------------
# Default: fall back to the fixed values from model config if hp search is off.
_best_params_final = {
    "n_estimators":     N_ESTIMATORS,
    "max_features":     "sqrt",
    "min_samples_leaf": MIN_SAMPLES_LEAF,
}

if enable_hp_search:
    from collections import Counter
    print(f"\n[SECTION] Aggregating HP search results  [{_now()}]")

    _all_fold_params = [m["best_params"] for m in cv_fold_metrics if m.get("best_params")]

    # Categorical / list HPs: pick the mode across folds
    for _hp in ["max_features", "n_estimators"]:
        _vals = [p[_hp] for p in _all_fold_params if _hp in p]
        if _vals:
            _counts = Counter(_vals).most_common()
            _mode   = _counts[0][0]
            _best_params_final[_hp] = _mode
            print(f"  {_hp:25s}: {_counts}  → chosen: {_mode}")

    # Integer HPs: pick the median across folds
    for _hp in ["min_samples_leaf"]:
        _vals = [p[_hp] for p in _all_fold_params if _hp in p]
        if _vals:
            _median = int(np.median(_vals))
            _best_params_final[_hp] = _median
            print(f"  {_hp:25s}: {sorted(_vals)}  → median: {_median}")

    print(f"\n  Final HP for fit: {_best_params_final}")
    wandb.log({
        "hp/final_max_features":     str(_best_params_final["max_features"]),
        "hp/final_min_samples_leaf": _best_params_final["min_samples_leaf"],
        "hp/final_n_estimators":     _best_params_final["n_estimators"],
        "hp/search_table": wandb.Table(
            columns=["fold", "max_features", "min_samples_leaf", "n_estimators"],
            data=[
                [
                    m["fold"],
                    str(m["best_params"].get("max_features", "n/a")),
                    m["best_params"].get("min_samples_leaf", "n/a"),
                    m["best_params"].get("n_estimators", "n/a"),
                ]
                for m in cv_fold_metrics
            ],
        ),
    })


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
        "threshold/best":                    best_threshold,
        f"threshold/oof_{THRESHOLD_METRIC}": best_score,
        "threshold/grid": wandb.Table(
            columns=["threshold", THRESHOLD_METRIC],
            data=[[t, s] for t, s in threshold_scores.items()],
        ),
    })

    if overwrite_threshold:
        print(f"  THRESHOLD updated: {THRESHOLD:.2f} → {best_threshold:.2f}")
        THRESHOLD = best_threshold
    else:
        print(f"  THRESHOLD kept at {THRESHOLD:.2f} (overwrite_threshold=False)")


# -----------------------------------------------------------------------------
# Final fit on full trainval
# -----------------------------------------------------------------------------
print(f"[SECTION] Fitting final model on full train/val set  [{_now()}]")
_t0 = time()

_final_rate_maps: dict[str, dict] = {}
X_trainval_final = X_trainval.copy()
if enable_true_rate_features and _tr_group_cols:
    for _feat, _src_col in _tr_group_cols.items():
        _rate_map = _grp_trainval.groupby(_src_col)["_label"].mean()
        X_trainval_final[_feat] = _grp_trainval[_src_col].map(_rate_map).fillna(true_rate_fallback).values
        X_holdout[_feat]        = _grp_holdout[_src_col].map(_rate_map).fillna(true_rate_fallback).values
        _final_rate_maps[_feat] = _rate_map.to_dict()

model = RandomForestClassifier(
    n_estimators=_best_params_final["n_estimators"],
    max_depth=MAX_DEPTH,
    min_samples_leaf=_best_params_final["min_samples_leaf"],
    max_features=_best_params_final["max_features"],
    class_weight=CLASS_WEIGHT,
    n_jobs=-1,
    random_state=42,
)
model.fit(X_trainval_final, y_trainval)
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
# Feature importance
# -----------------------------------------------------------------------------
# RandomForest exposes mean-decrease-in-impurity importance for every feature.
# Log the top 30 to W&B so we can identify which feature families drive predictions.
print("[SECTION] Computing feature importance")
feature_names_final = X_trainval_final.columns.tolist()
importances = model.feature_importances_
importance_df = (
    pd.DataFrame({"feature": feature_names_final, "importance": importances})
    .sort_values("importance", ascending=False)
    .reset_index(drop=True)
)
TOP_N = 30
print(f"  Top {TOP_N} features:")
for _, row in importance_df.head(TOP_N).iterrows():
    print(f"    {row['feature']:50s}  {row['importance']:.4f}")

importance_table = wandb.Table(
    columns=["rank", "feature", "importance"],
    data=[[i + 1, row["feature"], float(row["importance"])] for i, row in importance_df.iterrows()],
)
wandb.log({"feature_importance/table": importance_table})


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(y_holdout, y_proba)
prec_c, rec_c, _ = precision_recall_curve(y_holdout, y_proba)

fig, axes = plt.subplots(1, 4, figsize=(22, 5))

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

# Top-20 feature importance bar chart
axes[3].barh(
    importance_df["feature"].head(20).values[::-1],
    importance_df["importance"].head(20).values[::-1],
)
axes[3].set_title("Top 20 Feature Importances")
axes[3].set_xlabel("Mean Decrease in Impurity")
axes[3].tick_params(axis="y", labelsize=7)

plt.tight_layout()


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
    "roc_pr_importance":    wandb.Image(fig),
    "confusion_matrix":     wandb.plot.confusion_matrix(
        y_true=y_holdout.tolist(),
        preds=y_pred.tolist(),
        class_names=["True (0)", "False (1)"],
    ),
})

run.summary["holdout/macro_f1"]   = holdout_metrics["macro_f1"]
run.summary["holdout/roc_auc"]    = holdout_metrics["roc_auc"]
run.summary["cv_mean_macro_f1"]   = cv_summary["cv_mean_macro_f1"]

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
# Save model artifacts
# -----------------------------------------------------------------------------
print("[SECTION] Saving model")
saved_paths = save_model(
    model_pipeline=model,
    preprocessing_options=options,
    feature_names=X_trainval_final.columns.tolist(),
    project_root=project_root,
    model_name=model_name,
)
print(f"  Saved core artifacts: {saved_paths}")

_model_dir = project_root / "models" / model_name

# Save the OrdinalEncoder so a submission script can apply the same category
# mapping to test data without refitting on a different vocabulary.
_enc_path = _model_dir / f"{model_name}-ordinal-encoder.joblib"
joblib.dump(ordinal_enc, _enc_path)
print(f"  OrdinalEncoder saved: {_enc_path}  ({len(_cat_cols)} categorical columns)")

# Save the fitted TF-IDF vectorizer so test data is projected into the same
# vocabulary that was built on training data.
if statement_vectorizer_type == "tfidf":
    from sklearn.feature_extraction.text import TfidfVectorizer
    _vec = TfidfVectorizer(
        max_features=statement_vectorizer_max_features,
        min_df=statement_vectorizer_min_df,
        max_df=statement_vectorizer_max_df,
    )
    _vec.fit(df_processed[statement_output_col])
    _vec_path = _model_dir / f"{model_name}-vectorizer.joblib"
    joblib.dump(_vec, _vec_path)
    print(f"  TF-IDF vectorizer saved: {_vec_path}  (vocab: {len(_vec.vocabulary_):,})")

# Save the fixed decision threshold.
_threshold_path = _model_dir / f"{model_name}-threshold.joblib"
joblib.dump(THRESHOLD, _threshold_path)
print(f"  Threshold saved: {_threshold_path}  (value: {THRESHOLD:.2f})")

# Save true-rate maps so a submission script can look up speaker/subject/party
# rates for test rows without recomputing them on test labels.
if enable_true_rate_features and _tr_group_cols and _final_rate_maps:
    _rate_maps_path = _model_dir / f"{model_name}-true-rate-maps.joblib"
    joblib.dump(
        {"rate_maps": _final_rate_maps, "group_cols": _tr_group_cols, "fallback": true_rate_fallback},
        _rate_maps_path,
    )
    print(f"  True-rate maps saved: {_rate_maps_path}  ({len(_final_rate_maps)} features)")

# -----------------------------------------------------------------------------
# Kaggle submission CSV
# -----------------------------------------------------------------------------
if create_kaggle_csv:
    print(f"[SECTION] Creating Kaggle submission CSV from saved model  [{_now()}]")
    kaggle_tree_path = project_root / "src" / "submit" / "kaggle_module_forTrees.py"
    spec = importlib.util.spec_from_file_location("kaggle_module_forTrees", kaggle_tree_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kaggle module from: {kaggle_tree_path}")

    kaggle_tree = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_tree)

    submission_path = kaggle_tree.generate_tree_submission_csv(
        model_name=model_name,
        project_root=project_root,
        verbose=True,
    )
    print(f"Kaggle submission generated: {submission_path}")

print(f"\n[DONE] Total script time: {time()-_script_start:.1f}s  [{_now()}]")
