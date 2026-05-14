"""
Stacking ensemble: LR + RFC + LGBM-optB + CatBoost-optB → meta LogisticRegression.

Architecture
------------
- Single shared preprocessing (sentence embeddings + OrdinalEncoder, same as CatBoost-optB).
- All 4 base models trained with their known-best HPs; no inner HP search.
- 5-fold stratified CV collects OOF probas for each base model.
- Meta-LR trained on the stacked OOF matrix (N_trainval × 4).
- Threshold tuning on stacked OOF.
- Final evaluation on holdout.

True-rate features: drop_speaker_true_rate=True (confirmed best — speaker true-rate collapses ensemble diversity).
LR base model: fitted on StandardScaler-transformed features inside each fold.
"""
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
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


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
# PREPROCESSING OPTIONS  (shared across all 4 base models)
# Mirrors the CatBoost-optB config — best individual model.
# ============================================================

label_option      = 'skip'
label_source_col  = 'label'
id_option         = 'drop'

subject_source_col             = 'subject'
subject_keep_original          = False
subject_clean_text             = True
subject_normalize_separators   = True
subject_split_topics           = True
subject_primary_strategy       = 'most_frequent'
subject_rare_threshold         = 10
subject_rare_label             = 'other'
subject_max_topics_for_primary = None
subject_multi_topic_label      = 'multi-topic'
subject_add_primary            = True
subject_add_topic_count        = True
subject_add_multiple_topics_flag = True
subject_add_topic_list         = False
subject_add_length_features    = True
subject_add_grouped_primary    = True
subject_group_rare             = True
subject_add_subject_frequency  = True
subject_add_subject_is_rare    = True
subject_add_subject_primary_true_rate = False
subject_label_col              = None
subject_scale                  = 'none'
subject_verbose                = False

statement_source_col            = 'statement'
statement_original_output_col   = 'statement_original'
statement_output_col            = 'statement_clean'
statement_keep_original         = False
statement_lower                 = True
statement_remove_html           = True
statement_remove_urls           = True
statement_replace_numbers       = False
statement_number_token          = '<NUM>'
statement_stopword_removal      = False
statement_keep_negations        = True
statement_remove_punctuation    = False
statement_stemmer               = 'none'
statement_lemmatizer            = 'none'
statement_repair_polluted_statements = True
statement_add_rare_token_features = True
statement_rare_token_threshold  = 1
statement_token_freqs           = None
statement_add_spelling_errors   = True
statement_add_lexical_features  = True
statement_add_pollution_features = True
statement_add_ner_features      = True
statement_ner_model             = 'en_core_web_sm'
statement_vectorizer_type       = 'embeddings'
statement_vectorizer_max_features = 500
statement_vectorizer_min_df     = 2
statement_vectorizer_max_df     = 0.7
statement_embedding_model       = 'all-mpnet-base-v2'
statement_fitted_vectorizer     = None
statement_scale                 = 'none'
statement_verbose               = False

speaker_source_col            = 'speaker'
speaker_keep_original         = False
speaker_clean_text            = True
speaker_normalize_separators  = True
speaker_group_rare            = True
speaker_rare_threshold        = 5
speaker_rare_label            = 'other'
speaker_add_frequency         = True
speaker_add_is_rare           = True
speaker_add_grouped_speaker   = True
speaker_add_length_features   = True
speaker_add_title_flag        = True
speaker_add_comma_flag        = True
speaker_add_period_flag       = True
speaker_add_token_count       = False
speaker_add_speaker_primary_true_rate = False
speaker_label_col             = None
speaker_scale                 = 'none'
speaker_verbose               = False

speaker_job_source_col           = 'speaker_job'
speaker_job_keep_original        = False
speaker_job_clean_text           = True
speaker_job_normalize_separators = True
speaker_job_group_rare           = True
speaker_job_rare_threshold       = 5
speaker_job_rare_label           = 'other'
speaker_job_add_frequency        = True
speaker_job_add_is_rare          = True
speaker_job_add_grouped_job      = True
speaker_job_add_length_features  = True
speaker_job_add_title_flag       = True
speaker_job_add_comma_flag       = True
speaker_job_add_slash_flag       = True
speaker_job_add_ampersand_flag   = True
speaker_job_add_token_count      = False
speaker_job_add_job_primary_true_rate = False
speaker_job_job_label_col        = None
speaker_job_scale                = 'none'
speaker_job_verbose              = False

party_affiliation_source_col       = 'party_affiliation'
party_affiliation_keep_original    = False
party_affiliation_clean_text       = True
party_affiliation_group_rare       = True
party_affiliation_rare_threshold   = 5
party_affiliation_rare_label       = 'other'
party_affiliation_add_frequency    = True
party_affiliation_add_is_rare      = True
party_affiliation_add_grouped_party = True
party_affiliation_add_length_features = True
party_affiliation_add_token_count  = True
party_affiliation_add_is_major_party = True
party_affiliation_add_is_institutional = True
party_affiliation_add_slash_flag   = True
party_affiliation_add_ampersand_flag = True
party_affiliation_add_comma_flag   = False
party_affiliation_add_parentheses_flag = True
party_affiliation_add_party_primary_true_rate = False
party_affiliation_party_label_col  = None
party_affiliation_scale            = 'none'
party_affiliation_verbose          = False

state_source_col       = 'state_info'
state_drop             = False
state_keep_original    = False
state_clean_text       = True
state_normalize_state  = True
state_group_rare       = True
state_rare_threshold   = 5
state_rare_label       = 'other'
state_add_is_us_state  = True
state_add_frequency    = True
state_add_is_rare      = True
state_add_grouped_state = True
state_add_has_us_words = True
state_add_us_region    = True
state_add_length_features = False
state_add_token_count  = True
state_scale            = 'none'
state_verbose          = False

fe_statement_col          = 'statement_clean'
fe_statement_original_col = 'statement_original'
fe_speaker_col            = 'speaker_clean'
fe_subject_col            = 'subject_clean'
fe_party_col              = 'party_affiliation_clean'
fe_speaker_job_col        = 'speaker_job_clean'
fe_state_col              = 'state_info_clean'
fe_label_col              = None
fe_add_speaker_subject    = True
fe_add_speaker_party      = True
fe_add_subject_party      = True
fe_add_speaker_job_subject = True
fe_add_state_party        = True
fe_add_speaker_statement_len_bucket = True
fe_statement_len_bins     = (50, 150)
fe_add_speaker_true_rate  = False
fe_add_subject_true_rate  = False
fe_add_party_true_rate    = False
fe_add_speaker_avg_statement_len = True
fe_add_subject_avg_statement_len = True
fe_add_speaker_avg_punctuation   = True
fe_add_speaker_avg_number_ratio  = True
fe_add_negation_count    = True
fe_add_hedge_count       = True
fe_add_absolutist_count  = True
fe_add_numeral_count     = True
fe_add_proper_noun_count = False
fe_add_readability       = True
fe_add_sentiment         = True
fe_scale                 = 'none'
fe_verbose               = False

options = OneStepOptions(
    label_option=label_option,
    label_source_col=label_source_col,
    id_option=id_option,
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
print("[SECTION] Categorical encoding")

_all_obj_cols = df_processed.select_dtypes(include="object").columns.tolist()
_source_cols = {
    statement_source_col, speaker_source_col, subject_source_col,
    speaker_job_source_col, party_affiliation_source_col, state_source_col,
}
_text_cols = (
    {c for c in _all_obj_cols if c.endswith(("_clean", "_original"))}
    | (_source_cols & set(_all_obj_cols))
)
_cat_cols = [c for c in _all_obj_cols if c not in _text_cols and c != label_source_col]

print(f"  Text columns dropped    : {sorted(_text_cols)}")
print(f"  Categorical cols encoded: {_cat_cols}")

# dtype=int required so CatBoost receives integer-coded categoricals
ordinal_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=int)
_cat_encoded = pd.DataFrame(
    ordinal_enc.fit_transform(df_processed[_cat_cols]),
    columns=_cat_cols,
    index=df_processed.index,
)

df_features = pd.concat(
    [df_processed.select_dtypes(exclude="object"), _cat_encoded],
    axis=1,
)

# CatBoost: column names that should be treated as categoricals
_cat_feature_names = [c for c in _cat_encoded.columns if c != label_source_col]


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
# Train / holdout split  (identical seed to all other scripts)
# -----------------------------------------------------------------------------
print(f"[SECTION] Building train/holdout split  [{_now()}]")
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print(f"Train/val: {X_trainval.shape[0]:,}   Holdout: {X_holdout.shape[0]:,}   CV folds: {skf.get_n_splits()}")


# -----------------------------------------------------------------------------
# Stacking configuration
# -----------------------------------------------------------------------------
CLASS_WEIGHTS    = [1.42, 0.77]      # balanced weights for base models
CLASS_WEIGHT_D   = {0: 1.42, 1: 0.77}
THRESHOLD        = 0.5
model_name       = "stacking"
create_kaggle_csv = True

# ---- Late fusion: add transformer k-fold OOF as a 5th base model ----
# Base model artifacts take priority over small when both exist.
# Run transformer_kfold_base_extract.py (or transformer_kfold_extract.py) first.
def _find_transformer_artifacts(root: Path):
    """Return (dir, oof_filename, variant_label) for best available transformer artifacts."""
    candidates = [
        (root / "models" / "transformer_lora_kfold",  "mistral-7b-lora-kfold-oof.csv",  "mistral-7b-lora"),
        (root / "models" / "transformer_kfold_base",  "deberta-v3-base-kfold-oof.csv",   "deberta-v3-base"),
        (root / "models" / "transformer_kfold",       "deberta-v3-small-kfold-oof.csv",  "deberta-v3-small"),
    ]
    for d, oof_name, label in candidates:
        if (d / oof_name).exists() and (d / "ho_proba.npy").exists() and (d / "test_proba.npy").exists():
            return d, oof_name, label
    return None, None, None

_KFOLD_DIR, _TRANS_OOF_NAME, _TRANS_VARIANT = _find_transformer_artifacts(project_root)
use_transformer_fusion = _KFOLD_DIR is not None
if use_transformer_fusion:
    TRANSFORMER_OOF_PATH  = _KFOLD_DIR / _TRANS_OOF_NAME
    TRANSFORMER_HO_PATH   = _KFOLD_DIR / "ho_proba.npy"
    TRANSFORMER_TEST_PATH = _KFOLD_DIR / "test_proba.npy"
    print(f"  [Late fusion] Transformer artifacts found (variant={_TRANS_VARIANT}) — adding as 5th base model")
else:
    print(f"  [Late fusion] Disabled — run transformer_kfold_base_extract.py to enable")

enable_threshold_tuning = True
overwrite_threshold     = True
THRESHOLD_METRIC        = "macro_f1"

# drop_speaker_true_rate=True confirmed best across individual models AND stacking ensemble
drop_speaker_true_rate = True
enable_true_rate_features = True
true_rate_fallback = 0.5

# Base model HPs — best known from prior experiments, no inner search
# Experiment D: replace base LR (AUC 0.6720, weakest) with XGBoost
BASE_XGB_HP = dict(n_estimators=500, learning_rate=0.03, max_depth=6,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0,
                   eval_metric="logloss", random_state=42, n_jobs=-1, verbosity=0)
# XGBoost has no class_weight param; apply via sample_weight at fit time
_XGB_SW = {0: CLASS_WEIGHT_D[0], 1: CLASS_WEIGHT_D[1]}

BASE_RFC_HP = dict(n_estimators=500, max_features=0.3, min_samples_leaf=2,
                   class_weight=CLASS_WEIGHT_D, n_jobs=-1, random_state=42)

BASE_LGBM_HP = dict(n_estimators=500, learning_rate=0.03, num_leaves=63,
                    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.0, reg_lambda=0.0,
                    class_weight=CLASS_WEIGHT_D, n_jobs=-1, random_state=42, verbose=-1)

BASE_CAT_HP = dict(iterations=300, learning_rate=0.03, depth=4, l2_leaf_reg=5,
                   border_count=32, bagging_temperature=0.0,
                   class_weights=CLASS_WEIGHTS, thread_count=-1, random_seed=42, verbose=0)

# Meta-learner: plain LR on 4 stacked OOF probas.
# No class_weight — the base probas already encode the imbalance signal.
META_LR_HP = dict(C=0.1, penalty="l2", max_iter=1000, solver="lbfgs", random_state=42)

BASE_NAMES = ["xgb", "rfc", "lgbm", "cat"] + (["transformer"] if use_transformer_fusion else [])


# -----------------------------------------------------------------------------
# True-rate feature setup  (mirrors cat.py logic)
# -----------------------------------------------------------------------------
_tr_group_cols: dict[str, str] = {}
if enable_true_rate_features:
    _candidates = {
        "fe_subject_true_rate":     ["subject_primary_grouped", "subject_primary", "subject_clean"],
        "fe_party_true_rate":       ["party_affiliation_grouped", "party_affiliation_clean"],
        "fe_speaker_job_true_rate": ["speaker_job_grouped", "speaker_job_clean"],
    }
    if not drop_speaker_true_rate:
        _candidates["fe_speaker_true_rate"] = ["speaker_grouped", "speaker_clean"]
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
    project="truth-classifier-stacking",
    config={
        "model":                    "Stacking (LR+RFC+LGBM+CatBoost → meta-LR)",
        "base_models":              BASE_NAMES,
        "meta_learner":             "LogisticRegression",
        "meta_C":                   META_LR_HP["C"],
        "drop_speaker_true_rate":   drop_speaker_true_rate,
        "true_rate_features":       list(_tr_group_cols.keys()),
        "cv_folds":                 skf.get_n_splits(),
        "n_trainval":               int(X_trainval.shape[0]),
        "n_holdout":                int(X_holdout.shape[0]),
        "n_features_total":         int(X.shape[1]),
        "vectorizer_type":          statement_vectorizer_type,
        "embedding_model":          statement_embedding_model,
        "base_xgb_n_estimators":    BASE_XGB_HP["n_estimators"],
        "base_xgb_max_depth":       BASE_XGB_HP["max_depth"],
        "base_rfc_n_estimators":    BASE_RFC_HP["n_estimators"],
        "base_lgbm_n_estimators":   BASE_LGBM_HP["n_estimators"],
        "base_lgbm_lr":             BASE_LGBM_HP["learning_rate"],
        "base_cat_iterations":      BASE_CAT_HP["iterations"],
        "base_cat_lr":              BASE_CAT_HP["learning_rate"],
        "base_cat_depth":           BASE_CAT_HP["depth"],
    },
)


# -----------------------------------------------------------------------------
# Cross-validation — collect OOF probas from all 4 base models
# -----------------------------------------------------------------------------
print(f"[SECTION] Running cross-validation  [{_now()}]")
_cv_start = time()

# OOF arrays — one per base model, length = N_trainval
oof_xgb  = np.zeros(len(X_trainval))
oof_rfc  = np.zeros(len(X_trainval))
oof_lgbm = np.zeros(len(X_trainval))
oof_cat  = np.zeros(len(X_trainval))
oof_true = np.zeros(len(X_trainval), dtype=int)

cv_fold_metrics = []

for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
    _fold_t = time()

    X_fold_train = X_trainval.iloc[train_idx].copy()
    X_fold_val   = X_trainval.iloc[val_idx].copy()
    y_fold_train = y_trainval.iloc[train_idx]
    y_fold_val   = y_trainval.iloc[val_idx]

    # ---- True-rate features (fold-safe) ----
    if enable_true_rate_features and _tr_group_cols:
        _grp_tr = _grp_trainval.iloc[train_idx]
        _grp_vl = _grp_trainval.iloc[val_idx]
        for _feat, _src_col in _tr_group_cols.items():
            _rate_map = _grp_tr.groupby(_src_col)["_label"].mean()
            X_fold_train[_feat] = _grp_tr[_src_col].map(_rate_map).fillna(true_rate_fallback).values
            X_fold_val[_feat]   = _grp_vl[_src_col].map(_rate_map).fillna(true_rate_fallback).values

    # ---- XGBoost base model (Experiment D: replaces base LR) ----
    _xgb_sw = np.where(y_fold_train == 0, _XGB_SW[0], _XGB_SW[1])
    _m_xgb = XGBClassifier(**BASE_XGB_HP)
    _m_xgb.fit(X_fold_train, y_fold_train, sample_weight=_xgb_sw)
    oof_xgb[val_idx] = _m_xgb.predict_proba(X_fold_val)[:, 1]

    # ---- RFC base model ----
    _m_rfc = RandomForestClassifier(**BASE_RFC_HP)
    _m_rfc.fit(X_fold_train, y_fold_train)
    oof_rfc[val_idx] = _m_rfc.predict_proba(X_fold_val)[:, 1]

    # ---- LGBM-optB base model ----
    _m_lgbm = LGBMClassifier(**BASE_LGBM_HP)
    _m_lgbm.fit(X_fold_train, y_fold_train)
    oof_lgbm[val_idx] = _m_lgbm.predict_proba(X_fold_val)[:, 1]

    # ---- CatBoost-optB base model ----
    # cat_features indices are recomputed per fold because true-rate columns
    # are appended after OrdinalEncoded columns, shifting later positions.
    _cat_indices = [X_fold_train.columns.get_loc(c) for c in _cat_feature_names
                    if c in X_fold_train.columns]
    _m_cat = CatBoostClassifier(**BASE_CAT_HP, cat_features=_cat_indices)
    _m_cat.fit(X_fold_train, y_fold_train)
    oof_cat[val_idx] = _m_cat.predict_proba(X_fold_val)[:, 1]

    oof_true[val_idx] = y_fold_val.values

    # ---- Per-fold stacked OOF metrics (using default threshold 0.5) ----
    _meta_X_fold = np.column_stack([oof_xgb[val_idx], oof_rfc[val_idx],
                                    oof_lgbm[val_idx], oof_cat[val_idx]])
    _meta_avg   = _meta_X_fold.mean(axis=1)  # simple average as a proxy before meta-LR is trained
    _preds_avg  = (_meta_avg >= 0.5).astype(int)

    fold_metrics = {
        "fold":          fold_idx,
        "roc_auc_avg":   roc_auc_score(y_fold_val, _meta_avg),
        "macro_f1_avg":  f1_score(y_fold_val, _preds_avg, average="macro", zero_division=0),
        "roc_auc_xgb":   roc_auc_score(y_fold_val, oof_xgb[val_idx]),
        "roc_auc_rfc":   roc_auc_score(y_fold_val, oof_rfc[val_idx]),
        "roc_auc_lgbm":  roc_auc_score(y_fold_val, oof_lgbm[val_idx]),
        "roc_auc_cat":   roc_auc_score(y_fold_val, oof_cat[val_idx]),
    }
    cv_fold_metrics.append(fold_metrics)

    print(
        f"  Fold {fold_idx} | "
        f"avg-ROC={fold_metrics['roc_auc_avg']:.4f}  "
        f"avg-F1={fold_metrics['macro_f1_avg']:.4f}  "
        f"XGB={fold_metrics['roc_auc_xgb']:.4f}  "
        f"RFC={fold_metrics['roc_auc_rfc']:.4f}  "
        f"LGBM={fold_metrics['roc_auc_lgbm']:.4f}  "
        f"CAT={fold_metrics['roc_auc_cat']:.4f}  "
        f"({time()-_fold_t:.1f}s)"
    )
    wandb.log({
        "cv/fold":           fold_idx,
        "cv/roc_auc_avg":    fold_metrics["roc_auc_avg"],
        "cv/macro_f1_avg":   fold_metrics["macro_f1_avg"],
        "cv/roc_auc_xgb":    fold_metrics["roc_auc_xgb"],
        "cv/roc_auc_rfc":    fold_metrics["roc_auc_rfc"],
        "cv/roc_auc_lgbm":   fold_metrics["roc_auc_lgbm"],
        "cv/roc_auc_cat":    fold_metrics["roc_auc_cat"],
    })

print(f"\n[SECTION] Cross-validation summary  [total CV: {time()-_cv_start:.1f}s]")
for k in ["roc_auc_avg", "macro_f1_avg", "roc_auc_xgb", "roc_auc_rfc", "roc_auc_lgbm", "roc_auc_cat"]:
    vals = [m[k] for m in cv_fold_metrics]
    print(f"  {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


# -----------------------------------------------------------------------------
# Late fusion — load transformer OOF and align to trainval row order
# -----------------------------------------------------------------------------
if use_transformer_fusion:
    print(f"\n[SECTION] Loading transformer OOF  [{_now()}]")
    _trans_oof_df   = pd.read_csv(TRANSFORMER_OOF_PATH)
    _idx_to_proba   = dict(zip(_trans_oof_df["idx"].astype(int),
                                _trans_oof_df["oof_proba"]))
    oof_transformer = np.array([_idx_to_proba[i] for i in X_trainval.index],
                                dtype=np.float64)
    print(f"  Loaded {len(_trans_oof_df):,} OOF rows  "
          f"range=[{oof_transformer.min():.3f}, {oof_transformer.max():.3f}]")
    wandb.log({"transformer_oof/roc_auc":
               roc_auc_score(y_trainval, oof_transformer)})


# -----------------------------------------------------------------------------
# Train meta-LR on full stacked OOF
# -----------------------------------------------------------------------------
print(f"\n[SECTION] Training meta-LR on stacked OOF  [{_now()}]")
_oof_cols = [oof_xgb, oof_rfc, oof_lgbm, oof_cat]
if use_transformer_fusion:
    _oof_cols.append(oof_transformer)
meta_X_train = np.column_stack(_oof_cols)
meta_lr = LogisticRegression(**META_LR_HP)
meta_lr.fit(meta_X_train, y_trainval)

meta_coefs = dict(zip(BASE_NAMES, meta_lr.coef_[0].tolist()))
print(f"  Meta-LR coefficients: {meta_coefs}")
_coef_log = {f"meta/coef_{n}": meta_coefs[n] for n in BASE_NAMES}
wandb.log(_coef_log)


# -----------------------------------------------------------------------------
# Threshold tuning on stacked OOF (meta-LR probas)
# -----------------------------------------------------------------------------
oof_meta_proba = meta_lr.predict_proba(meta_X_train)[:, 1]

if enable_threshold_tuning:
    print(f"\n[SECTION] Threshold tuning on stacked OOF  [{_now()}]")

    _metric_fn = {
        "macro_f1":     lambda t, yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "mcc":          lambda t, yt, yp: matthews_corrcoef(yt, yp),
        "balanced_acc": lambda t, yt, yp: balanced_accuracy_score(yt, yp),
    }[THRESHOLD_METRIC]

    threshold_grid   = np.arange(0.20, 0.76, 0.01)
    threshold_scores = {}
    for t in threshold_grid:
        preds = (oof_meta_proba >= t).astype(int)
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


# -----------------------------------------------------------------------------
# Final fit — retrain all 4 base models on full trainval
# -----------------------------------------------------------------------------
print(f"[SECTION] Fitting final base models on full train/val set  [{_now()}]")
_t0 = time()

_final_rate_maps: dict[str, dict] = {}
X_trainval_final = X_trainval.copy()
X_holdout_final  = X_holdout.copy()

if enable_true_rate_features and _tr_group_cols:
    for _feat, _src_col in _tr_group_cols.items():
        _rate_map = _grp_trainval.groupby(_src_col)["_label"].mean()
        X_trainval_final[_feat] = _grp_trainval[_src_col].map(_rate_map).fillna(true_rate_fallback).values
        X_holdout_final[_feat]  = _grp_holdout[_src_col].map(_rate_map).fillna(true_rate_fallback).values
        _final_rate_maps[_feat] = _rate_map.to_dict()

# XGBoost (Experiment D: replaces base LR)
_xgb_sw_final = np.where(np.array(y_trainval) == 0, _XGB_SW[0], _XGB_SW[1])
final_xgb = XGBClassifier(**BASE_XGB_HP)
final_xgb.fit(X_trainval_final, y_trainval, sample_weight=_xgb_sw_final)

# RFC
final_rfc = RandomForestClassifier(**BASE_RFC_HP)
final_rfc.fit(X_trainval_final, y_trainval)

# LGBM
final_lgbm = LGBMClassifier(**BASE_LGBM_HP)
final_lgbm.fit(X_trainval_final, y_trainval)

# CatBoost
_cat_indices_final = [X_trainval_final.columns.get_loc(c) for c in _cat_feature_names
                      if c in X_trainval_final.columns]
final_cat = CatBoostClassifier(**BASE_CAT_HP, cat_features=_cat_indices_final)
final_cat.fit(X_trainval_final, y_trainval)

print(f"  Done in {time()-_t0:.1f}s")


# -----------------------------------------------------------------------------
# Holdout evaluation
# -----------------------------------------------------------------------------
print(f"[SECTION] Evaluating on holdout set  [{_now()}]")
print(f"  Using threshold: {THRESHOLD:.2f}")

h_xgb  = final_xgb.predict_proba(X_holdout_final)[:, 1]
h_rfc  = final_rfc.predict_proba(X_holdout_final)[:, 1]
h_lgbm = final_lgbm.predict_proba(X_holdout_final)[:, 1]
h_cat  = final_cat.predict_proba(X_holdout_final)[:, 1]

_ho_cols = [h_xgb, h_rfc, h_lgbm, h_cat]
if use_transformer_fusion:
    _ho_idx_kfold    = np.load(_KFOLD_DIR / "ho_idx.npy")
    _ho_trans_lookup = dict(zip(_ho_idx_kfold.tolist(),
                                np.load(TRANSFORMER_HO_PATH).tolist()))
    _h_transformer   = np.array([_ho_trans_lookup[i] for i in X_holdout.index])
    _ho_cols.append(_h_transformer)
    print(f"  Transformer holdout range=[{_h_transformer.min():.3f}, {_h_transformer.max():.3f}]")
meta_X_holdout = np.column_stack(_ho_cols)
y_proba = meta_lr.predict_proba(meta_X_holdout)[:, 1]
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

# Individual base model holdout ROC-AUC for reference
print("  Base model holdout ROC-AUC:")
print(f"    XGB : {roc_auc_score(y_holdout, h_xgb):.4f}")
print(f"    RFC : {roc_auc_score(y_holdout, h_rfc):.4f}")
print(f"    LGBM: {roc_auc_score(y_holdout, h_lgbm):.4f}")
print(f"    CAT : {roc_auc_score(y_holdout, h_cat):.4f}")


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------
print("[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(y_holdout, y_proba)
prec_c, rec_c, _ = precision_recall_curve(y_holdout, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr, tpr, label=f"Stack ROC-AUC = {holdout_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set_title("ROC Curve — Stacking (holdout)")
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
    "holdout/roc_auc_xgb":  roc_auc_score(y_holdout, h_xgb),
    "holdout/roc_auc_rfc":  roc_auc_score(y_holdout, h_rfc),
    "holdout/roc_auc_lgbm": roc_auc_score(y_holdout, h_lgbm),
    "holdout/roc_auc_cat":  roc_auc_score(y_holdout, h_cat),
    "roc_pr_cm":            wandb.Image(fig),
    "confusion_matrix":     wandb.plot.confusion_matrix(
        y_true=y_holdout.tolist(),
        preds=y_pred.tolist(),
        class_names=["True (0)", "False (1)"],
    ),
})

run.summary["holdout/macro_f1"] = holdout_metrics["macro_f1"]
run.summary["holdout/roc_auc"]  = holdout_metrics["roc_auc"]

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
# Save stacking artifacts
# -----------------------------------------------------------------------------
print("[SECTION] Saving stacking artifacts")
_model_dir = project_root / "models" / model_name
_model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(meta_lr,        _model_dir / "stacking-meta-lr.joblib")
joblib.dump(final_xgb,      _model_dir / "stacking-base-xgb.joblib")
joblib.dump(final_rfc,      _model_dir / "stacking-base-rfc.joblib")
joblib.dump(final_lgbm,     _model_dir / "stacking-base-lgbm.joblib")
joblib.dump(final_cat,      _model_dir / "stacking-base-cat.joblib")
joblib.dump(ordinal_enc,    _model_dir / "stacking-ordinal-encoder.joblib")
joblib.dump(THRESHOLD,      _model_dir / "stacking-threshold.joblib")
joblib.dump(options,        _model_dir / "stacking-options.joblib")
joblib.dump(X_trainval_final.columns.tolist(), _model_dir / "stacking-feature-names.joblib")

if enable_true_rate_features and _final_rate_maps:
    joblib.dump(
        {"rate_maps": _final_rate_maps, "group_cols": _tr_group_cols, "fallback": true_rate_fallback},
        _model_dir / "stacking-true-rate-maps.joblib",
    )

print(f"  Artifacts saved to: {_model_dir}")


# -----------------------------------------------------------------------------
# Kaggle submission CSV
# -----------------------------------------------------------------------------
if create_kaggle_csv:
    print(f"[SECTION] Creating Kaggle submission CSV  [{_now()}]")
    test_path = project_root / "data" / "test_nolabel.csv"
    df_test   = pd.read_csv(test_path)

    _test_opts = OneStepOptions(**{k: v for k, v in options.__dict__.items()
                                   if k != "label_option"} | {"label_option": "skip"})
    df_test_proc = preprocess_one_step(df_test, options=_test_opts)

    _all_obj_test = df_test_proc.select_dtypes(include="object").columns.tolist()
    _source_cols_test = {
        statement_source_col, speaker_source_col, subject_source_col,
        speaker_job_source_col, party_affiliation_source_col, state_source_col,
    }
    _text_cols_test = (
        {c for c in _all_obj_test if c.endswith(("_clean", "_original"))}
        | (_source_cols_test & set(_all_obj_test))
    )
    _cat_cols_test = [c for c in _all_obj_test if c not in _text_cols_test]

    # Apply the same OrdinalEncoder (fitted on train data)
    _cat_enc_test = pd.DataFrame(
        ordinal_enc.transform(df_test_proc[_cat_cols_test]),
        columns=_cat_cols_test,
        index=df_test_proc.index,
    )
    df_test_feat = pd.concat(
        [df_test_proc.select_dtypes(exclude="object"), _cat_enc_test],
        axis=1,
    )

    X_test = df_test_feat.reindex(columns=X_trainval_final.columns, fill_value=0.0)

    # Fill true-rate features using trainval-fitted rate maps
    if enable_true_rate_features and _final_rate_maps:
        _grp_test = pd.DataFrame(
            {col: df_test_proc[col].values if col in df_test_proc.columns
                  else [None] * len(df_test_proc)
             for col in _tr_group_cols.values()}
        )
        for _feat, _src_col in _tr_group_cols.items():
            _rmap = _final_rate_maps[_feat]
            X_test[_feat] = _grp_test[_src_col].map(_rmap).fillna(true_rate_fallback).values

    # Run through the stacking pipeline
    t_xgb  = final_xgb.predict_proba(X_test)[:, 1]
    t_rfc  = final_rfc.predict_proba(X_test)[:, 1]
    t_lgbm = final_lgbm.predict_proba(X_test)[:, 1]
    t_cat  = final_cat.predict_proba(X_test)[:, 1]

    _test_cols = [t_xgb, t_rfc, t_lgbm, t_cat]
    if use_transformer_fusion:
        _test_cols.append(np.load(TRANSFORMER_TEST_PATH))
    meta_X_test = np.column_stack(_test_cols)
    test_proba  = meta_lr.predict_proba(meta_X_test)[:, 1]
    test_pred   = (test_proba >= THRESHOLD).astype(int)

    submissions_dir = project_root / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission_path = submissions_dir / f"submission-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}.csv"

    submission = pd.DataFrame({"id": df_test["id"], "label": test_pred})
    submission.to_csv(submission_path, index=False)
    print(f"  Kaggle submission saved: {submission_path}  ({len(submission):,} rows)")

print(f"\n[DONE] Total script time: {time()-_script_start:.1f}s  [{_now()}]")
