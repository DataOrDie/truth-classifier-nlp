"""Feature sweep over all feature_engineering.py option combinations.

Speed strategy:
  - Statement TF-IDF + lemmatization, subject, speaker, and party features are
    precomputed ONCE. Each iteration only reruns the FE module (fast: counts
    and groupby aggregates).
  - 3-fold CV during the sweep; holdout evaluated only for the winning config.
  - liblinear solver + lower max_iter for faster ranking fits.

FE has three feature families swept here:
  - Interaction  : produce string columns (e.g. "obama__health-care")
                   → frequency-encoded automatically before fitting LR.
  - Aggregate    : numeric per-group statistics (no leakage risk).
  - Text-style   : linguistic counts derived from statement_clean.

Three options are excluded:
  - fe_add_speaker_true_rate   → target-encoding leakage risk (CV folds only)
  - fe_add_subject_true_rate   → target-encoding leakage risk
  - fe_add_party_true_rate     → target-encoding leakage risk

Total unique configs: 2^17 × 2 ≈ 262,144 — set MAX_RUNS to a manageable
number (e.g. 200) for a first pass. Full sweep is feasible overnight since
each iteration is fast (no TF-IDF or lemmatization).

------------------------------------------------------------------------------
Interaction feature encoding — interaction columns (fe_speaker_subject, fe_state_party, etc.) come out as strings from the
  preprocessor. The loop frequency-encodes them automatically before feeding to LR (maps each unique value to its count in the dataset).
   This is the standard approach for high-cardinality interaction keys with linear models.
                                                                                                                                        
  Grid size — the full grid is ~262K configs. Since each FE iteration is fast (no TF-IDF, no lemmatization reruns), a realistic full
  sweep could run overnight. Strongly recommended to start with MAX_RUNS = 200 to validate things work and get a first sense of which
  features help, then set MAX_RUNS = None for the full sweep.
------------------------------------------------------------------------------
analysis by feature group for LinearSVC/LR specifically:                                                                    
                                                                                                                                      
  Fix to False — high-cardinality interactions, noise for linear models:                                                                
  - fe_add_speaker_job_subject — 3-way combo, extremely high cardinality, frequency-encoding is mostly noise                            
  - fe_add_speaker_statement_len_bucket — speaker has hundreds of unique values; the bucket info adds nothing LR can't get from the base
   speaker frequency feature                                                                                                            
                                                                                                                                        
  Fix to True — established deception-detection signals, safe to always include:                                                      
  - fe_add_negation_count
  - fe_add_hedge_count
  - fe_add_absolutist_count
  - fe_add_numeral_count — specific numbers in political claims are a strong Politifact signal

  Keep sweeping (genuinely uncertain for this dataset):
  - 4 interaction features: speaker_subject, speaker_party, subject_party, state_party
  - 3 text-style: proper_noun_count, readability, sentiment
  - fe_scale

  That reduces the grid from ~16,320 → ~240 configs (2⁴ × 2³ × 2, deduped)

  -----------------------------------------------------------------------------

  Best FE config  (cv_mean_macro_f1=0.6086):
  fe_add_speaker_subject: False
  fe_add_speaker_party: False
  fe_add_subject_party: False
  fe_add_state_party: False
  fe_add_proper_noun_count: False
  fe_add_readability: True
  fe_add_sentiment: True
  fe_scale: standardize
"""

from itertools import product
from pathlib import Path
import sys
from time import sleep

import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# -----------------------------------------------------------------------------
# Path setup
# -----------------------------------------------------------------------------

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "data" / "train.csv").exists() and (candidate / "src").exists():
            return candidate
    raise FileNotFoundError("Could not locate project root.")


project_root = find_project_root(Path(__file__).resolve())
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import OneStepOptions, preprocess_one_step


# -----------------------------------------------------------------------------
# W&B login
# -----------------------------------------------------------------------------

print("[SECTION] W&B authentication")
# wandb.login()


# -----------------------------------------------------------------------------
# Load raw data once
# -----------------------------------------------------------------------------

print("[SECTION] Loading data")
df = pd.read_csv(project_root / "data" / "train.csv")
print(f"Loaded {len(df):,} rows")


# =============================================================================
# BASE CONFIG VARIABLES
# Statement, subject, speaker, party stay fixed — computed ONCE.
# FE options are left at defaults (no add_* flags) in the base pass.
# =============================================================================

# --- Statement (best config: cv_mean_macro_f1=0.6079, holdout_macro_f1=0.6021) ---
statement_vectorizer_type         = "tfidf"
statement_vectorizer_max_features = 5000
statement_stopword_removal        = False
statement_stemmer                 = "porter"
statement_lemmatizer              = "none"
statement_add_lexical_features    = True
statement_scale                   = "standardize"

# Statement options fixed for all runs
STATEMENT_FIXED = {
    "statement_lower":                       True,
    "statement_remove_html":                 True,
    "statement_remove_urls":                 True,
    "statement_repair_polluted_statements":  True,
    "statement_keep_negations":              True,
    "statement_vectorizer_min_df":           2,
    "statement_vectorizer_max_df":           0.9,
    "statement_remove_punctuation":          False,
    "statement_replace_numbers":             False,
}

# --- Subject (best config: cv_mean_macro_f1=0.5999, holdout_macro_f1=0.6036) ---
subject_add_primary              = True
subject_primary_strategy         = "most_frequent"
subject_add_topic_count          = True
subject_add_multiple_topics_flag = False
subject_add_length_features      = False
subject_add_subject_frequency    = False
subject_add_subject_is_rare      = True
subject_add_grouped_primary      = True
subject_scale                    = "standardize"

# --- Speaker (best config: cv_mean_macro_f1=0.5915) ---
speaker_add_frequency       = True
speaker_add_is_rare         = True
speaker_add_grouped_speaker = True
speaker_add_length_features = False
speaker_add_title_flag      = True
speaker_add_comma_flag      = True
speaker_add_period_flag     = False
speaker_add_token_count     = False
speaker_scale               = "standardize"
speaker_group_rare          = True

# --- Speaker_job (best config: cv_mean_macro_f1=0.5923) ---
speaker_job_add_frequency       = False
speaker_job_add_is_rare         = False
speaker_job_add_grouped_job     = False
speaker_job_add_length_features = True
speaker_job_add_title_flag      = True
speaker_job_add_comma_flag      = True
speaker_job_add_slash_flag      = False
speaker_job_add_ampersand_flag  = False
speaker_job_add_token_count     = False
speaker_job_scale               = "none"
speaker_job_group_rare          = False  # must match add_grouped_job

# --- Party affiliation (best config: cv_mean_macro_f1=0.5988) ---
party_affiliation_add_frequency        = True
party_affiliation_add_is_rare          = True
party_affiliation_add_grouped_party    = True
party_affiliation_add_length_features  = True
party_affiliation_add_slash_flag       = True
party_affiliation_add_ampersand_flag   = True
party_affiliation_add_comma_flag       = False
party_affiliation_add_parentheses_flag = True
party_affiliation_add_token_count      = True
party_affiliation_add_is_major_party   = True
party_affiliation_add_is_institutional = True
party_affiliation_scale                = "standardize"

# --- State (best config: cv_mean_macro_f1=0.6087) ---
state_drop                = False
state_add_is_us_state     = True
state_add_frequency       = False
state_add_is_rare         = False
state_add_grouped_state   = True
state_group_rare          = True   # must match add_grouped_state
state_add_length_features = False
state_add_token_count     = True
state_add_has_us_words    = False
state_add_us_region       = True
state_scale               = "none"


# =============================================================================
# STEP 1 — PRECOMPUTE BASE FEATURES (runs once)
# FE module runs at defaults here (no fe_add_* flags active).
# =============================================================================

print("\n[SECTION] Precomputing base features (statement + subject + speaker + speaker_job + party + state)")
base_options = OneStepOptions(
    # Statement — best config (precomputed once)
    statement_vectorizer_type=statement_vectorizer_type,
    statement_vectorizer_max_features=statement_vectorizer_max_features,
    statement_stopword_removal=statement_stopword_removal,
    statement_stemmer=statement_stemmer,
    statement_lemmatizer=statement_lemmatizer,
    statement_add_lexical_features=statement_add_lexical_features,
    statement_scale=statement_scale,
    **STATEMENT_FIXED,
    # Subject
    subject_add_primary=subject_add_primary,
    subject_primary_strategy=subject_primary_strategy,
    subject_add_topic_count=subject_add_topic_count,
    subject_add_multiple_topics_flag=subject_add_multiple_topics_flag,
    subject_add_length_features=subject_add_length_features,
    subject_add_subject_frequency=subject_add_subject_frequency,
    subject_add_subject_is_rare=subject_add_subject_is_rare,
    subject_add_grouped_primary=subject_add_grouped_primary,
    subject_scale=subject_scale,
    # Speaker
    speaker_add_frequency=speaker_add_frequency,
    speaker_add_is_rare=speaker_add_is_rare,
    speaker_add_grouped_speaker=speaker_add_grouped_speaker,
    speaker_group_rare=speaker_group_rare,
    speaker_add_length_features=speaker_add_length_features,
    speaker_add_title_flag=speaker_add_title_flag,
    speaker_add_comma_flag=speaker_add_comma_flag,
    speaker_add_period_flag=speaker_add_period_flag,
    speaker_add_token_count=speaker_add_token_count,
    speaker_scale=speaker_scale,
    # Speaker job
    speaker_job_add_frequency=speaker_job_add_frequency,
    speaker_job_add_is_rare=speaker_job_add_is_rare,
    speaker_job_add_grouped_job=speaker_job_add_grouped_job,
    speaker_job_group_rare=speaker_job_group_rare,
    speaker_job_add_length_features=speaker_job_add_length_features,
    speaker_job_add_title_flag=speaker_job_add_title_flag,
    speaker_job_add_comma_flag=speaker_job_add_comma_flag,
    speaker_job_add_slash_flag=speaker_job_add_slash_flag,
    speaker_job_add_ampersand_flag=speaker_job_add_ampersand_flag,
    speaker_job_add_token_count=speaker_job_add_token_count,
    speaker_job_scale=speaker_job_scale,
    # Party
    party_affiliation_add_frequency=party_affiliation_add_frequency,
    party_affiliation_add_is_rare=party_affiliation_add_is_rare,
    party_affiliation_add_grouped_party=party_affiliation_add_grouped_party,
    party_affiliation_add_length_features=party_affiliation_add_length_features,
    party_affiliation_add_slash_flag=party_affiliation_add_slash_flag,
    party_affiliation_add_ampersand_flag=party_affiliation_add_ampersand_flag,
    party_affiliation_add_comma_flag=party_affiliation_add_comma_flag,
    party_affiliation_add_parentheses_flag=party_affiliation_add_parentheses_flag,
    party_affiliation_add_token_count=party_affiliation_add_token_count,
    party_affiliation_add_is_major_party=party_affiliation_add_is_major_party,
    party_affiliation_add_is_institutional=party_affiliation_add_is_institutional,
    party_affiliation_scale=party_affiliation_scale,
    # State — best config
    state_drop=state_drop,
    state_add_is_us_state=state_add_is_us_state,
    state_add_frequency=state_add_frequency,
    state_add_is_rare=state_add_is_rare,
    state_add_grouped_state=state_add_grouped_state,
    state_group_rare=state_group_rare,
    state_add_length_features=state_add_length_features,
    state_add_token_count=state_add_token_count,
    state_add_has_us_words=state_add_has_us_words,
    state_add_us_region=state_add_us_region,
    state_scale=state_scale,
    # FE: all defaults (no add_* flags) — this is what we're sweeping
)

df_base   = preprocess_one_step(df, options=base_options)
y_all     = df_base["label"]
X_base    = df_base.drop(columns=["label"]).select_dtypes(exclude="object")
base_cols = set(X_base.columns)

print(f"Base feature matrix: {X_base.shape[0]:,} rows × {X_base.shape[1]:,} features")


# =============================================================================
# FE OPTION GRID
#
# Excluded (target-encoding leakage — compute only inside CV folds):
#   fe_add_speaker_true_rate, fe_add_subject_true_rate, fe_add_party_true_rate
#
# Interaction features produce string columns; they are frequency-encoded
# automatically in the loop before fitting LR.
#
# fe_add_sentiment requires: pip install textblob
# =============================================================================

# Fixed at False: high-cardinality interactions, noise for linear models.
# Fixed at True: established deception-detection signals, always include.
_FE_FIXED = {
    "fe_add_speaker_job_subject":          False,  # 3-way combo, too high cardinality
    "fe_add_speaker_statement_len_bucket": False,  # speaker × bucket redundant with base speaker freq
    "fe_add_negation_count":               True,
    "fe_add_hedge_count":                  True,
    "fe_add_absolutist_count":             True,
    "fe_add_numeral_count":                True,
}

# Used to determine whether fe_scale has any effect (only numeric text-style keys).
_TEXT_STYLE_KEYS = {
    "fe_add_proper_noun_count",
    "fe_add_readability",
    "fe_add_sentiment",
}

FE_GRID = {
    # --- Interaction features (string output → freq-encoded) ---
    "fe_add_speaker_subject":  [False, True],
    "fe_add_speaker_party":    [False, True],
    "fe_add_subject_party":    [False, True],
    "fe_add_state_party":      [False, True],
    # --- Text-style features (numeric, no leakage risk) ---
    "fe_add_proper_noun_count": [False, True],
    "fe_add_readability":       [False, True],
    "fe_add_sentiment":         [False, True],
    # --- Scale ---
    "fe_scale":                 ["none", "standardize"],
}

# Full grid is ~240 configs (2^4 interaction × 2^3 text-style × 2 scale, deduped).
# Set MAX_RUNS to None to run all.
MAX_RUNS = None


def generate_fe_configs():
    keys, seen, result = list(FE_GRID.keys()), set(), []
    for values in product(*FE_GRID.values()):
        cfg = dict(zip(keys, values))

        # fe_scale has no effect when all numeric (text-style) FE features are off.
        if not any(cfg[k] for k in _TEXT_STYLE_KEYS):
            cfg["fe_scale"] = "none"

        frozen = tuple(sorted(cfg.items()))
        if frozen not in seen:
            seen.add(frozen)
            result.append(cfg)
    return result


fe_configs = generate_fe_configs()
if MAX_RUNS is not None:
    fe_configs = fe_configs[:MAX_RUNS]

print(f"\nTotal FE configurations to run: {len(fe_configs)}")
print("(Each run is fast — no TF-IDF or lemmatization reruns)")
print("(Set MAX_RUNS or narrow FE_GRID values to limit the sweep)\n")


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

CLASS_WEIGHT = {0: 1.42, 1: 0.77}
C_VALUE      = 1.0
MAX_ITER     = 300
CV_FOLDS     = 3


# =============================================================================
# STEP 2 — SWEEP LOOP
# Each iteration only reruns the FE module (fast).
# =============================================================================

all_results = []

for run_idx, fe_cfg in enumerate(fe_configs, 1):
    print(f"\n{'='*60}")
    print(f"Run {run_idx} / {len(fe_configs)}")
    print(f"FE config: {fe_cfg}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Fast FE-only preprocessing pass
    # Statement module runs basic cleaning only (no vectorizer, no lemmatizer).
    # Subject / speaker / party run at defaults — no extra columns added.
    # -------------------------------------------------------------------------
    iter_options = OneStepOptions(
        # Statement: skip expensive parts (already in base features)
        statement_vectorizer_type="none",
        statement_lemmatizer="none",
        statement_stopword_removal=False,
        statement_add_lexical_features=False,
        # Subject / speaker / party: defaults only (already in base features)
        # FE: fixed defaults + current config under test
        **_FE_FIXED,
        **fe_cfg,
    )
    df_iter    = preprocess_one_step(df, options=iter_options)
    X_iter_raw = df_iter.drop(columns=["label"], errors="ignore")

    # Frequency-encode interaction string columns before dropping object types.
    # Interaction features produce strings like "obama__health-care"; LR needs
    # numeric input, so we map each value to its frequency in the dataset.
    fe_str_cols = [
        c for c in X_iter_raw.columns
        if c.startswith("fe_") and X_iter_raw[c].dtype == object
    ]
    for col in fe_str_cols:
        freq_map           = X_iter_raw[col].value_counts()
        X_iter_raw[col]    = X_iter_raw[col].map(freq_map).fillna(0).astype(float)

    X_iter   = X_iter_raw.select_dtypes(exclude="object")
    new_cols = [c for c in X_iter.columns if c not in base_cols]

    if new_cols:
        X = pd.concat([X_base, X_iter[new_cols]], axis=1)
    else:
        X = X_base.copy()

    y = y_all
    print(f"Features: {X_base.shape[1]} base + {len(new_cols)} FE = {X.shape[1]} total")

    # -------------------------------------------------------------------------
    # Train / holdout split
    # -------------------------------------------------------------------------
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # W&B run
    # -------------------------------------------------------------------------
    # wandb_run = wandb.init(
    #     project="truth-classifier-lr-fe-sweep",
    #     config={
    #         "model":      "LogisticRegression",
    #         "solver":     "liblinear",
    #         "C":          C_VALUE,
    #         "max_iter":   MAX_ITER,
    #         "cv_folds":   CV_FOLDS,
    #         "n_features": int(X_trainval.shape[1]),
    #         **{f"fe__{k}": str(v) for k, v in fe_cfg.items()},
    #     },
    #     reinit=True,
    # )

    # -------------------------------------------------------------------------
    # 3-fold CV
    # -------------------------------------------------------------------------
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    fold_macro_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        fold_model = LogisticRegression(
            C=C_VALUE,
            solver="liblinear",
            max_iter=MAX_ITER,
            class_weight=CLASS_WEIGHT,
            random_state=42,
        )
        fold_model.fit(X_trainval.iloc[train_idx], y_trainval.iloc[train_idx])
        macro_f1 = f1_score(
            y_trainval.iloc[val_idx],
            fold_model.predict(X_trainval.iloc[val_idx]),
            average="macro",
            zero_division=0,
        )
        fold_macro_f1s.append(macro_f1)
        # wandb.log({"cv/fold": fold, "cv/macro_f1": macro_f1})

    cv_mean = float(np.mean(fold_macro_f1s))
    cv_std  = float(np.std(fold_macro_f1s))
    print(f"CV macro_f1: {cv_mean:.4f} ± {cv_std:.4f}")

    # wandb.log({"cv/mean_macro_f1": cv_mean, "cv/std_macro_f1": cv_std})
    # wandb_run.summary["cv_mean_macro_f1"] = cv_mean
    # wandb_run.finish()

    all_results.append({
        "run":              run_idx,
        "cv_mean_macro_f1": cv_mean,
        "cv_std_macro_f1":  cv_std,
        "n_fe_cols":        len(new_cols),
        **fe_cfg,
    })

    sleep(1)


# =============================================================================
# RESULTS SUMMARY + HOLDOUT EVALUATION FOR THE WINNER
# =============================================================================

print("\n" + "="*70)
print("SWEEP COMPLETE — ranked by CV mean macro F1 (descending)")
print("="*70)

results_df = pd.DataFrame(all_results).sort_values("cv_mean_macro_f1", ascending=False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(results_df.to_string(index=False))

best_cfg = results_df.iloc[0]
print(f"\nBest FE config  (cv_mean_macro_f1={best_cfg['cv_mean_macro_f1']:.4f}):")
for k in FE_GRID:
    print(f"  {k}: {best_cfg[k]}")


# --- Reconstruct best feature matrix and evaluate on holdout ---
print("\n[SECTION] Evaluating best config on holdout set")

best_fe_cfg = {k: best_cfg[k] for k in FE_GRID}

iter_options = OneStepOptions(
    statement_vectorizer_type="none",
    statement_lemmatizer="none",
    statement_stopword_removal=False,
    statement_add_lexical_features=False,
    **best_fe_cfg,
)
df_best    = preprocess_one_step(df, options=iter_options)
X_best_raw = df_best.drop(columns=["label"], errors="ignore")

fe_str_cols = [
    c for c in X_best_raw.columns
    if c.startswith("fe_") and X_best_raw[c].dtype == object
]
for col in fe_str_cols:
    freq_map        = X_best_raw[col].value_counts()
    X_best_raw[col] = X_best_raw[col].map(freq_map).fillna(0).astype(float)

X_best   = X_best_raw.select_dtypes(exclude="object")
new_cols = [c for c in X_best.columns if c not in base_cols]

X_final = pd.concat([X_base, X_best[new_cols]], axis=1) if new_cols else X_base.copy()
y_final = y_all

X_trainval_f, X_holdout_f, y_trainval_f, y_holdout_f = train_test_split(
    X_final, y_final, test_size=0.2, random_state=42, stratify=y_final
)

final_model = LogisticRegression(
    C=C_VALUE, solver="liblinear", max_iter=MAX_ITER,
    class_weight=CLASS_WEIGHT, random_state=42,
)
final_model.fit(X_trainval_f, y_trainval_f)

y_pred_h = final_model.predict(X_holdout_f)
y_prob_h = final_model.predict_proba(X_holdout_f)[:, 1]

print(f"Holdout macro_f1 : {f1_score(y_holdout_f, y_pred_h, average='macro', zero_division=0):.4f}")
print(f"Holdout roc_auc  : {roc_auc_score(y_holdout_f, y_prob_h):.4f}")
print(f"Holdout accuracy : {accuracy_score(y_holdout_f, y_pred_h):.4f}")
print(f"\n{classification_report(y_holdout_f, y_pred_h, target_names=['true(0)', 'false(1)'])}")
