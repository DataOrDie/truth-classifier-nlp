"""Automated sweep over all subject.py preprocessing option combinations.

For every valid combination of subject options, trains a LogisticRegression
and logs the result to a separate W&B run. Primary metric: macro F1.

Two subject options are intentionally excluded from the sweep:
  - subject_add_topic_list              → produces a raw Python list, not a numeric column
  - subject_add_subject_primary_true_rate → target-encoding leakage risk in a simple CV setup

Total unique configs generated (after dependency filtering): ~272
Set MAX_RUNS to an integer to cap the sweep (useful for quick testing).
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
wandb.login()


# -----------------------------------------------------------------------------
# Load data once — reused across all runs
# -----------------------------------------------------------------------------

print("[SECTION] Loading data")
df = pd.read_csv(project_root / "data" / "train.csv")
print(f"Loaded {len(df):,} rows")


# =============================================================================
# BASE CONFIG VARIABLES
# Statement, speaker, party, and feature-engineering options stay fixed.
# These are the recommended [linear] settings from PREPROCESSING_OPTIONS.md.
# =============================================================================

# --- Statement ---
statement_vectorizer_type         = "tfidf"
statement_vectorizer_max_features = 10000
statement_vectorizer_min_df       = 2
statement_vectorizer_max_df       = 0.9
statement_stopword_removal        = True
statement_lemmatizer              = "wordnet"
statement_keep_negations          = True
statement_add_lexical_features    = True
statement_scale                   = "standardize"

# --- Speaker ---
speaker_add_frequency       = True
speaker_add_grouped_speaker = True
speaker_group_rare          = True
speaker_scale               = "standardize"

# --- Party ---
party_affiliation_add_is_major_party = True
party_affiliation_add_frequency      = True

# --- Feature engineering ---
fe_add_negation_count   = True
fe_add_hedge_count      = True
fe_add_absolutist_count = True
fe_add_readability      = True
fe_scale                = "standardize"


# =============================================================================
# SUBJECT OPTION GRID
#
# Every key is a valid OneStepOptions field from subject.py.
# Values list the candidates to try for that option.
# Reduce any list to a single value to skip variation on that option.
# =============================================================================

SUBJECT_GRID = {
    "subject_add_primary":              [False, True],
    "subject_primary_strategy":         ["first", "most_frequent"],
    "subject_add_topic_count":          [False, True],
    "subject_add_multiple_topics_flag": [False, True],
    "subject_add_length_features":      [False, True],
    "subject_add_subject_frequency":    [False, True],
    "subject_add_subject_is_rare":      [False, True],
    "subject_add_grouped_primary":      [False, True],
    "subject_scale":                    ["none", "standardize"],
}

# Set to an integer to cap the number of runs (useful for testing).
# Set to None to run all generated configurations.
MAX_RUNS = None


# =============================================================================
# CONFIG GENERATOR
# =============================================================================

def generate_subject_configs():
    """Return deduplicated list of valid subject option dicts from SUBJECT_GRID."""
    keys   = list(SUBJECT_GRID.keys())
    seen   = set()
    result = []

    for values in product(*SUBJECT_GRID.values()):
        cfg = dict(zip(keys, values))

        # Options that require subject_add_primary=True have no effect otherwise.
        if not cfg["subject_add_primary"]:
            cfg["subject_primary_strategy"]      = "first"   # no-op
            cfg["subject_add_subject_frequency"] = False
            cfg["subject_add_subject_is_rare"]   = False
            cfg["subject_add_grouped_primary"]   = False

        # subject_group_rare must be True whenever subject_add_grouped_primary=True.
        cfg["subject_group_rare"] = cfg["subject_add_grouped_primary"]

        frozen = tuple(sorted(cfg.items()))
        if frozen not in seen:
            seen.add(frozen)
            result.append(cfg)

    return result


subject_configs = generate_subject_configs()
if MAX_RUNS is not None:
    subject_configs = subject_configs[:MAX_RUNS]

print(f"\nTotal subject configurations to run: {len(subject_configs)}")
print("(Set MAX_RUNS or narrow SUBJECT_GRID values to limit the sweep)\n")


# =============================================================================
# MODEL HYPERPARAMETERS (fixed across all runs)
# =============================================================================

CLASS_WEIGHT = {0: 1.42, 1: 0.77}   # from CLAUDE.md — handles 35%/65% imbalance
C_VALUE      = 1.0
MAX_ITER     = 1000


# =============================================================================
# TRAINING LOOP
# =============================================================================

all_results = []

for run_idx, subj_cfg in enumerate(subject_configs, 1):
    print(f"\n{'='*60}")
    print(f"Run {run_idx} / {len(subject_configs)}")
    print(f"Subject config: {subj_cfg}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Build OneStepOptions: base variables + current subject config
    # -------------------------------------------------------------------------
    options = OneStepOptions(
        # Statement
        statement_vectorizer_type=statement_vectorizer_type,
        statement_vectorizer_max_features=statement_vectorizer_max_features,
        statement_vectorizer_min_df=statement_vectorizer_min_df,
        statement_vectorizer_max_df=statement_vectorizer_max_df,
        statement_stopword_removal=statement_stopword_removal,
        statement_lemmatizer=statement_lemmatizer,
        statement_keep_negations=statement_keep_negations,
        statement_add_lexical_features=statement_add_lexical_features,
        statement_scale=statement_scale,
        # Subject — spread the current config dict directly
        **subj_cfg,
        # Speaker
        speaker_add_frequency=speaker_add_frequency,
        speaker_add_grouped_speaker=speaker_add_grouped_speaker,
        speaker_group_rare=speaker_group_rare,
        speaker_scale=speaker_scale,
        # Party
        party_affiliation_add_is_major_party=party_affiliation_add_is_major_party,
        party_affiliation_add_frequency=party_affiliation_add_frequency,
        # Feature engineering
        fe_add_negation_count=fe_add_negation_count,
        fe_add_hedge_count=fe_add_hedge_count,
        fe_add_absolutist_count=fe_add_absolutist_count,
        fe_add_readability=fe_add_readability,
        fe_scale=fe_scale,
    )

    # -------------------------------------------------------------------------
    # Preprocess
    # -------------------------------------------------------------------------
    print("[SECTION] Preprocessing data with current subject config")
    df_proc = preprocess_one_step(df, options=options)

    y = df_proc["label"]
    X = df_proc.drop(columns=["label"])

    # Drop string columns — LogisticRegression only accepts numeric input
    string_cols = X.select_dtypes(include="object").columns.tolist()
    if string_cols:
        X = X.drop(columns=string_cols)

    print(f"Feature matrix: {X.shape[0]:,} rows × {X.shape[1]:,} features")

    # -------------------------------------------------------------------------
    # Train / holdout split — same random_state every run for fair comparison
    # -------------------------------------------------------------------------
    print("[SECTION] Train/holdout split")
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # Init W&B run — log base config + subject config as hyperparams
    # -------------------------------------------------------------------------
    wandb_run = wandb.init(
        project="truth-classifier-lr-subject-sweep",
        config={
            "model":       "LogisticRegression",
            "C":           C_VALUE,
            "max_iter":    MAX_ITER,
            "n_features":  int(X_trainval.shape[1]),
            **{f"subj__{k}": str(v) for k, v in subj_cfg.items()},
        },
        reinit=True,
    )

    # -------------------------------------------------------------------------
    # 5-fold stratified cross-validation (primary metric: macro F1)
    # -------------------------------------------------------------------------
    print("[SECTION] 5-fold stratified cross-validation")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_macro_f1s = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        fold_model = LogisticRegression(
            C=C_VALUE,
            solver="lbfgs",
            max_iter=MAX_ITER,
            class_weight=CLASS_WEIGHT,
            random_state=42,
            n_jobs=-1,
        )
        print(f"  Training fold {fold}...")
        fold_model.fit(X_trainval.iloc[train_idx], y_trainval.iloc[train_idx])
        y_pred_fold = fold_model.predict(X_trainval.iloc[val_idx])
        macro_f1    = f1_score(y_trainval.iloc[val_idx], y_pred_fold, average="macro", zero_division=0)
        fold_macro_f1s.append(macro_f1)
        wandb.log({"cv/fold": fold, "cv/macro_f1": macro_f1})

    cv_mean = float(np.mean(fold_macro_f1s))
    cv_std  = float(np.std(fold_macro_f1s))
    print(f"CV macro_f1: {cv_mean:.4f} ± {cv_std:.4f}")
    wandb.log({"cv/mean_macro_f1": cv_mean, "cv/std_macro_f1": cv_std})

    # -------------------------------------------------------------------------
    # Refit on full train/val, evaluate once on holdout
    # -------------------------------------------------------------------------
    print("[SECTION] Final training on full train/val and evaluation on holdout")
    final_model = LogisticRegression(
        C=C_VALUE,
        solver="lbfgs",
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=42,
        n_jobs=-1,
    )
    final_model.fit(X_trainval, y_trainval)
    print("Evaluating on holdout set...")

    y_pred_holdout = final_model.predict(X_holdout)
    y_prob_holdout = final_model.predict_proba(X_holdout)[:, 1]

    holdout_macro_f1 = f1_score(y_holdout, y_pred_holdout, average="macro", zero_division=0)
    holdout_roc_auc  = roc_auc_score(y_holdout, y_prob_holdout)
    holdout_accuracy = accuracy_score(y_holdout, y_pred_holdout)

    print(
        f"Holdout  macro_f1={holdout_macro_f1:.4f}  "
        f"roc_auc={holdout_roc_auc:.4f}  "
        f"acc={holdout_accuracy:.4f}"
    )

    wandb.log({
        "holdout/macro_f1": holdout_macro_f1,
        "holdout/roc_auc":  holdout_roc_auc,
        "holdout/accuracy": holdout_accuracy,
    })
    wandb_run.summary["macro_f1"]         = holdout_macro_f1
    wandb_run.summary["cv_mean_macro_f1"] = cv_mean
    wandb_run.finish()

    all_results.append({
        "run":              run_idx,
        "cv_mean_macro_f1": cv_mean,
        "cv_std_macro_f1":  cv_std,
        "holdout_macro_f1": holdout_macro_f1,
        "holdout_roc_auc":  holdout_roc_auc,
        **subj_cfg,
    })

    sleep(2)  # brief pause between runs to ensure W&B logging completes properly


# =============================================================================
# RESULTS SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SWEEP COMPLETE — ranked by CV mean macro F1 (descending)")
print("="*70)

results_df = pd.DataFrame(all_results).sort_values("cv_mean_macro_f1", ascending=False)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\nBest subject config  (cv_mean_macro_f1={best['cv_mean_macro_f1']:.4f}, "
      f"holdout_macro_f1={best['holdout_macro_f1']:.4f}):")
for k in SUBJECT_GRID:
    print(f"  {k}: {best[k]}")

best = results_df.iloc[0]
print(f"\nBest subject config  (cv_mean_macro_f1={best['cv_mean_macro_f1']:.4f}, "
      f"holdout_macro_f1={best['holdout_macro_f1']:.4f}):")
for k in SUBJECT_GRID:
    print(f"  {k}: {best[k]}")
