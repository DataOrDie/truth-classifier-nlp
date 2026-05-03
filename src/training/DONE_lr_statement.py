"""Feature sweep over statement_ds.py preprocessing option combinations.

Unlike the other module sweeps, the statement module IS the expensive part
(lemmatization + TF-IDF fitting), so it must rerun every iteration — there is
no way to precompute it. The base pass precomputes everything else (subject,
speaker, party, FE) so those never repeat.

Expect ~30-120 seconds per run depending on the normalization and vectorizer.

Options fixed for all runs (not swept):
  statement_lower, remove_html, remove_urls, repair_polluted  → always True
  statement_keep_negations                                     → always True
  statement_vectorizer_min_df                                  → fixed at 2
  statement_vectorizer_max_df                                  → fixed at 0.9
  statement_remove_punctuation                                 → fixed at False
  statement_replace_numbers                                    → fixed at False
  statement_add_ner_features                                   → excluded (spaCy, very slow)
  statement_add_spelling_errors                                → excluded (slow)
  statement_add_pollution_features                             → excluded (low signal post-repair)
  statement_add_rare_token_features                            → excluded (minor signal)
  statement_vectorizer_type = "embeddings"                     → excluded (sentence-transformers, very slow)

Total unique configs generated (after filtering invalid stemmer+lemmatizer combos): ~192
Set MAX_RUNS to a small number (e.g. 20) for a quick first pass.

Best statement config  (cv_mean_macro_f1=0.6079):
  statement_vectorizer_type: tfidf
  statement_vectorizer_max_features: 5000
  statement_stopword_removal: False
  statement_stemmer: porter
  statement_lemmatizer: none
  statement_add_lexical_features: True
  statement_scale: standardize

[SECTION] Evaluating best config on holdout set
Holdout macro_f1 : 0.6021
Holdout roc_auc  : 0.6505
Holdout accuracy : 0.6184
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
wandb.login()


# -----------------------------------------------------------------------------
# Load raw data once
# -----------------------------------------------------------------------------

print("[SECTION] Loading data")
df = pd.read_csv(project_root / "data" / "train.csv")
print(f"Loaded {len(df):,} rows")


# =============================================================================
# BASE CONFIG VARIABLES
# Subject, speaker, party, and FE stay fixed — computed ONCE.
# Statement options are left out of the base pass entirely
# (statement_vectorizer_type="none", no lexical features).
# =============================================================================


# --- Subject (best config: cv_mean_macro_f1=0.5999, holdout_macro_f1=0.6036) ---
subject_add_primary              = True
subject_primary_strategy         = "most_frequent"
subject_add_topic_count          = True
subject_add_multiple_topics_flag = False
subject_add_length_features      = False
subject_add_subject_frequency    = False  # adding frequency actually hurts performance in the full sweep, even though it helps in the ablation — likely overfitting to training distribution quirks. Exclude from base features to avoid polluting the TF-IDF with a noisy feature.
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

# Best party_affiliation config  (cv_mean_macro_f1=0.5988):
party_affiliation_add_frequency = True
party_affiliation_add_is_rare = True
party_affiliation_add_grouped_party = True
party_affiliation_add_length_features = True
party_affiliation_add_slash_flag = True
party_affiliation_add_ampersand_flag = True
party_affiliation_add_comma_flag = False
party_affiliation_add_parentheses_flag = True
party_affiliation_add_token_count = True
party_affiliation_add_is_major_party = True
party_affiliation_add_is_institutional = True
party_affiliation_scale = "standardize"

# --- Feature engineering ---
fe_add_negation_count   = True
fe_add_hedge_count      = True
fe_add_absolutist_count = True
fe_add_readability      = True
fe_scale                = "standardize"

# Statement options that are FIXED across all runs (not swept)
STATEMENT_FIXED = {
    "statement_lower":                       True,
    "statement_remove_html":                 True,
    "statement_remove_urls":                 True,
    "statement_repair_polluted_statements":  True,
    "statement_keep_negations":              True,   # preserves "not", "never", etc.
    "statement_vectorizer_min_df":           2,
    "statement_vectorizer_max_df":           0.9,
    "statement_remove_punctuation":          False,  # hurts TF-IDF quality for linear models
    "statement_replace_numbers":             False,
}


# =============================================================================
# STEP 1 — PRECOMPUTE BASE FEATURES (runs once)
# No statement vectorization or lexical features here — those vary per run.
# =============================================================================

print("\n[SECTION] Precomputing base features (subject + speaker + party + FE)")
base_options = OneStepOptions(
    # Statement: skip everything except basic cleaning
    statement_vectorizer_type="none",
    statement_add_lexical_features=False,
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
    # Feature engineering
    fe_add_negation_count=fe_add_negation_count,
    fe_add_hedge_count=fe_add_hedge_count,
    fe_add_absolutist_count=fe_add_absolutist_count,
    fe_add_readability=fe_add_readability,
    fe_scale=fe_scale,
)

df_base   = preprocess_one_step(df, options=base_options)
y_all     = df_base["label"]
X_base    = df_base.drop(columns=["label"]).select_dtypes(exclude="object")
base_cols = set(X_base.columns)

print(f"Base feature matrix: {X_base.shape[0]:,} rows × {X_base.shape[1]:,} features")


# =============================================================================
# STATEMENT OPTION GRID
#
# Excluded from the grid (see module docstring for reasons):
#   statement_vectorizer_type = "embeddings"
#   statement_add_ner_features, add_spelling_errors,
#   add_pollution_features, add_rare_token_features
#
# stemmer and lemmatizer are mutually exclusive — both do morphological
# normalization. Invalid pairs are filtered out in generate_statement_configs().
# =============================================================================

STATEMENT_GRID = {
    "statement_vectorizer_type":         ["tfidf", "bigram", "binary"],
    "statement_vectorizer_max_features": [5000, 10000],
    "statement_stopword_removal":        [False, True],
    "statement_stemmer":                 ["none", "porter", "snowball"],
    "statement_lemmatizer":              ["none", "wordnet"],
    "statement_add_lexical_features":    [False, True],
    "statement_scale":                   ["none", "standardize"],
}

# Set to an integer to cap the number of runs. None = run all (~192 configs).
# Recommended: start with MAX_RUNS=20 to test, then increase.
MAX_RUNS = None


def generate_statement_configs():
    """Return valid, deduplicated statement option dicts from STATEMENT_GRID."""
    keys, seen, result = list(STATEMENT_GRID.keys()), set(), []
    for values in product(*STATEMENT_GRID.values()):
        cfg = dict(zip(keys, values))

        # Stemmer and lemmatizer are mutually exclusive: applying both is wrong.
        if cfg["statement_stemmer"] != "none" and cfg["statement_lemmatizer"] != "none":
            continue

        frozen = tuple(sorted(cfg.items()))
        if frozen not in seen:
            seen.add(frozen)
            result.append(cfg)
    return result


statement_configs = generate_statement_configs()
if MAX_RUNS is not None:
    statement_configs = statement_configs[:MAX_RUNS]

print(f"\nTotal statement configurations to run: {len(statement_configs)}")
print("WARNING: each run re-fits lemmatization + TF-IDF — expect 30-120s per run")
print("(Adjust STATEMENT_GRID or set MAX_RUNS to limit)\n")


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

CLASS_WEIGHT = {0: 1.42, 1: 0.77}
C_VALUE      = 1.0
MAX_ITER     = 300
CV_FOLDS     = 3


# =============================================================================
# STEP 2 — SWEEP LOOP
# Each iteration reruns statement preprocessing with varying options.
# Subject / speaker / party / FE are at defaults (already in base features).
# =============================================================================

all_results = []

for run_idx, stmt_cfg in enumerate(statement_configs, 1):
    print(f"\n{'='*60}")
    print(f"Run {run_idx} / {len(statement_configs)}")
    print(f"Statement config: {stmt_cfg}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Statement-only preprocessing pass (varying config)
    # All other modules run at defaults — no extra columns added.
    # -------------------------------------------------------------------------
    iter_options = OneStepOptions(
        # Statement: current config under test + fixed values
        **stmt_cfg,
        **STATEMENT_FIXED,
        # Subject / speaker / party / FE: defaults only (already in base features)
    )
    df_iter  = preprocess_one_step(df, options=iter_options)
    X_iter   = df_iter.drop(columns=["label"], errors="ignore").select_dtypes(exclude="object")
    new_cols = [c for c in X_iter.columns if c not in base_cols]

    if new_cols:
        X = pd.concat([X_base, X_iter[new_cols]], axis=1)
    else:
        X = X_base.copy()

    y = y_all
    print(f"Features: {X_base.shape[1]} base + {len(new_cols)} statement = {X.shape[1]} total")

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
    #     project="truth-classifier-lr-statement-sweep",
    #     config={
    #         "model":      "LogisticRegression",
    #         "solver":     "liblinear",
    #         "C":          C_VALUE,
    #         "max_iter":   MAX_ITER,
    #         "cv_folds":   CV_FOLDS,
    #         "n_features": int(X_trainval.shape[1]),
    #         **{f"stmt__{k}": str(v) for k, v in stmt_cfg.items()},
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
        "n_statement_cols": len(new_cols),
        **stmt_cfg,
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
print(f"\nBest statement config  (cv_mean_macro_f1={best_cfg['cv_mean_macro_f1']:.4f}):")
for k in STATEMENT_GRID:
    print(f"  {k}: {best_cfg[k]}")


# --- Reconstruct best feature matrix and evaluate on holdout ---
print("\n[SECTION] Evaluating best config on holdout set")

best_stmt_cfg = {k: best_cfg[k] for k in STATEMENT_GRID}

iter_options = OneStepOptions(
    **best_stmt_cfg,
    **STATEMENT_FIXED,
)
df_best  = preprocess_one_step(df, options=iter_options)
X_best   = df_best.drop(columns=["label"], errors="ignore").select_dtypes(exclude="object")
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
