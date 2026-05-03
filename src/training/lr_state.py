"""Feature sweep over all state.py preprocessing option combinations.

Speed strategy:
  - Statement TF-IDF + lemmatization is precomputed ONCE.
  - Subject, speaker, and speaker_job features (recommended configs) are also in the base pass.
  - Each iteration only reruns the state module (cheap: frequency/groupby/flags).
  - 3-fold CV during the sweep; holdout evaluated only for the winning config.
  - liblinear solver + lower max_iter for faster ranking fits.

Note: state_drop=True is a valid option (drop the column entirely) and is
included in the grid as a standalone config with all add_* flags forced False.

Best state config  (cv_mean_macro_f1=0.6087):
  state_drop: False
  state_add_is_us_state: True
  state_add_frequency: False
  state_add_is_rare: False
  state_add_grouped_state: True
  state_add_length_features: False
  state_add_token_count: True
  state_add_has_us_words: False
  state_add_us_region: True
  state_scale: none
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
# Statement, subject, speaker, speaker_job, party, and FE stay fixed — computed ONCE.
# State options are left at defaults (no features added in the base pass).
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

# --- Feature engineering ---
fe_add_negation_count   = True
fe_add_hedge_count      = True
fe_add_absolutist_count = True
fe_add_readability      = True
fe_scale                = "standardize"


# =============================================================================
# STEP 1 — PRECOMPUTE BASE FEATURES (runs once)
# State options are at defaults here (only state_info_clean is produced).
# =============================================================================

print("\n[SECTION] Precomputing base features (statement + subject + speaker + party + FE)")
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
    # State — defaults only (no add_* flags), so no state features added in base pass
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
# STATE OPTION GRID
# =============================================================================

STATE_GRID = {
    "state_drop":             [False, True],
    "state_add_is_us_state":  [False, True],
    "state_add_frequency":    [False, True],
    "state_add_is_rare":      [False, True],
    "state_add_grouped_state":[False, True],
    "state_add_length_features": [False, True],
    "state_add_token_count":  [False, True],
    "state_add_has_us_words": [False, True],
    "state_add_us_region":    [False, True],
    "state_scale":            ["none", "standardize"],
}

# Set to an integer to cap the number of runs. None = run all.
MAX_RUNS = None


def generate_state_configs():
    keys, seen, result = list(STATE_GRID.keys()), set(), []
    for values in product(*STATE_GRID.values()):
        cfg = dict(zip(keys, values))

        # When state_drop=True all add_* flags are meaningless — the column is gone.
        if cfg["state_drop"]:
            for k in list(cfg):
                if k.startswith("state_add_"):
                    cfg[k] = False
            cfg["state_scale"]      = "none"
            cfg["state_group_rare"] = False
        else:
            # state_group_rare must be True whenever state_add_grouped_state=True.
            cfg["state_group_rare"] = cfg["state_add_grouped_state"]

        frozen = tuple(sorted(cfg.items()))
        if frozen not in seen:
            seen.add(frozen)
            result.append(cfg)
    return result


state_configs = generate_state_configs()
if MAX_RUNS is not None:
    state_configs = state_configs[:MAX_RUNS]

print(f"\nTotal state configurations to run: {len(state_configs)}")
print("(Adjust STATE_GRID or set MAX_RUNS to limit)\n")


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

CLASS_WEIGHT = {0: 1.42, 1: 0.77}
C_VALUE      = 1.0
MAX_ITER     = 300
CV_FOLDS     = 3


# =============================================================================
# STEP 2 — SWEEP LOOP
# Each iteration only reruns the state module (no TF-IDF, no lemmatization).
# =============================================================================

all_results = []

for run_idx, state_cfg in enumerate(state_configs, 1):
    print(f"\n{'='*60}")
    print(f"Run {run_idx} / {len(state_configs)}")
    print(f"State config: {state_cfg}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Fast state-only preprocessing pass
    # Statement module runs basic cleaning only (vectorizer + lemmatizer off).
    # All other modules run at defaults — no extra columns added.
    # -------------------------------------------------------------------------
    iter_options = OneStepOptions(
        # Statement: skip expensive parts (already in base features)
        statement_vectorizer_type="none",
        statement_lemmatizer="none",
        statement_stopword_removal=False,
        statement_add_lexical_features=False,
        # Subject / speaker / speaker_job / party / FE: defaults only (already in base)
        # State: current config under test
        **state_cfg,
    )
    df_iter  = preprocess_one_step(df, options=iter_options)
    X_iter   = df_iter.drop(columns=["label"], errors="ignore").select_dtypes(exclude="object")
    new_cols = [c for c in X_iter.columns if c not in base_cols]

    if new_cols:
        X = pd.concat([X_base, X_iter[new_cols]], axis=1)
    else:
        X = X_base.copy()

    y = y_all
    print(f"Features: {X_base.shape[1]} base + {len(new_cols)} state = {X.shape[1]} total")

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
    #     project="truth-classifier-lr-state-sweep",
    #     config={
    #         "model":      "LogisticRegression",
    #         "solver":     "liblinear",
    #         "C":          C_VALUE,
    #         "max_iter":   MAX_ITER,
    #         "cv_folds":   CV_FOLDS,
    #         "n_features": int(X_trainval.shape[1]),
    #         **{f"state__{k}": str(v) for k, v in state_cfg.items()},
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
        "n_state_cols":     len(new_cols),
        **state_cfg,
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
print(f"\nBest state config  (cv_mean_macro_f1={best_cfg['cv_mean_macro_f1']:.4f}):")
for k in STATE_GRID:
    print(f"  {k}: {best_cfg[k]}")


# --- Reconstruct best feature matrix and evaluate on holdout ---
print("\n[SECTION] Evaluating best config on holdout set")

best_state_cfg = {k: best_cfg[k] for k in list(STATE_GRID.keys()) + ["state_group_rare"]}

iter_options = OneStepOptions(
    statement_vectorizer_type="none",
    statement_lemmatizer="none",
    statement_stopword_removal=False,
    statement_add_lexical_features=False,
    **best_state_cfg,
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
