"""Feature sweep over all speaker_job.py preprocessing option combinations.

Speed strategy:
  - Statement TF-IDF + lemmatization is precomputed ONCE.
  - Subject and speaker features (recommended configs) are also in the base pass.
  - Each iteration only reruns the speaker_job module (cheap: frequency/groupby).
  - 3-fold CV during the sweep; holdout evaluated only for the winning config.
  - liblinear solver + lower max_iter for faster ranking fits.

One speaker_job option is excluded:
  - speaker_job_add_job_primary_true_rate → target-encoding leakage risk

------------------------------------------------------------------------------

Best speaker_job config  (cv_mean_macro_f1=0.5923):
  speaker_job_add_frequency: False
  speaker_job_add_is_rare: False
  speaker_job_add_grouped_job: False
  speaker_job_add_length_features: True
  speaker_job_add_title_flag: True
  speaker_job_add_comma_flag: True
  speaker_job_add_slash_flag: False
  speaker_job_add_ampersand_flag: False
  speaker_job_add_token_count: False
  speaker_job_scale: none

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
# Statement, subject, speaker, party, and FE stay fixed — computed ONCE.
# Speaker_job options are left at defaults (no features added in the base pass).
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

# --- Subject (fixed recommended config) ---
# subject_add_primary           = True
# subject_add_subject_frequency = True
# subject_add_topic_count       = True
# subject_scale                 = "standardize"

#Best subject config  (cv_mean_macro_f1=0.5999, holdout_macro_f1=0.6036):
subject_add_primary = True
subject_primary_strategy = "most_frequent"
subject_add_topic_count = True
subject_add_multiple_topics_flag = False
subject_add_length_features = False
subject_add_subject_frequency = False
subject_add_subject_is_rare = True
subject_add_grouped_primary = True
subject_scale = "standardize"

# --- Speaker (fixed recommended config) ---
# speaker_add_frequency       = True
# speaker_add_grouped_speaker = True
# speaker_group_rare          = True
# speaker_scale               = "standardize"

# Best speaker config  (cv_mean_macro_f1=0.5915):
speaker_add_frequency=True
speaker_add_is_rare=True
speaker_add_grouped_speaker=True
speaker_add_length_features=False
speaker_add_title_flag=True
speaker_add_comma_flag=True
speaker_add_period_flag=False
speaker_add_token_count=False
speaker_scale="standardize"
speaker_group_rare=True

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
# STEP 1 — PRECOMPUTE BASE FEATURES (runs once)
# Speaker_job options are at defaults here (only speaker_job_clean is produced).
# =============================================================================

print("\n[SECTION] Precomputing base features (statement + subject + speaker + party + FE)")
base_options = OneStepOptions(
    # Statement — full expensive config
    statement_vectorizer_type=statement_vectorizer_type,
    statement_vectorizer_max_features=statement_vectorizer_max_features,
    statement_vectorizer_min_df=statement_vectorizer_min_df,
    statement_vectorizer_max_df=statement_vectorizer_max_df,
    statement_stopword_removal=statement_stopword_removal,
    statement_lemmatizer=statement_lemmatizer,
    statement_keep_negations=statement_keep_negations,
    statement_add_lexical_features=statement_add_lexical_features,
    statement_scale=statement_scale,
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
    speaker_add_length_features=speaker_add_length_features,
    speaker_add_title_flag=speaker_add_title_flag,
    speaker_add_comma_flag=speaker_add_comma_flag,
    speaker_add_period_flag=speaker_add_period_flag,
    speaker_add_token_count=speaker_add_token_count,
    speaker_group_rare=speaker_group_rare,
    speaker_scale=speaker_scale,
    # Speaker_job — defaults only (no add_* flags), so no job features are added
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

df_base   = preprocess_one_step(df, options=base_options)
y_all     = df_base["label"]
X_base    = df_base.drop(columns=["label"]).select_dtypes(exclude="object")
base_cols = set(X_base.columns)

print(f"Base feature matrix: {X_base.shape[0]:,} rows × {X_base.shape[1]:,} features")


# =============================================================================
# SPEAKER_JOB OPTION GRID
# =============================================================================

SPEAKER_JOB_GRID = {
    "speaker_job_add_frequency":       [False, True],
    "speaker_job_add_is_rare":         [False, True],
    "speaker_job_add_grouped_job":     [False, True],
    "speaker_job_add_length_features": [False, True],
    "speaker_job_add_title_flag":      [False, True],
    "speaker_job_add_comma_flag":      [False, True],
    "speaker_job_add_slash_flag":      [False, True],
    "speaker_job_add_ampersand_flag":  [False, True],
    "speaker_job_add_token_count":     [False, True],
    "speaker_job_scale":               ["none", "standardize"],
}

# Set to an integer to cap the number of runs. None = run all.
MAX_RUNS = None


def generate_speaker_job_configs():
    keys, seen, result = list(SPEAKER_JOB_GRID.keys()), set(), []
    for values in product(*SPEAKER_JOB_GRID.values()):
        cfg = dict(zip(keys, values))
        # speaker_job_group_rare must be True whenever speaker_job_add_grouped_job=True.
        cfg["speaker_job_group_rare"] = cfg["speaker_job_add_grouped_job"]
        frozen = tuple(sorted(cfg.items()))
        if frozen not in seen:
            seen.add(frozen)
            result.append(cfg)
    return result


speaker_job_configs = generate_speaker_job_configs()
if MAX_RUNS is not None:
    speaker_job_configs = speaker_job_configs[:MAX_RUNS]

print(f"\nTotal speaker_job configurations to run: {len(speaker_job_configs)}")
print("(Adjust SPEAKER_JOB_GRID or set MAX_RUNS to limit)\n")


# =============================================================================
# MODEL HYPERPARAMETERS
# =============================================================================

CLASS_WEIGHT = {0: 1.42, 1: 0.77}
C_VALUE      = 1.0
MAX_ITER     = 300
CV_FOLDS     = 3


# =============================================================================
# STEP 2 — SWEEP LOOP
# Each iteration only reruns the speaker_job module (no TF-IDF, no lemmatization).
# =============================================================================

all_results = []

for run_idx, job_cfg in enumerate(speaker_job_configs, 1):
    print(f"\n{'='*60}")
    print(f"Run {run_idx} / {len(speaker_job_configs)}")
    print(f"Speaker_job config: {job_cfg}")
    print("="*60)

    # -------------------------------------------------------------------------
    # Fast speaker_job-only preprocessing pass
    # Statement module runs basic cleaning only (vectorizer + lemmatizer off).
    # All other modules run at defaults — no extra columns added.
    # -------------------------------------------------------------------------
    iter_options = OneStepOptions(
        # Statement: skip expensive parts (already in base features)
        statement_vectorizer_type="none",
        statement_lemmatizer="none",
        statement_stopword_removal=False,
        statement_add_lexical_features=False,
        # Subject / speaker / party / FE: defaults only (already in base features)
        # Speaker_job: current config under test
        **job_cfg,
    )
    df_iter  = preprocess_one_step(df, options=iter_options)
    X_iter   = df_iter.drop(columns=["label"], errors="ignore").select_dtypes(exclude="object")
    new_cols = [c for c in X_iter.columns if c not in base_cols]

    if new_cols:
        X = pd.concat([X_base, X_iter[new_cols]], axis=1)
    else:
        X = X_base.copy()

    y = y_all
    print(f"Features: {X_base.shape[1]} base + {len(new_cols)} speaker_job = {X.shape[1]} total")

    # -------------------------------------------------------------------------
    # Train / holdout split
    # -------------------------------------------------------------------------
    X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -------------------------------------------------------------------------
    # W&B run
    # -------------------------------------------------------------------------
    wandb_run = wandb.init(
        project="truth-classifier-lr-spjob-sweep",
        config={
            "model":      "LogisticRegression",
            "solver":     "liblinear",
            "C":          C_VALUE,
            "max_iter":   MAX_ITER,
            "cv_folds":   CV_FOLDS,
            "n_features": int(X_trainval.shape[1]),
            **{f"job__{k}": str(v) for k, v in job_cfg.items()},
        },
        reinit=True,
    )

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
        wandb.log({"cv/fold": fold, "cv/macro_f1": macro_f1})

    cv_mean = float(np.mean(fold_macro_f1s))
    cv_std  = float(np.std(fold_macro_f1s))
    print(f"CV macro_f1: {cv_mean:.4f} ± {cv_std:.4f}")

    wandb.log({"cv/mean_macro_f1": cv_mean, "cv/std_macro_f1": cv_std})
    wandb_run.summary["cv_mean_macro_f1"] = cv_mean
    wandb_run.finish()

    all_results.append({
        "run":              run_idx,
        "cv_mean_macro_f1": cv_mean,
        "cv_std_macro_f1":  cv_std,
        "n_job_cols":       len(new_cols),
        **job_cfg,
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
print(f"\nBest speaker_job config  (cv_mean_macro_f1={best_cfg['cv_mean_macro_f1']:.4f}):")
for k in SPEAKER_JOB_GRID:
    print(f"  {k}: {best_cfg[k]}")


# --- Reconstruct best feature matrix and evaluate on holdout ---
print("\n[SECTION] Evaluating best config on holdout set")

best_job_cfg = {k: best_cfg[k] for k in list(SPEAKER_JOB_GRID.keys()) + ["speaker_job_group_rare"]}

iter_options = OneStepOptions(
    statement_vectorizer_type="none",
    statement_lemmatizer="none",
    statement_stopword_removal=False,
    statement_add_lexical_features=False,
    **best_job_cfg,
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
