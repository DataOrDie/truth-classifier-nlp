"""Logistic Regression baseline for the truth-classifier Kaggle competition.

Recommended preprocessing config for [linear] models from PREPROCESSING_OPTIONS.md.
Validation metric: macro F1.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split


# -----------------------------------------------------------------------------
# Path setup — works from any working directory
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
from submit.save_model import save_model


# -----------------------------------------------------------------------------
# W&B login
# -----------------------------------------------------------------------------

print("[SECTION] W&B authentication")
wandb.login()


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

print("[SECTION] Loading data")
df = pd.read_csv(project_root / "data" / "train.csv")
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")


# -----------------------------------------------------------------------------
# Preprocessing — recommended config for LinearSVC / LogisticRegression
# (source: PREPROCESSING_OPTIONS.md — LinearSVC / LogisticRegression section)
# -----------------------------------------------------------------------------

print("[SECTION] Configuring preprocessing options")
options = OneStepOptions(
    # --- Text (TF-IDF with stopword removal + lemmatization) ---
    statement_vectorizer_type="tfidf",
    statement_vectorizer_max_features=10000,
    statement_vectorizer_min_df=2,
    statement_vectorizer_max_df=0.9,
    statement_stopword_removal=True,
    statement_lemmatizer="wordnet",
    statement_keep_negations=True,      # keep "not", "never", etc. even when removing stopwords
    statement_add_lexical_features=True, # char_len, word_count, upper_ratio, punctuation counts
    statement_scale="standardize",

    # --- Subject ---
    subject_add_primary=True,           # subject_primary string (dropped later; frequency is kept)
    subject_add_subject_frequency=True, # how often this topic appears in the dataset
    subject_add_topic_count=True,       # number of comma-separated topics per row
    subject_scale="standardize",

    # --- Speaker ---
    speaker_add_frequency=True,         # how often this speaker appears
    speaker_add_grouped_speaker=True,   # rare speakers collapsed to "other"
    speaker_group_rare=True,
    speaker_scale="standardize",

    # --- Party ---
    party_affiliation_add_is_major_party=True,  # 1 if democrat or republican
    party_affiliation_add_frequency=True,

    # --- Feature engineering (text-style signals, no leakage risk) ---
    fe_add_negation_count=True,         # count of negation words (not, never, ...)
    fe_add_hedge_count=True,            # count of uncertainty words (maybe, possibly, ...)
    fe_add_absolutist_count=True,       # count of extreme words (always, everyone, hoax, ...)
    fe_add_readability=True,            # Flesch Reading Ease score
    fe_scale="standardize",
)

print("[SECTION] Running preprocessing")
df_proc = preprocess_one_step(df, options=options)
print(f"Shape after preprocessing: {df_proc.shape}")


# -----------------------------------------------------------------------------
# Build feature matrix X and target y
# -----------------------------------------------------------------------------

print("[SECTION] Building feature matrix")
target_col = "label"
y = df_proc[target_col]
X = df_proc.drop(columns=[target_col])

# Drop string/object columns — LogisticRegression only accepts numeric input.
# The cleaned text columns (*_clean) and categorical strings (subject_primary,
# speaker_grouped) are dropped here; their information is captured in the
# numeric features we enabled above (frequencies, TF-IDF vectors, etc.).
string_cols = X.select_dtypes(include="object").columns.tolist()
if string_cols:
    print(f"Dropping {len(string_cols)} string column(s): {string_cols}")
    X = X.drop(columns=string_cols)

print(f"Final feature matrix: {X.shape[0]:,} rows × {X.shape[1]:,} features")
print(f"Label distribution:  0 (true)={int((y == 0).sum())}  1 (false)={int((y == 1).sum())}")


# -----------------------------------------------------------------------------
# Train / holdout split
# -----------------------------------------------------------------------------

print("[SECTION] Splitting into train/val and holdout sets")
X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train/val: {X_trainval.shape[0]:,}  |  Holdout: {X_holdout.shape[0]:,}")


# -----------------------------------------------------------------------------
# Model config
# Class weights from CLAUDE.md to handle the 35%/65% imbalance:
#   true (0)  → upweight: 1.42
#   false (1) → downweight: 0.77
# -----------------------------------------------------------------------------

CLASS_WEIGHT = {0: 1.42, 1: 0.77}
C_VALUE      = 1.0      # inverse regularization strength — higher = less regularization
MAX_ITER     = 1000     # increase if you see convergence warnings


# -----------------------------------------------------------------------------
# W&B run init
# -----------------------------------------------------------------------------

print("[SECTION] Initializing W&B run")
run = wandb.init(
    project="truth-classifier-lr",
    config={
        "model":             "LogisticRegression",
        "C":                 C_VALUE,
        "solver":            "lbfgs",
        "max_iter":          MAX_ITER,
        "class_weight":      str(CLASS_WEIGHT),
        "vectorizer_type":   "tfidf",
        "max_features":      10000,
        "min_df":            2,
        "max_df":            0.9,
        "stopword_removal":  True,
        "lemmatizer":        "wordnet",
        "cv_folds":          5,
        "n_trainval":        int(X_trainval.shape[0]),
        "n_holdout":         int(X_holdout.shape[0]),
        "n_features":        int(X_trainval.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# 5-fold stratified cross-validation
# Primary metric: macro F1
# -----------------------------------------------------------------------------

print("[SECTION] Running 5-fold cross-validation (primary metric: macro F1)")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
    X_train_fold = X_trainval.iloc[train_idx]
    y_train_fold = y_trainval.iloc[train_idx]
    X_val_fold   = X_trainval.iloc[val_idx]
    y_val_fold   = y_trainval.iloc[val_idx]

    fold_model = LogisticRegression(
        C=C_VALUE,
        solver="lbfgs",
        max_iter=MAX_ITER,
        class_weight=CLASS_WEIGHT,
        random_state=42,
        n_jobs=-1,
    )
    fold_model.fit(X_train_fold, y_train_fold)

    y_pred_fold = fold_model.predict(X_val_fold)
    y_prob_fold = fold_model.predict_proba(X_val_fold)[:, 1]

    metrics = {
        "fold":     fold,
        "macro_f1": f1_score(y_val_fold, y_pred_fold, average="macro", zero_division=0),
        "roc_auc":  roc_auc_score(y_val_fold, y_prob_fold),
        "accuracy": accuracy_score(y_val_fold, y_pred_fold),
    }
    fold_results.append(metrics)

    print(
        f"  Fold {fold} | "
        f"macro_f1={metrics['macro_f1']:.4f}  "
        f"roc_auc={metrics['roc_auc']:.4f}  "
        f"acc={metrics['accuracy']:.4f}"
    )

    wandb.log({
        "cv/fold":     fold,
        "cv/macro_f1": metrics["macro_f1"],
        "cv/roc_auc":  metrics["roc_auc"],
        "cv/accuracy": metrics["accuracy"],
    })

mean_macro_f1 = float(np.mean([m["macro_f1"] for m in fold_results]))
std_macro_f1  = float(np.std([m["macro_f1"]  for m in fold_results]))
mean_roc_auc  = float(np.mean([m["roc_auc"]  for m in fold_results]))
std_roc_auc   = float(np.std([m["roc_auc"]   for m in fold_results]))

print(f"\nCV macro_f1 : {mean_macro_f1:.4f} ± {std_macro_f1:.4f}")
print(f"CV roc_auc  : {mean_roc_auc:.4f} ± {std_roc_auc:.4f}")

wandb.log({
    "cv/mean_macro_f1": mean_macro_f1,
    "cv/std_macro_f1":  std_macro_f1,
    "cv/mean_roc_auc":  mean_roc_auc,
    "cv/std_roc_auc":   std_roc_auc,
})


# -----------------------------------------------------------------------------
# Refit on all train/val data, then evaluate once on the holdout set
# -----------------------------------------------------------------------------

print("[SECTION] Refitting on full train/val set")
final_model = LogisticRegression(
    C=C_VALUE,
    solver="lbfgs",
    max_iter=MAX_ITER,
    class_weight=CLASS_WEIGHT,
    random_state=42,
    n_jobs=-1,
)
final_model.fit(X_trainval, y_trainval)

y_pred_holdout = final_model.predict(X_holdout)
y_prob_holdout = final_model.predict_proba(X_holdout)[:, 1]

holdout = {
    "macro_f1": f1_score(y_holdout, y_pred_holdout, average="macro", zero_division=0),
    "roc_auc":  roc_auc_score(y_holdout, y_prob_holdout),
    "pr_auc":   average_precision_score(y_holdout, y_prob_holdout),
    "accuracy": accuracy_score(y_holdout, y_pred_holdout),
}

cm = confusion_matrix(y_holdout, y_pred_holdout)

print("\n[SECTION] Holdout results")
for name, val in holdout.items():
    print(f"  {name}: {val:.4f}")
print(f"\nConfusion matrix (rows=actual, cols=predicted):\n{cm}")
print(f"\n{classification_report(y_holdout, y_pred_holdout, target_names=['true(0)', 'false(1)'])}")


# -----------------------------------------------------------------------------
# Log holdout metrics and confusion matrix to W&B
# -----------------------------------------------------------------------------

print("[SECTION] Logging to W&B")
wandb.log({
    "holdout/macro_f1": holdout["macro_f1"],
    "holdout/roc_auc":  holdout["roc_auc"],
    "holdout/pr_auc":   holdout["pr_auc"],
    "holdout/accuracy": holdout["accuracy"],
    "holdout/tn": int(cm[0, 0]),
    "holdout/fp": int(cm[0, 1]),
    "holdout/fn": int(cm[1, 0]),
    "holdout/tp": int(cm[1, 1]),
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=y_holdout.tolist(),
        preds=y_pred_holdout.tolist(),
        class_names=["true(0)", "false(1)"],
    ),
})

run.summary["macro_f1"]          = holdout["macro_f1"]
run.summary["cv_mean_macro_f1"]  = mean_macro_f1
run.finish()


# -----------------------------------------------------------------------------
# Save model artifacts (model + options + feature names → models/lr-v1/)
# -----------------------------------------------------------------------------

print("[SECTION] Saving model artifacts")
saved = save_model(
    model_pipeline=final_model,
    preprocessing_options=options,
    feature_names=X_trainval.columns.tolist(),
    project_root=project_root,
    model_name="lr-v1",
)
print(f"Model saved to: {saved['model_dir']}")
print("Done.")
