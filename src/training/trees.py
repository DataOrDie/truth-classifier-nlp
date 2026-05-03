from pathlib import Path
import sys
import importlib.util

import numpy as np
import optuna
import pandas as pd
import wandb
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# -----------------------------------------------------------------------------
# W&B authentication
# -----------------------------------------------------------------------------
print("[SECTION] W&B authentication")
wandb.login()


# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------
project_root = Path(__file__).resolve().parents[2]
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from preprocessing.one_step import OneStepOptions, preprocess_one_step
from submit.save_model import save_model


# -----------------------------------------------------------------------------
# Dataset load
# -----------------------------------------------------------------------------
print("[SECTION] Loading dataset")
df_candidate = globals().get("df", None)
if not isinstance(df_candidate, pd.DataFrame):
    df = pd.read_csv(project_root / "data" / "train.csv")
else:
    df = df_candidate


# -----------------------------------------------------------------------------
# Preprocessing options
# -----------------------------------------------------------------------------
print("[SECTION] Configuring preprocessing options")
noemp_option: str = "trees"
newexist_option: str = "A"
createjob_option: str = "drop"
retainedjob_option: str = "trees"
disbursementgross_option: str = "trees"
balancegross_option: str = "drop"

approvaldate_option: str = "C"
approvalfy_option: str = "C"
franchise_option: str = "binary"
urbanrural_option: str = "onehot"
revlinecr_option: str = "C"
lowdoc_option: str = "C"
accept_option: str = ""

local_state: str = "IL"
citybank_option: str = "freq_bucket"
city_top_k: int = 120
bank_top_k: int = 80
city_min_count: int | None = None
bank_min_count: int | None = None
citybank_other_label: str = "OTHER"
citybank_suffix: str = "_bucket"
citybank_drop_original: bool = False

options = OneStepOptions(
    noemp_option=noemp_option,
    newexist_option=newexist_option,
    createjob_option=createjob_option,
    retainedjob_option=retainedjob_option,
    approvaldate_option=approvaldate_option,
    approvalfy_option=approvalfy_option,
    franchise_option=franchise_option,
    urbanrural_option=urbanrural_option,
    revlinecr_option=revlinecr_option,
    lowdoc_option=lowdoc_option,
    disbursementgross_option=disbursementgross_option,
    balancegross_option=balancegross_option,
    local_state=local_state,
    citybank_option=citybank_option,
    city_top_k=city_top_k,
    bank_top_k=bank_top_k,
    city_min_count=city_min_count,
    bank_min_count=bank_min_count,
    citybank_other_label=citybank_other_label,
    citybank_suffix=citybank_suffix,
    citybank_drop_original=citybank_drop_original,
)


# -----------------------------------------------------------------------------
# Preprocess
# -----------------------------------------------------------------------------
print("[SECTION] Running preprocessing")
df_processed = preprocess_one_step(df, options=options, is_tree_model=True)
print(f"Rows: {len(df_processed):,}")
print(f"Features: {df_processed.shape[1]}")


# -----------------------------------------------------------------------------
# Split strategy
# -----------------------------------------------------------------------------
print("[SECTION] Building train/holdout split strategy")
target_col = "Accept"
X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

X_trainval, X_holdout, y_trainval, y_holdout = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"Train/Val set size: {X_trainval.shape[0]}")
print(f"Holdout set size: {X_holdout.shape[0]}")
print(f"Train/Val target distribution:\n{y_trainval.value_counts()}\n")
print(f"Holdout target distribution:\n{y_holdout.value_counts()}\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
n_splits = skf.get_n_splits()
print(f"StratifiedKFold splits: {n_splits}")


# -----------------------------------------------------------------------------
# RandomForest bagging strategy
# -----------------------------------------------------------------------------
print("[SECTION] Configuring RandomForest bagging strategy")
random_state: int = 42
# balance_strategy: str = "class_weight"
balance_strategy: str = "oversample_reject"
# Options: "none" | "class_weight" | "oversample_reject"
tuning_trials: int = 24
threshold_grid = np.linspace(0.12, 0.88, 153)

base_rf_params = {
    "n_estimators": 900,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": 0.35,
    "bootstrap": True,
    "max_samples": 0.85,
    "class_weight": "balanced_subsample",
}

def build_rf_pipeline(random_state: int = 42, model_params: dict | None = None) -> Pipeline:
    if model_params is None:
        model_params = base_rf_params

    if balance_strategy == "class_weight":
        class_weight = model_params.get("class_weight")
        if class_weight is None:
            class_weight = "balanced_subsample"
    else:
        class_weight = None

    max_samples = model_params.get("max_samples", None)

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(steps=[("imputer", SimpleImputer(strategy="median"))]),
                make_column_selector(dtype_include=np.number),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=model_params["n_estimators"],
                    criterion=model_params.get("criterion", "gini"),
                    max_depth=model_params["max_depth"],
                    min_samples_split=model_params["min_samples_split"],
                    min_samples_leaf=model_params["min_samples_leaf"],
                    max_features=model_params["max_features"],
                    bootstrap=model_params.get("bootstrap", True),
                    max_samples=max_samples,
                    class_weight=class_weight,
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )


def compute_binary_metrics(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }


def rebalance_training_data(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """Apply class balancing on the training split only."""

    if strategy != "oversample_reject":
        return X_train, y_train

    train_frame = X_train.copy()
    train_frame[target_col] = y_train.values

    reject_rows = train_frame[train_frame[target_col] == 0]
    approve_rows = train_frame[train_frame[target_col] == 1]

    if reject_rows.empty or approve_rows.empty:
        return X_train, y_train

    reject_rows = reject_rows.sample(
        n=len(approve_rows),
        replace=True,
        random_state=random_state,
    )

    balanced = pd.concat([approve_rows, reject_rows], ignore_index=True)
    balanced = balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    return balanced.drop(columns=[target_col]), balanced[target_col]


def evaluate_rf_config(model_params: dict) -> dict:
    fold_metrics = []
    oof_scores = np.zeros(len(X_trainval), dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
        X_fold_train = X_trainval.iloc[train_idx]
        y_fold_train = y_trainval.iloc[train_idx]
        X_fold_val = X_trainval.iloc[val_idx]
        y_fold_val = y_trainval.iloc[val_idx]

        X_fold_train, y_fold_train = rebalance_training_data(
            X_fold_train,
            y_fold_train,
            strategy=balance_strategy,
            random_state=random_state + fold_idx,
        )

        fold_pipeline = build_rf_pipeline(
            random_state=random_state + fold_idx,
            model_params=model_params,
        )
        fold_pipeline.fit(X_fold_train, y_fold_train)

        y_fold_score = fold_pipeline.predict_proba(X_fold_val)[:, 1]
        oof_scores[val_idx] = y_fold_score

        fold_metrics.append(
            {
                "fold": fold_idx,
                **compute_binary_metrics(y_fold_val, y_fold_score, threshold=0.5),
            }
        )

    best_threshold = 0.5
    best_oof_macro_f1 = float("-inf")
    for threshold in threshold_grid:
        threshold_pred = (oof_scores >= threshold).astype(int)
        threshold_macro_f1 = f1_score(y_trainval, threshold_pred, average="macro", zero_division=0)
        if threshold_macro_f1 > best_oof_macro_f1:
            best_oof_macro_f1 = float(threshold_macro_f1)
            best_threshold = float(threshold)

    oof_metrics = compute_binary_metrics(y_trainval, oof_scores, threshold=best_threshold)

    return {
        "params": model_params,
        "fold_metrics": fold_metrics,
        "best_threshold": best_threshold,
        "oof_macro_f1": best_oof_macro_f1,
        "oof_metrics": oof_metrics,
        "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in fold_metrics])),
        "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in fold_metrics])),
        "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in fold_metrics])),
        "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in fold_metrics])),
    }


# -----------------------------------------------------------------------------
# W&B run
# -----------------------------------------------------------------------------
print("[SECTION] Initializing W&B run")
model_name = "bagging-randforest"
create_kaggle_csv: bool = True

run = wandb.init(
    project="MS BAGGING - RANDOM FOREST",
    config={
        "model_name": model_name,
        "random_state": random_state,
        "tuning_trials": tuning_trials,
        "balance_strategy": balance_strategy,
        "balance_target_column": target_col,
        "accept_label": 1,
        "reject_label": 0,
        "oversample_target_label": 0 if balance_strategy == "oversample_reject" else None,
        "rebalance_applied_in_cv": balance_strategy == "oversample_reject",
        "rebalance_applied_in_final_fit": balance_strategy == "oversample_reject",
        "base_rf_params": base_rf_params,
        "tuning_algorithm": "optuna_tpe",
        "noemp_option": noemp_option,
        "newexist_option": newexist_option,
        "createjob_option": createjob_option,
        "retainedjob_option": retainedjob_option,
        "approvaldate_option": approvaldate_option,
        "approvalfy_option": approvalfy_option,
        "franchise_option": franchise_option,
        "urbanrural_option": urbanrural_option,
        "revlinecr_option": revlinecr_option,
        "lowdoc_option": lowdoc_option,
        "disbursementgross_option": disbursementgross_option,
        "balancegross_option": balancegross_option,
        "local_state": local_state,
        "citybank_option": citybank_option,
        "city_top_k": city_top_k,
        "bank_top_k": bank_top_k,
        "city_min_count": city_min_count,
        "bank_min_count": bank_min_count,
        "citybank_other_label": citybank_other_label,
        "citybank_suffix": citybank_suffix,
        "citybank_drop_original": citybank_drop_original,
        "cv_n_splits": n_splits,
        "n_trainval_rows": int(X_trainval.shape[0]),
        "n_holdout_rows": int(X_holdout.shape[0]),
        "n_features": int(X_trainval.shape[1]),
    },
)


# -----------------------------------------------------------------------------
# Optuna tuning on train/val split
# -----------------------------------------------------------------------------
print("[SECTION] Running Optuna tuning with OOF threshold optimization")

tuning_results = []


def sample_rf_params(trial: optuna.Trial) -> dict:
    if balance_strategy == "class_weight":
        class_weight = trial.suggest_categorical("class_weight", ["balanced", "balanced_subsample"])
    else:
        class_weight = None

    return {
        "n_estimators": trial.suggest_int("n_estimators", 700, 1500, step=100),
        "criterion": "gini",
        "max_depth": trial.suggest_categorical("max_depth", [None, 16, 24, 32]),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10, step=2),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 6),
        "max_features": trial.suggest_float("max_features", 0.20, 0.70),
        "bootstrap": True,
        "class_weight": class_weight,
        "max_samples": trial.suggest_float("max_samples", 0.70, 1.00),
    }


def tuning_objective(trial: optuna.Trial) -> float:
    candidate_params = sample_rf_params(trial)
    result = evaluate_rf_config(candidate_params)
    trial_idx = trial.number + 1

    tuning_results.append(result)

    wandb.log(
        {
            "tuning/trial": trial_idx,
            "tuning/oof_macro_f1": result["oof_macro_f1"],
            "tuning/cv_mean_macro_f1": result["cv_mean_macro_f1"],
            "tuning/cv_std_macro_f1": result["cv_std_macro_f1"],
            "tuning/cv_mean_roc_auc": result["cv_mean_roc_auc"],
            "tuning/cv_mean_pr_auc": result["cv_mean_pr_auc"],
            "tuning/cv_mean_balanced_accuracy": result["cv_mean_balanced_accuracy"],
            "tuning/best_threshold": result["best_threshold"],
        }
    )

    print(
        f"Trial {trial_idx:02d}/{tuning_trials} | "
        f"n_estimators={candidate_params['n_estimators']} "
        f"max_depth={candidate_params['max_depth']} "
        f"min_samples_leaf={candidate_params['min_samples_leaf']} "
        f"max_features={candidate_params['max_features']} | "
        f"OOF Macro-F1={result['oof_macro_f1']:.4f} "
        f"BestThreshold={result['best_threshold']:.3f}"
    )

    return float(result["oof_macro_f1"])


study = optuna.create_study(
    direction="maximize",
    sampler=optuna.samplers.TPESampler(seed=random_state),
)
study.enqueue_trial(
    {
        "n_estimators": base_rf_params["n_estimators"],
        "max_depth": base_rf_params["max_depth"],
        "min_samples_split": base_rf_params["min_samples_split"],
        "min_samples_leaf": base_rf_params["min_samples_leaf"],
        "max_features": base_rf_params["max_features"],
        "max_samples": base_rf_params["max_samples"],
        **(
            {"class_weight": base_rf_params["class_weight"]}
            if balance_strategy == "class_weight"
            else {}
        ),
    }
)
study.optimize(tuning_objective, n_trials=tuning_trials)

tuning_results_sorted = sorted(
    tuning_results,
    key=lambda item: item["oof_macro_f1"],
    reverse=True,
)
best_tuning_result = tuning_results_sorted[0]
best_rf_params = best_tuning_result["params"]
decision_threshold = best_tuning_result["best_threshold"]

print("[SECTION] Best RF tuning result selected")
print(
    f"Best OOF Macro-F1={best_tuning_result['oof_macro_f1']:.4f} | "
    f"Decision threshold={decision_threshold:.3f}"
)

tuning_table = wandb.Table(
    columns=[
        "n_estimators",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "max_features",
        "class_weight",
        "max_samples",
        "cv_mean_macro_f1",
        "cv_std_macro_f1",
        "cv_mean_roc_auc",
        "cv_mean_pr_auc",
        "cv_mean_balanced_accuracy",
        "oof_roc_auc",
        "oof_pr_auc",
        "oof_f1",
        "oof_macro_f1",
        "oof_balanced_accuracy",
        "best_threshold",
    ],
    data=[
        [
            int(item["params"]["n_estimators"]),
            str(item["params"]["max_depth"]),
            int(item["params"]["min_samples_split"]),
            int(item["params"]["min_samples_leaf"]),
            str(item["params"]["max_features"]),
            str(item["params"].get("class_weight")),
            float(item["params"].get("max_samples", 1.0)),
            float(item["cv_mean_macro_f1"]),
            float(item["cv_std_macro_f1"]),
            float(item["cv_mean_roc_auc"]),
            float(item["cv_mean_pr_auc"]),
            float(item["cv_mean_balanced_accuracy"]),
            float(item["oof_metrics"]["roc_auc"]),
            float(item["oof_metrics"]["pr_auc"]),
            float(item["oof_metrics"]["f1"]),
            float(item["oof_metrics"]["macro_f1"]),
            float(item["oof_metrics"]["balanced_accuracy"]),
            float(item["best_threshold"]),
        ]
        for item in tuning_results_sorted
    ],
)

wandb.log(
    {
        "tuning/results_table": tuning_table,
        "tuning/best_oof_macro_f1": float(best_tuning_result["oof_macro_f1"]),
        "tuning/best_threshold": float(decision_threshold),
        "selected/n_estimators": int(best_rf_params["n_estimators"]),
        "selected/max_depth": str(best_rf_params["max_depth"]),
        "selected/min_samples_split": int(best_rf_params["min_samples_split"]),
        "selected/min_samples_leaf": int(best_rf_params["min_samples_leaf"]),
        "selected/max_features": str(best_rf_params["max_features"]),
        "selected/class_weight": str(best_rf_params.get("class_weight")),
        "selected/max_samples": float(best_rf_params.get("max_samples", 1.0)),
    }
)

# Re-log fold metrics in the existing cv/* format with the selected config.
cv_fold_metrics = best_tuning_result["fold_metrics"]
for fold_metrics in cv_fold_metrics:
    fold_idx = fold_metrics["fold"]
    wandb.log(
        {
            "cv/fold": fold_idx,
            "cv/roc_auc": fold_metrics["roc_auc"],
            "cv/pr_auc": fold_metrics["pr_auc"],
            "cv/f1": fold_metrics["f1"],
            "cv/macro_f1": fold_metrics["macro_f1"],
            "cv/balanced_accuracy": fold_metrics["balanced_accuracy"],
            "cv/precision": fold_metrics["precision"],
            "cv/recall": fold_metrics["recall"],
            "cv/accuracy": fold_metrics["accuracy"],
        }
    )

cv_summary = {
    "cv_mean_roc_auc": float(np.mean([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_std_roc_auc": float(np.std([m["roc_auc"] for m in cv_fold_metrics])),
    "cv_mean_pr_auc": float(np.mean([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_std_pr_auc": float(np.std([m["pr_auc"] for m in cv_fold_metrics])),
    "cv_mean_f1": float(np.mean([m["f1"] for m in cv_fold_metrics])),
    "cv_std_f1": float(np.std([m["f1"] for m in cv_fold_metrics])),
    "cv_mean_macro_f1": float(np.mean([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_std_macro_f1": float(np.std([m["macro_f1"] for m in cv_fold_metrics])),
    "cv_mean_balanced_accuracy": float(np.mean([m["balanced_accuracy"] for m in cv_fold_metrics])),
    "cv_std_balanced_accuracy": float(np.std([m["balanced_accuracy"] for m in cv_fold_metrics])),
    "cv_mean_precision": float(np.mean([m["precision"] for m in cv_fold_metrics])),
    "cv_std_precision": float(np.std([m["precision"] for m in cv_fold_metrics])),
    "cv_mean_recall": float(np.mean([m["recall"] for m in cv_fold_metrics])),
    "cv_std_recall": float(np.std([m["recall"] for m in cv_fold_metrics])),
    "cv_mean_accuracy": float(np.mean([m["accuracy"] for m in cv_fold_metrics])),
    "cv_std_accuracy": float(np.std([m["accuracy"] for m in cv_fold_metrics])),
    "oof_roc_auc": float(best_tuning_result["oof_metrics"]["roc_auc"]),
    "oof_pr_auc": float(best_tuning_result["oof_metrics"]["pr_auc"]),
    "oof_f1": float(best_tuning_result["oof_metrics"]["f1"]),
    "oof_macro_f1": float(best_tuning_result["oof_metrics"]["macro_f1"]),
    "oof_balanced_accuracy": float(best_tuning_result["oof_metrics"]["balanced_accuracy"]),
    "decision_threshold": float(decision_threshold),
}

cv_table = wandb.Table(
    columns=[
        "fold",
        "roc_auc",
        "pr_auc",
        "f1",
        "macro_f1",
        "balanced_accuracy",
        "precision",
        "recall",
        "accuracy",
    ],
    data=[
        [
            int(m["fold"]),
            float(m["roc_auc"]),
            float(m["pr_auc"]),
            float(m["f1"]),
            float(m["macro_f1"]),
            float(m["balanced_accuracy"]),
            float(m["precision"]),
            float(m["recall"]),
            float(m["accuracy"]),
        ]
        for m in cv_fold_metrics
    ],
)

wandb.log({"cv/folds_table": cv_table, **cv_summary})

print("[SECTION] Cross-validation summary")
print(f"CV ROC-AUC: {cv_summary['cv_mean_roc_auc']:.4f} +/- {cv_summary['cv_std_roc_auc']:.4f}")
print(f"CV PR-AUC: {cv_summary['cv_mean_pr_auc']:.4f} +/- {cv_summary['cv_std_pr_auc']:.4f}")
print(f"CV F1: {cv_summary['cv_mean_f1']:.4f} +/- {cv_summary['cv_std_f1']:.4f}")
print(f"CV Macro-F1: {cv_summary['cv_mean_macro_f1']:.4f} +/- {cv_summary['cv_std_macro_f1']:.4f}")
print(f"OOF Macro-F1 (best trial): {cv_summary['oof_macro_f1']:.4f}")
print(f"Decision threshold from OOF: {decision_threshold:.3f}")


# -----------------------------------------------------------------------------
# Train final model on train/val split
# -----------------------------------------------------------------------------
print("[SECTION] Training RandomForest pipeline on full train/val split")
rf_pipeline = build_rf_pipeline(random_state=random_state, model_params=best_rf_params)
X_trainval_fit, y_trainval_fit = rebalance_training_data(
    X_trainval,
    y_trainval,
    strategy=balance_strategy,
    random_state=random_state,
)
rf_pipeline.fit(X_trainval_fit, y_trainval_fit)


# -----------------------------------------------------------------------------
# Running holdout predictions and metric evaluation
# -----------------------------------------------------------------------------
print("[SECTION] Running holdout predictions and metric evaluation")
y_score = rf_pipeline.predict_proba(X_holdout)[:, 1]
y_pred = (y_score >= decision_threshold).astype(int)

metrics = compute_binary_metrics(y_holdout, y_score, decision_threshold)

cm = confusion_matrix(y_holdout, y_pred)
report = classification_report(y_holdout, y_pred, output_dict=True, zero_division=0)
positive_rate = float((y_pred == 1).mean())

print(f"Holdout ROC-AUC: {metrics['roc_auc']:.4f}")
print(f"Holdout PR-AUC: {metrics['pr_auc']:.4f}")
print(f"Holdout F1: {metrics['f1']:.4f}")
print(f"Holdout Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Holdout Balanced-Accuracy: {metrics['balanced_accuracy']:.4f}")
print(f"Holdout Precision: {metrics['precision']:.4f}")
print(f"Holdout Recall: {metrics['recall']:.4f}")
print(f"Holdout Accuracy: {metrics['accuracy']:.4f}")
print(f"Predicted positive rate: {positive_rate:.4f}")


# -----------------------------------------------------------------------------
# W&B logging
# -----------------------------------------------------------------------------
print("[SECTION] Logging to W&B")
wandb.log(
    {
        "holdout/roc_auc": metrics["roc_auc"],
        "holdout/pr_auc": metrics["pr_auc"],
        "holdout/f1": metrics["f1"],
        "holdout/macro_f1": metrics["macro_f1"],
        "holdout/balanced_accuracy": metrics["balanced_accuracy"],
        "holdout/precision": metrics["precision"],
        "holdout/recall": metrics["recall"],
        "holdout/accuracy": metrics["accuracy"],
        "predicted_positive_rate": positive_rate,
        "decision_threshold": decision_threshold,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
)

run.summary["macro_f1"] = metrics["macro_f1"]
run.summary["holdout/macro_f1"] = metrics["macro_f1"]
run.summary["decision_threshold"] = decision_threshold

report_rows = []
for label, values in report.items():
    if isinstance(values, dict):
        report_rows.append(
            [
                label,
                float(values.get("precision", 0.0)),
                float(values.get("recall", 0.0)),
                float(values.get("f1-score", 0.0)),
                float(values.get("support", 0.0)),
            ]
        )

report_table = wandb.Table(
    columns=["label", "precision", "recall", "f1_score", "support"],
    data=report_rows,
)

wandb.log(
    {
        "classification_report": report_table,
        "confusion_matrix": wandb.plot.confusion_matrix(
            y_true=y_holdout.tolist(),
            preds=y_pred.tolist(),
            class_names=["Reject", "Accept"],
        ),
    }
)

print("[SECTION] Finishing W&B run")
run.finish()


# Save model for reuse/submission
saved_paths = save_model(
    model_pipeline=rf_pipeline,
    preprocessing_options=options,
    feature_names=X_trainval.columns.tolist(),
    project_root=project_root,
    model_name=model_name,
)

if create_kaggle_csv:
    print("[SECTION] Creating Kaggle submission CSV from saved model")
    kaggle_module_path = project_root / "src" / "submit" / "kaggle-modulo.py"
    spec = importlib.util.spec_from_file_location("kaggle_modulo", kaggle_module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load Kaggle module from: {kaggle_module_path}")

    kaggle_modulo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kaggle_modulo)

    submission_path = kaggle_modulo.generate_submission_csv(
        model_name=model_name,
        project_root=project_root,
        is_tree_model=True,
        verbose=False,
    )
    print(f"Kaggle submission generated: {submission_path}")