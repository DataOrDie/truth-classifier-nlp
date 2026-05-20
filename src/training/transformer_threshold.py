"""
Transformer-only threshold sweep.
Drops all GBDT base models — their ROC-AUC (0.65-0.67) is far below the transformer
(~0.73) and contributes only downward pull when stacked via meta-LR.
Loads the saved transformer k-fold OOF probas and sweeps threshold directly.
"""
from datetime import datetime
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
from sklearn.model_selection import train_test_split


# ============================================================
# Project root
# ============================================================

def find_project_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / 'data' / 'train.csv').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError('Could not locate the project root.')

project_root = find_project_root(Path.cwd())
print(f'Project root: {project_root}')

_script_start = time()
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ============================================================
# Labels + train/holdout split  (same seed as all other scripts)
# ============================================================
df = pd.read_csv(project_root / 'data' / 'train.csv')

df_trainval, df_holdout = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
y_trainval   = df_trainval["label"]
y_holdout    = df_holdout["label"]
idx_trainval = df_trainval.index
idx_holdout  = df_holdout.index

print(f"Train/val: {len(idx_trainval):,}   Holdout: {len(idx_holdout):,}")


# ============================================================
# Locate transformer artifacts
# ============================================================

def _find_transformer_artifacts(root: Path):
    """Return (dir, oof_filename, variant_label) for best available transformer artifacts."""
    candidates = [
        (root / "models" / "transformer_lora_kfold",  "mistral-7b-lora3-kfold-oof.csv", "mistral-7b-lora3"),
        (root / "models" / "transformer_lora_kfold",  "mistral-7b-lora-kfold-oof.csv",  "mistral-7b-lora"),
        (root / "models" / "transformer_kfold_base",  "deberta-v3-base-kfold-oof.csv",   "deberta-v3-base"),
        (root / "models" / "transformer_kfold",       "deberta-v3-small-kfold-oof.csv",  "deberta-v3-small"),
    ]
    for d, oof_name, label in candidates:
        if (d / oof_name).exists() and (d / "ho_proba.npy").exists() and (d / "test_proba.npy").exists():
            return d, oof_name, label
    return None, None, None

_KFOLD_DIR, _TRANS_OOF_NAME, _TRANS_VARIANT = _find_transformer_artifacts(project_root)
if _KFOLD_DIR is None:
    raise FileNotFoundError(
        "No transformer artifacts found. "
        "Run transformer_lora_kfold_extract.py or transformer_kfold_base_extract.py first."
    )

TRANSFORMER_OOF_PATH  = _KFOLD_DIR / _TRANS_OOF_NAME
TRANSFORMER_HO_PATH   = _KFOLD_DIR / "ho_proba.npy"
TRANSFORMER_TEST_PATH = _KFOLD_DIR / "test_proba.npy"
HO_IDX_PATH           = _KFOLD_DIR / "ho_idx.npy"

print(f"Transformer variant : {_TRANS_VARIANT}")
print(f"Artifacts directory : {_KFOLD_DIR}")


# ============================================================
# Load OOF probas — align to trainval row order
# ============================================================
print(f"\n[SECTION] Loading transformer OOF  [{_now()}]")
_trans_oof_df   = pd.read_csv(TRANSFORMER_OOF_PATH)
_idx_to_proba   = dict(zip(_trans_oof_df["idx"].astype(int), _trans_oof_df["oof_proba"]))
oof_transformer = np.array([_idx_to_proba[i] for i in idx_trainval], dtype=np.float64)

oof_roc_auc = roc_auc_score(y_trainval, oof_transformer)
print(f"  OOF rows : {len(_trans_oof_df):,}  "
      f"range=[{oof_transformer.min():.3f}, {oof_transformer.max():.3f}]")
print(f"  OOF ROC-AUC : {oof_roc_auc:.4f}")


# ============================================================
# Threshold sweep on OOF
# ============================================================
THRESHOLD_METRIC  = "macro_f1"
model_name        = "transformer_threshold"
create_kaggle_csv = True

print(f"\n[SECTION] Threshold sweep on OOF  [{_now()}]")

_metric_fns = {
    "macro_f1":     lambda t, yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
    "mcc":          lambda t, yt, yp: matthews_corrcoef(yt, yp),
    "balanced_acc": lambda t, yt, yp: balanced_accuracy_score(yt, yp),
}
_metric_fn = _metric_fns[THRESHOLD_METRIC]

threshold_grid   = np.arange(0.20, 0.76, 0.01)
threshold_scores = {}
for t in threshold_grid:
    preds = (oof_transformer >= t).astype(int)
    threshold_scores[round(float(t), 2)] = _metric_fn(t, y_trainval, preds)

best_threshold = max(threshold_scores, key=threshold_scores.get)
best_score     = threshold_scores[best_threshold]

print(f"  {'threshold':>10}   {THRESHOLD_METRIC}")
for t, s in threshold_scores.items():
    marker = "  <--" if t == best_threshold else ""
    print(f"  {t:>10.2f}   {s:.4f}{marker}")
print(f"\n  Best threshold: {best_threshold:.2f}  (OOF {THRESHOLD_METRIC}={best_score:.4f})")

THRESHOLD = best_threshold


# ============================================================
# W&B init
# ============================================================
print("[SECTION] Initializing W&B run")
wandb.login()
run = wandb.init(
    project="truth-classifier-stacking",
    config={
        "model":               "transformer-threshold",
        "transformer_variant": _TRANS_VARIANT,
        "threshold_metric":    THRESHOLD_METRIC,
        "best_threshold":      best_threshold,
        "oof_roc_auc":         round(oof_roc_auc, 4),
    },
)


# ============================================================
# Holdout evaluation
# ============================================================
print(f"\n[SECTION] Evaluating on holdout  [{_now()}]")
print(f"  Using threshold: {THRESHOLD:.2f}")

_ho_idx_saved   = np.load(HO_IDX_PATH)
_ho_proba_saved = np.load(TRANSFORMER_HO_PATH)
_ho_lookup      = dict(zip(_ho_idx_saved.tolist(), _ho_proba_saved.tolist()))
y_proba = np.array([_ho_lookup[i] for i in idx_holdout])
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


# ============================================================
# Plots
# ============================================================
print("[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(y_holdout, y_proba)
prec_c, rec_c, _ = precision_recall_curve(y_holdout, y_proba)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr, tpr, label=f"{_TRANS_VARIANT}  ROC-AUC={holdout_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set_title(f"ROC Curve — {_TRANS_VARIANT} (holdout)")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend()

axes[1].plot(rec_c, prec_c, label=f"PR-AUC={holdout_metrics['pr_auc']:.4f}")
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


# ============================================================
# W&B logging
# ============================================================
print("[SECTION] Logging to W&B")
wandb.log({
    "threshold/best":                    best_threshold,
    f"threshold/oof_{THRESHOLD_METRIC}": best_score,
    "threshold/grid": wandb.Table(
        columns=["threshold", THRESHOLD_METRIC],
        data=[[t, s] for t, s in threshold_scores.items()],
    ),
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


# ============================================================
# Save artifacts
# ============================================================
print("[SECTION] Saving artifacts")
_model_dir = project_root / "models" / model_name
_model_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(THRESHOLD,      _model_dir / "threshold.joblib")
joblib.dump(_TRANS_VARIANT, _model_dir / "transformer-variant.joblib")
print(f"  Artifacts saved to: {_model_dir}")


# ============================================================
# Kaggle submission CSV
# ============================================================
if create_kaggle_csv:
    print(f"\n[SECTION] Creating Kaggle submission CSV  [{_now()}]")
    df_test    = pd.read_csv(project_root / "data" / "test_nolabel.csv")
    test_proba = np.load(TRANSFORMER_TEST_PATH)
    test_pred  = (test_proba >= THRESHOLD).astype(int)

    submissions_dir = project_root / "submissions"
    submissions_dir.mkdir(exist_ok=True)
    submission_path = (
        submissions_dir
        / f"submission-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
    )

    pd.DataFrame({"id": df_test["id"], "label": test_pred}).to_csv(submission_path, index=False)
    print(f"  Saved: {submission_path}  ({len(df_test):,} rows)")

print(f"\n[DONE] Total script time: {time()-_script_start:.1f}s  [{_now()}]")
