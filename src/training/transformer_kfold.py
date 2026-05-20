"""
Transformer fine-tuning — Experiment 2: 5-fold stratified k-fold ensemble.

Strategy
--------
Keep the same 20% holdout as all prior runs (seed=42, stratified) for apples-to-apples
comparison. Apply StratifiedKFold(5) to the remaining 80% (trainval, ~7,160 rows).

Each fold:
  - Trains on 4/5 of trainval (~5,728 rows)
  - Validates on 1/5 of trainval (~1,432 rows) → checkpoint selection only
  - Collects OOF probas on its val rows
  - Collects predictions on holdout and test (accumulated across folds)

After all folds:
  - OOF macro_f1 on full trainval — unbiased train-set estimate
  - Threshold tuned on OOF proba vs. trainval labels
  - Ensemble holdout probas (mean of 5 fold outputs) → held-out macro_f1
  - Ensemble test probas → Kaggle submission CSV

Config: Exp 1b confirmed sweet spot — FREEZE_EPOCHS=0, EPOCHS=3, text-format input.
"""
from datetime import datetime
from pathlib import Path
import sys
from time import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Environment detection
# ============================================================
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    DATA_DIR   = Path("/kaggle/input/truth-classifier-nlp")
    OUTPUT_DIR = Path("/kaggle/working/models/transformer_kfold")
else:
    def _find_root(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "data" / "train.csv").exists():
                return p
        raise FileNotFoundError("Cannot find project root")
    _root      = _find_root(Path.cwd())
    DATA_DIR   = _root / "data"
    OUTPUT_DIR = _root / "models" / "transformer_kfold"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config  (Exp 1b confirmed sweet spot)
# ============================================================
MODEL_NAME    = "microsoft/deberta-v3-small"
MAX_LENGTH    = 128
BATCH_SIZE    = 16
EPOCHS        = 3
FREEZE_EPOCHS = 0      # no freeze — text-format input; confirmed optimal
CLS_DROPOUT   = 0.3
LR            = 2e-5
LLRD_FACTOR   = 0.9
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
K_FOLDS       = 5

CLASS_WEIGHTS = [1.42, 0.77]
THRESHOLD     = 0.5
SEED          = 42

create_kaggle_csv = True
model_slug        = "deberta-v3-small-kfold"

NUM_WORKERS = 0 if sys.platform == "win32" else 2

torch.manual_seed(SEED)
np.random.seed(SEED)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False

print(f"Device : {device}")
if device.type == "cuda":
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print(f"  AMP  : {USE_AMP}")


# ============================================================
# Timing helper
# ============================================================
_script_start = time()
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ============================================================
# Input formatter  (identical to transformer_textformat.py)
# ============================================================
def format_input(speaker: str, party: str, subject: str, statement: str) -> str:
    def _clean(val) -> str:
        if pd.isna(val) or str(val).strip() == "":
            return "unknown"
        return str(val).strip().lower()
    primary_subject = _clean(subject).split(",")[0].strip()
    return (
        f"speaker: {_clean(speaker)} | "
        f"party: {_clean(party)} | "
        f"subject: {primary_subject} | "
        f"{statement}"
    )


# ============================================================
# Data loading
# ============================================================
print(f"\n[SECTION] Loading data  [{_now()}]")
df = pd.read_csv(DATA_DIR / "train.csv")
print(f"  Rows: {len(df):,}  |  Labels: {df['label'].value_counts().to_dict()}")

all_texts  = df.apply(
    lambda r: format_input(r["speaker"], r["party_affiliation"],
                           r["subject"], r["statement"]),
    axis=1,
).tolist()
all_labels = np.array(df["label"].tolist())
all_idx    = np.arange(len(df))


# ============================================================
# Outer split — same 80/20 holdout as all prior runs
# ============================================================
tv_idx, ho_idx = train_test_split(
    all_idx, test_size=0.2, random_state=SEED, stratify=all_labels
)
tv_labels = all_labels[tv_idx]
ho_labels = all_labels[ho_idx]
X_ho      = [all_texts[i] for i in ho_idx]

print(f"  Trainval: {len(tv_idx):,}   Holdout: {len(ho_idx):,}")


# ============================================================
# Tokenizer
# ============================================================
print(f"\n[SECTION] Loading tokenizer  [{_now()}]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class StatementDataset(Dataset):
    def __init__(self, texts: list[str], labels):
        self.enc    = tokenizer(texts, truncation=True, padding="max_length",
                                max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


# Build holdout dataset once — reused across all folds
print(f"  Tokenizing holdout ({len(X_ho):,} rows)...")
holdout_ds     = StatementDataset(X_ho, ho_labels)
holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS)

# Load test data for submission
if create_kaggle_csv:
    df_test    = pd.read_csv(DATA_DIR / "test_nolabel.csv")
    test_texts = df_test.apply(
        lambda r: format_input(r["speaker"], r["party_affiliation"],
                               r["subject"], r["statement"]),
        axis=1,
    ).tolist()
    print(f"  Test rows: {len(test_texts):,}")


# ============================================================
# LLRD + freeze helpers  (identical to transformer_textformat.py)
# ============================================================
def _build_llrd_param_groups(model, base_lr: float, llrd_factor: float,
                              weight_decay: float) -> list:
    no_decay   = {"bias", "LayerNorm.weight", "layer_norm.weight"}
    num_layers = len(model.deberta.encoder.layer)
    param_dict = dict(model.named_parameters())
    assigned   = set()
    groups     = []

    def _add(names: list[str], lr: float) -> None:
        wd = [param_dict[n] for n in names if not any(nd in n for nd in no_decay)]
        nd = [param_dict[n] for n in names if     any(nd in n for nd in no_decay)]
        if wd: groups.append({"params": wd, "lr": lr, "weight_decay": weight_decay})
        if nd: groups.append({"params": nd, "lr": lr, "weight_decay": 0.0})
        assigned.update(names)

    head = [n for n in param_dict
            if "deberta.encoder.layer." not in n and "deberta.embeddings." not in n]
    _add(head, base_lr)

    for layer_idx in range(num_layers - 1, -1, -1):
        depth = num_layers - layer_idx
        lr    = base_lr * (llrd_factor ** depth)
        _add([n for n in param_dict if f"deberta.encoder.layer.{layer_idx}." in n], lr)

    embed_lr = base_lr * (llrd_factor ** (num_layers + 1))
    _add([n for n in param_dict if "deberta.embeddings." in n and n not in assigned],
         embed_lr)
    return groups


def _freeze_backbone(model) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name and "pooler" not in name:
            param.requires_grad_(False)
    frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Backbone frozen — frozen={frozen:,}  trainable={trainable:,}")


def _unfreeze_backbone(model) -> None:
    for param in model.parameters():
        param.requires_grad_(True)
    print(f"  Backbone unfrozen — trainable={sum(p.numel() for p in model.parameters()):,}")


# ============================================================
# Training helpers
# ============================================================
def train_epoch(model, loader, optimizer, scheduler, criterion) -> float:
    model.train()
    total_loss = 0.0
    for batch_idx, (inputs, labs) in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labs   = labs.to(device)
        optimizer.zero_grad()
        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(**inputs).logits
            loss   = criterion(logits, labs)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        if batch_idx == 0:
            print(f"    Batch 0 — loss={loss.item():.4f}")
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}  avg_loss={total_loss/(batch_idx+1):.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def predict_proba(model, loader) -> tuple[float, np.ndarray, np.ndarray]:
    """Returns (avg_loss, proba, labels). Loss is 0 if no labels available."""
    model.eval()
    all_logits, all_labels, total_loss = [], [], 0.0
    for inputs, labs in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(**inputs).logits
            loss   = criterion(logits, labs.to(device))
        total_loss += loss.item()
        all_logits.append(logits.float().cpu())
        all_labels.append(labs)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels).numpy()
    proba  = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return total_loss / len(loader), proba, labels


@torch.no_grad()
def predict_proba_texts(model, texts: list[str]) -> np.ndarray:
    """Run inference on a raw text list; returns proba[:,1] array."""
    model.eval()
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="pt")
    all_proba = []
    _bsz = BATCH_SIZE * 2
    for i in range(0, len(texts), _bsz):
        batch = {k: v[i:i + _bsz].to(device) for k, v in enc.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(**batch).logits
        all_proba.append(torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(all_proba)


# ============================================================
# W&B
# ============================================================
print("\n[SECTION] Initializing W&B run")
wandb.login()
run = wandb.init(
    project="truth-classifier-transformers",
    config={
        "model":          MODEL_NAME,
        "input_format":   "speaker | party | subject | statement",
        "k_folds":        K_FOLDS,
        "max_length":     MAX_LENGTH,
        "batch_size":     BATCH_SIZE,
        "epochs":         EPOCHS,
        "freeze_epochs":  FREEZE_EPOCHS,
        "cls_dropout":    CLS_DROPOUT,
        "lr":             LR,
        "llrd_factor":    LLRD_FACTOR,
        "warmup_ratio":   WARMUP_RATIO,
        "weight_decay":   WEIGHT_DECAY,
        "class_weights":  CLASS_WEIGHTS,
        "scheduler":      "linear",
        "seed":           SEED,
        "n_trainval":     len(tv_idx),
        "n_holdout":      len(ho_idx),
        "use_amp":        USE_AMP,
        "device":         str(device),
    },
)

loss_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
criterion    = nn.CrossEntropyLoss(weight=loss_weights)


# ============================================================
# K-Fold loop
# ============================================================
print(f"\n[SECTION] K-Fold training  [{_now()}]")
print(f"  K={K_FOLDS}  EPOCHS={EPOCHS}  FREEZE_EPOCHS={FREEZE_EPOCHS}")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

oof_proba        = np.zeros(len(tv_idx))   # OOF probas over all trainval rows
ho_proba_folds   = []                       # holdout proba from each fold
test_proba_folds = []                       # test proba from each fold

for fold_k, (tr_rel, val_rel) in enumerate(skf.split(tv_idx, tv_labels)):
    print(f"\n{'='*70}")
    print(f"  FOLD {fold_k + 1}/{K_FOLDS}  [{_now()}]")
    print(f"{'='*70}")

    tr_abs  = tv_idx[tr_rel]
    val_abs = tv_idx[val_rel]

    X_tr_f  = [all_texts[i] for i in tr_abs]
    y_tr_f  = all_labels[tr_abs]
    X_val_f = [all_texts[i] for i in val_abs]
    y_val_f = all_labels[val_abs]

    print(f"  Train: {len(X_tr_f):,}   Val: {len(X_val_f):,}")

    # Build fold datasets
    _t0 = time()
    train_ds = StatementDataset(X_tr_f, y_tr_f)
    val_ds   = StatementDataset(X_val_f, y_val_f)
    print(f"  Tokenized in {time()-_t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,   shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Load fresh model
    _cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    _cfg.cls_dropout = CLS_DROPOUT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=_cfg, torch_dtype=torch.float32,
    )
    model.to(device)

    # LLRD optimizer — no freeze path (FREEZE_EPOCHS=0)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    p2_steps  = total_steps
    p2_warmup = warmup_steps

    if FREEZE_EPOCHS > 0:
        _freeze_backbone(model)
        p1_steps  = len(train_loader) * FREEZE_EPOCHS
        p1_warmup = int(p1_steps * WARMUP_RATIO)
        p1_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(p1_params, lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = get_linear_schedule_with_warmup(optimizer, p1_warmup, p1_steps)
        p2_steps  = len(train_loader) * (EPOCHS - FREEZE_EPOCHS)
        p2_warmup = int(p2_steps * WARMUP_RATIO)
        print(f"  Phase 1 optimizer (frozen) — lr={LR:.1e}")
    else:
        param_groups = _build_llrd_param_groups(model, LR, LLRD_FACTOR, WEIGHT_DECAY)
        _lr_min = min(g["lr"] for g in param_groups)
        _lr_max = max(g["lr"] for g in param_groups)
        optimizer = AdamW(param_groups)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        print(f"  LLRD optimizer — {len(param_groups)} groups  LR range: [{_lr_min:.2e}, {_lr_max:.2e}]")

    # Training loop
    best_val_f1 = -1.0
    best_ckpt   = OUTPUT_DIR / f"fold{fold_k+1}-best.pt"

    for epoch in range(1, EPOCHS + 1):
        _t = time()
        if FREEZE_EPOCHS > 0 and epoch == FREEZE_EPOCHS + 1:
            print(f"\n  [Phase 2] Unfreezing backbone + LLRD")
            _unfreeze_backbone(model)
            param_groups = _build_llrd_param_groups(model, LR, LLRD_FACTOR, WEIGHT_DECAY)
            _lr_min = min(g["lr"] for g in param_groups)
            _lr_max = max(g["lr"] for g in param_groups)
            optimizer = AdamW(param_groups)
            scheduler = get_linear_schedule_with_warmup(optimizer, p2_warmup, p2_steps)
            print(f"  LLRD groups: {len(param_groups)}  LR range: [{_lr_min:.2e}, {_lr_max:.2e}]")

        print(f"\n  --- Epoch {epoch}/{EPOCHS} ---  [{_now()}]")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
        val_loss, val_proba, val_labels_epoch = predict_proba(model, val_loader)

        val_pred = (val_proba >= 0.5).astype(int)
        val_f1   = f1_score(val_labels_epoch, val_pred, average="macro", zero_division=0)
        val_auc  = roc_auc_score(val_labels_epoch, val_proba)

        print(
            f"  Fold {fold_k+1} Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  val_roc_auc={val_auc:.4f}  "
            f"({time()-_t:.1f}s)"
        )
        wandb.log({
            "epoch": (fold_k * EPOCHS) + epoch,
            f"fold{fold_k+1}/train_loss": train_loss,
            f"fold{fold_k+1}/val_loss":   val_loss,
            f"fold{fold_k+1}/val_macro_f1": val_f1,
            f"fold{fold_k+1}/val_roc_auc":  val_auc,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), best_ckpt)
            print(f"    New best val macro_f1={best_val_f1:.4f} — checkpoint saved")

    print(f"\n  Fold {fold_k+1} best val macro_f1={best_val_f1:.4f}")
    wandb.log({f"fold{fold_k+1}/best_val_macro_f1": best_val_f1})

    # Load best checkpoint for this fold
    model.load_state_dict(torch.load(best_ckpt, map_location=device))

    # OOF probas on this fold's val rows
    _, val_proba_best, _ = predict_proba(model, val_loader)
    oof_proba[val_rel]   = val_proba_best

    # Holdout probas from this fold
    _, ho_proba_fold, _ = predict_proba(model, holdout_loader)
    ho_proba_folds.append(ho_proba_fold)

    # Test probas from this fold
    if create_kaggle_csv:
        test_proba_folds.append(predict_proba_texts(model, test_texts))
        print(f"  Test inference done ({len(test_texts):,} rows)")

    # Free GPU memory before next fold
    del model
    torch.cuda.empty_cache()
    print(f"  GPU memory cleared")


# ============================================================
# OOF evaluation (full trainval)
# ============================================================
print(f"\n[SECTION] OOF evaluation (trainval, N={len(tv_idx):,})  [{_now()}]")

print(f"  Threshold tuning on OOF probas")
grid = np.arange(0.20, 0.76, 0.01)
oof_scores = {
    round(float(t), 2): f1_score(tv_labels, (oof_proba >= t).astype(int),
                                 average="macro", zero_division=0)
    for t in grid
}
best_t = max(oof_scores, key=oof_scores.get)
print(f"  {'threshold':>10}   macro_f1")
for t, s in oof_scores.items():
    print(f"  {t:>10.2f}   {s:.4f}{'  ←' if t == best_t else ''}")
print(f"\n  Best OOF threshold: {best_t:.2f}  (OOF macro_f1={oof_scores[best_t]:.4f})")
THRESHOLD = best_t

oof_pred   = (oof_proba >= THRESHOLD).astype(int)
oof_f1     = f1_score(tv_labels, oof_pred, average="macro", zero_division=0)
oof_auc    = roc_auc_score(tv_labels, oof_proba)
print(f"\n  OOF macro_f1={oof_f1:.4f}  OOF roc_auc={oof_auc:.4f}")

wandb.log({"oof/macro_f1": oof_f1, "oof/roc_auc": oof_auc,
           "threshold/oof_best": best_t})


# ============================================================
# Ensemble holdout evaluation
# ============================================================
print(f"\n[SECTION] Ensemble holdout evaluation  [{_now()}]")
print(f"  Threshold: {THRESHOLD:.2f}")

ho_proba_ensemble = np.mean(ho_proba_folds, axis=0)
ho_pred = (ho_proba_ensemble >= THRESHOLD).astype(int)

holdout_metrics = {
    "roc_auc":      roc_auc_score(ho_labels, ho_proba_ensemble),
    "pr_auc":       average_precision_score(ho_labels, ho_proba_ensemble),
    "macro_f1":     f1_score(ho_labels, ho_pred, average="macro", zero_division=0),
    "f1":           f1_score(ho_labels, ho_pred, zero_division=0),
    "precision":    precision_score(ho_labels, ho_pred, zero_division=0),
    "recall":       recall_score(ho_labels, ho_pred, zero_division=0),
    "accuracy":     accuracy_score(ho_labels, ho_pred),
    "mcc":          matthews_corrcoef(ho_labels, ho_pred),
    "balanced_acc": balanced_accuracy_score(ho_labels, ho_pred),
}
cm = confusion_matrix(ho_labels, ho_pred)

print("\nHoldout results (5-fold ensemble):")
for name, val in holdout_metrics.items():
    print(f"  {name}: {val:.4f}")
print(f"\n{classification_report(ho_labels, ho_pred)}")


# ============================================================
# Plots + W&B logging
# ============================================================
print("\n[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(ho_labels, ho_proba_ensemble)
prec_c, rec_c, _ = precision_recall_curve(ho_labels, ho_proba_ensemble)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(fpr, tpr, label=f"ROC-AUC = {holdout_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set(title=f"ROC Curve — {model_slug} (holdout)", xlabel="FPR", ylabel="TPR")
axes[0].legend()
axes[1].plot(rec_c, prec_c, label=f"PR-AUC = {holdout_metrics['pr_auc']:.4f}")
axes[1].set(title="Precision-Recall Curve (holdout)", xlabel="Recall", ylabel="Precision")
axes[1].legend()
im = axes[2].imshow(cm, interpolation="nearest", cmap="Blues")
axes[2].set_title("Confusion Matrix (holdout, ensemble)")
axes[2].set_xticks([0, 1]); axes[2].set_xticklabels(["True (0)", "False (1)"])
axes[2].set_yticks([0, 1]); axes[2].set_yticklabels(["True (0)", "False (1)"])
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
fig.colorbar(im, ax=axes[2])
plt.tight_layout()

wandb.log({
    **{f"holdout/{k}": v for k, v in holdout_metrics.items()},
    "holdout/tn": int(cm[0, 0]), "holdout/fp": int(cm[0, 1]),
    "holdout/fn": int(cm[1, 0]), "holdout/tp": int(cm[1, 1]),
    "roc_pr_cm": wandb.Image(fig),
    "confusion_matrix": wandb.plot.confusion_matrix(
        y_true=ho_labels.tolist(), preds=ho_pred.tolist(),
        class_names=["True (0)", "False (1)"],
    ),
})
run.summary["holdout/macro_f1"] = holdout_metrics["macro_f1"]
run.summary["holdout/roc_auc"]  = holdout_metrics["roc_auc"]
run.summary["oof/macro_f1"]     = oof_f1

print("\n[SECTION] Finishing W&B run")
run.finish()


# ============================================================
# Save OOF probas (for late-fusion stacking in Experiment 3)
# ============================================================
print(f"\n[SECTION] Saving OOF probas  [{_now()}]")
oof_df = pd.DataFrame({
    "idx":       tv_idx,
    "oof_proba": oof_proba,
    "label":     tv_labels,
})
oof_path = OUTPUT_DIR / f"{model_slug}-oof.csv"
oof_df.to_csv(oof_path, index=False)
print(f"  OOF saved: {oof_path}  ({len(oof_df):,} rows)")


# ============================================================
# Kaggle submission CSV
# ============================================================
if create_kaggle_csv:
    print(f"\n[SECTION] Creating Kaggle submission  [{_now()}]")
    test_proba_ensemble = np.mean(test_proba_folds, axis=0)
    test_pred = (test_proba_ensemble >= THRESHOLD).astype(int)

    sub_dir = Path("/kaggle/working") if IS_KAGGLE else (_root / "submissions")
    sub_dir.mkdir(exist_ok=True)
    sub_path = sub_dir / f"submission-{model_slug}-{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
    pd.DataFrame({"id": df_test["id"], "label": test_pred}).to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}  ({len(test_pred):,} rows)")

print(f"\n[DONE] Total time: {time()-_script_start:.1f}s  [{_now()}]")
