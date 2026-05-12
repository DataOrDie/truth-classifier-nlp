"""
Transformer fine-tuning — Option A: text-only classification.

Architecture
------------
[statement tokens] → DeBERTa-v3-small encoder → [CLS] → dropout → Linear(768 → 2)

Loss       : CrossEntropyLoss with class weights {0: 1.42, 1: 0.77}
Optimizer  : AdamW with layer-wise LR decay (LLRD): head=2e-5, each encoder layer ×0.9 deeper
Epochs     : 3  (best checkpoint saved by val macro_f1)
Max tokens : 128  (covers 99%+ of statements; halves VRAM vs. 512)
Precision  : FP32  (DeBERTa-v3 unstable in BF16)

Split
-----
  80% trainval → 90/10 → train | val   (val used for checkpoint selection + threshold tuning)
  20% holdout                           (final metric reporting only — not seen during training)

Both splits use random_state=42, stratified — identical holdout to stacking.py.
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
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Environment detection
# ============================================================
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    # Add the competition dataset as a Kaggle dataset input named "truth-classifier-nlp"
    DATA_DIR   = Path("/kaggle/input/truth-classifier-nlp")
    OUTPUT_DIR = Path("/kaggle/working/models/transformer")
else:
    def _find_root(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "data" / "train.csv").exists():
                return p
        raise FileNotFoundError("Cannot find project root — run from anywhere inside the repo")
    _root      = _find_root(Path.cwd())
    DATA_DIR   = _root / "data"
    OUTPUT_DIR = _root / "models" / "transformer"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================
MODEL_NAME   = "microsoft/deberta-v3-small"
MAX_LENGTH   = 128
BATCH_SIZE   = 16       # safe for 12 GB VRAM; bump to 32 on Kaggle T4 (16 GB)
EPOCHS       = 3
LR           = 2e-5     # head LR; encoder layers decay by LLRD_FACTOR per layer
LLRD_FACTOR  = 0.9      # layer-wise LR decay multiplier
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

CLASS_WEIGHTS = [1.42, 0.77]   # {0: true, 1: false} — same as all other scripts
THRESHOLD     = 0.5
SEED          = 42

enable_threshold_tuning = True
create_kaggle_csv       = True
model_slug              = "deberta-v3-small"

# num_workers > 0 causes issues with multiprocessing on Windows
NUM_WORKERS = 0 if sys.platform == "win32" else 2

torch.manual_seed(SEED)
np.random.seed(SEED)

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = False  # DeBERTa-v3 is unstable in BF16; FP32 fits in 12 GB at batch_size=16

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
# Data — 3-way split
# ============================================================
print(f"\n[SECTION] Loading data  [{_now()}]")
df = pd.read_csv(DATA_DIR / "train.csv")
print(f"  Rows: {len(df):,}  |  Labels: {df['label'].value_counts().to_dict()}")

print(f"\n[SECTION] Dataset features")
meta_cols = [c for c in df.columns if c not in ("id", "label", "statement")]
print(f"  Used       : ['statement']  (raw text, tokenized)")
print(f"  Not used   : {meta_cols}")
tok_len = df["statement"].str.split().str.len()
print(f"  Token len  : min={tok_len.min()}  median={tok_len.median():.0f}  p99={tok_len.quantile(0.99):.0f}  max={tok_len.max()}")
for col in meta_cols:
    n_unique = df[col].nunique()
    top3 = df[col].value_counts().head(3).index.tolist()
    print(f"  {col:<22}: {n_unique:>4} unique  top-3={top3}")

texts  = df["statement"].tolist()
labels = df["label"].tolist()

# Outer split: same seed + stratify as stacking.py so holdout rows are identical
X_tv, X_ho, y_tv, y_ho = train_test_split(
    texts, labels, test_size=0.2, random_state=SEED, stratify=labels
)
# Inner split: 10% of trainval → val for threshold tuning & early stopping
X_tr, X_val, y_tr, y_val = train_test_split(
    X_tv, y_tv, test_size=0.1, random_state=SEED, stratify=y_tv
)
print(f"  Train: {len(X_tr):,}   Val: {len(X_val):,}   Holdout: {len(X_ho):,}")


# ============================================================
# Tokenizer + Dataset
# ============================================================
print(f"\n[SECTION] Tokenizing  [{_now()}]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class StatementDataset(Dataset):
    def __init__(self, texts: list[str], labels: list[int]):
        self.enc    = tokenizer(texts, truncation=True, padding="max_length",
                                max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


_t0 = time()
train_ds   = StatementDataset(X_tr,  y_tr)
val_ds     = StatementDataset(X_val, y_val)
holdout_ds = StatementDataset(X_ho,  y_ho)
print(f"  Tokenized in {time()-_t0:.1f}s")

_pin = USE_AMP
train_loader   = DataLoader(train_ds,   batch_size=BATCH_SIZE,   shuffle=True,
                            num_workers=NUM_WORKERS, pin_memory=_pin)
val_loader     = DataLoader(val_ds,     batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=_pin)
holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE*2, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=_pin)


# ============================================================
# LLRD optimizer builder
# ============================================================
def _build_llrd_param_groups(model, base_lr: float, llrd_factor: float, weight_decay: float) -> list:
    """Layer-wise LR decay: head gets base_lr, each encoder layer is multiplied by llrd_factor going down."""
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

    # Head (classifier + pooler) — full LR
    head = [n for n in param_dict
            if "deberta.encoder.layer." not in n and "deberta.embeddings." not in n]
    _add(head, base_lr)

    # Encoder layers: top (layer 11) → bottom (layer 0), depth=1 at top
    for layer_idx in range(num_layers - 1, -1, -1):
        depth = num_layers - layer_idx                  # 1 for top layer, num_layers for bottom
        lr    = base_lr * (llrd_factor ** depth)
        _add([n for n in param_dict if f"deberta.encoder.layer.{layer_idx}." in n], lr)

    # Embeddings — lowest LR
    embed_lr = base_lr * (llrd_factor ** (num_layers + 1))
    _add([n for n in param_dict if "deberta.embeddings." in n and n not in assigned], embed_lr)

    return groups


# ============================================================
# Model
# ============================================================
print(f"\n[SECTION] Loading model: {MODEL_NAME}  [{_now()}]")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2, torch_dtype=torch.float32)
model.to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Parameters: {n_params:,}")

loss_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
criterion    = nn.CrossEntropyLoss(weight=loss_weights)

param_groups = _build_llrd_param_groups(model, LR, LLRD_FACTOR, WEIGHT_DECAY)
optimizer    = AdamW(param_groups)
print(f"  LLRD groups: {len(param_groups)}  LR range: [{min(g['lr'] for g in param_groups):.2e}, {max(g['lr'] for g in param_groups):.2e}]")

total_steps  = len(train_loader) * EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)


# ============================================================
# W&B
# ============================================================
print("\n[SECTION] Initializing W&B run")
wandb.login()
run = wandb.init(
    project="truth-classifier-transformers",
    config={
        "model":         MODEL_NAME,
        "max_length":    MAX_LENGTH,
        "batch_size":    BATCH_SIZE,
        "epochs":        EPOCHS,
        "lr":            LR,
        "warmup_ratio":  WARMUP_RATIO,
        "weight_decay":  WEIGHT_DECAY,
        "llrd_factor":   LLRD_FACTOR,
        "class_weights": CLASS_WEIGHTS,
        "scheduler":     "linear",
        "seed":          SEED,
        "n_train":       len(X_tr),
        "n_val":         len(X_val),
        "n_holdout":     len(X_ho),
        "use_amp":       USE_AMP,
        "device":        str(device),
    },
)


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
            print(f"    Batch 0 — logits dtype={logits.dtype}  labs dtype={labs.dtype}  loss={loss.item():.4f}")
        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}  avg_loss={total_loss/(batch_idx+1):.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader) -> tuple[float, np.ndarray, np.ndarray]:
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


# ============================================================
# Training loop — save best checkpoint by val macro_f1
# ============================================================
print(f"\n[SECTION] Training  [{_now()}]")
print(f"  Model dtype     : {next(model.parameters()).dtype}")
print(f"  loss_weights    : {loss_weights}")
print(f"  Train batches   : {len(train_loader)}  Val batches: {len(val_loader)}")

best_val_f1 = -1.0
best_ckpt   = OUTPUT_DIR / f"{model_slug}-best.pt"

for epoch in range(1, EPOCHS + 1):
    _t = time()
    print(f"\n  --- Epoch {epoch}/{EPOCHS} ---  [{_now()}]")
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion)
    print(f"  Train loss: {train_loss:.4f}  — starting val evaluation")
    val_loss, val_proba, val_labels = evaluate(model, val_loader)
    print(f"  Val proba range: [{val_proba.min():.4f}, {val_proba.max():.4f}]  NaNs: {np.isnan(val_proba).sum()}")

    val_pred = (val_proba >= 0.5).astype(int)
    val_f1   = f1_score(val_labels, val_pred, average="macro", zero_division=0)
    val_auc  = roc_auc_score(val_labels, val_proba)

    print(
        f"  Epoch {epoch}/{EPOCHS}  "
        f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
        f"val_macro_f1={val_f1:.4f}  val_roc_auc={val_auc:.4f}  "
        f"({time()-_t:.1f}s)"
    )
    wandb.log({"epoch": epoch, "train/loss": train_loss,
               "val/loss": val_loss, "val/macro_f1": val_f1, "val/roc_auc": val_auc})

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), best_ckpt)
        print(f"    New best val macro_f1={best_val_f1:.4f} — checkpoint saved")


# ============================================================
# Load best checkpoint
# ============================================================
print(f"\n[SECTION] Loading best checkpoint  [{_now()}]")
model.load_state_dict(torch.load(best_ckpt, map_location=device))


# ============================================================
# Threshold tuning on val set
# ============================================================
_, val_proba, val_labels = evaluate(model, val_loader)

if enable_threshold_tuning:
    print(f"\n[SECTION] Threshold tuning on val set  [{_now()}]")
    grid   = np.arange(0.20, 0.76, 0.01)
    scores = {
        round(float(t), 2): f1_score(val_labels, (val_proba >= t).astype(int),
                                     average="macro", zero_division=0)
        for t in grid
    }
    best_t = max(scores, key=scores.get)

    print(f"  {'threshold':>10}   macro_f1")
    for t, s in scores.items():
        print(f"  {t:>10.2f}   {s:.4f}{'  ←' if t == best_t else ''}")
    print(f"\n  Best threshold: {best_t:.2f}  (val macro_f1={scores[best_t]:.4f})")

    wandb.log({"threshold/best": best_t, "threshold/val_macro_f1": scores[best_t]})
    THRESHOLD = best_t


# ============================================================
# Holdout evaluation
# ============================================================
print(f"\n[SECTION] Holdout evaluation  [{_now()}]")
print(f"  Threshold: {THRESHOLD:.2f}")

_, ho_proba, ho_labels = evaluate(model, holdout_loader)
ho_pred = (ho_proba >= THRESHOLD).astype(int)

holdout_metrics = {
    "roc_auc":      roc_auc_score(ho_labels, ho_proba),
    "pr_auc":       average_precision_score(ho_labels, ho_proba),
    "macro_f1":     f1_score(ho_labels, ho_pred, average="macro", zero_division=0),
    "f1":           f1_score(ho_labels, ho_pred, zero_division=0),
    "precision":    precision_score(ho_labels, ho_pred, zero_division=0),
    "recall":       recall_score(ho_labels, ho_pred, zero_division=0),
    "accuracy":     accuracy_score(ho_labels, ho_pred),
    "mcc":          matthews_corrcoef(ho_labels, ho_pred),
    "balanced_acc": balanced_accuracy_score(ho_labels, ho_pred),
}
cm = confusion_matrix(ho_labels, ho_pred)

print("\nHoldout results:")
for name, val in holdout_metrics.items():
    print(f"  {name}: {val:.4f}")
print(f"\n{classification_report(ho_labels, ho_pred)}")


# ============================================================
# Plots + W&B logging
# ============================================================
print("\n[SECTION] Generating plots")
fpr, tpr, _      = roc_curve(ho_labels, ho_proba)
prec_c, rec_c, _ = precision_recall_curve(ho_labels, ho_proba)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(fpr, tpr, label=f"ROC-AUC = {holdout_metrics['roc_auc']:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.6)
axes[0].set(title=f"ROC Curve — {model_slug} (holdout)", xlabel="FPR", ylabel="TPR")
axes[0].legend()

axes[1].plot(rec_c, prec_c, label=f"PR-AUC = {holdout_metrics['pr_auc']:.4f}")
axes[1].set(title="Precision-Recall Curve (holdout)", xlabel="Recall", ylabel="Precision")
axes[1].legend()

im = axes[2].imshow(cm, interpolation="nearest", cmap="Blues")
axes[2].set_title("Confusion Matrix (holdout)")
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

print("\n[SECTION] Finishing W&B run")
run.finish()


# ============================================================
# Save artifacts
# ============================================================
print(f"\n[SECTION] Saving artifacts  [{_now()}]")
model.save_pretrained(OUTPUT_DIR / f"{model_slug}-model")
tokenizer.save_pretrained(OUTPUT_DIR / f"{model_slug}-tokenizer")
joblib.dump(THRESHOLD, OUTPUT_DIR / f"{model_slug}-threshold.joblib")
print(f"  Artifacts saved to: {OUTPUT_DIR}")


# ============================================================
# Kaggle submission CSV
# ============================================================
if create_kaggle_csv:
    print(f"\n[SECTION] Creating Kaggle submission  [{_now()}]")
    df_test    = pd.read_csv(DATA_DIR / "test_nolabel.csv")
    test_texts = df_test["statement"].tolist()

    test_enc = tokenizer(
        test_texts, truncation=True, padding="max_length",
        max_length=MAX_LENGTH, return_tensors="pt",
    )

    model.eval()
    all_proba = []
    _bsz = BATCH_SIZE * 2
    with torch.no_grad():
        for i in range(0, len(test_texts), _bsz):
            batch  = {k: v[i:i + _bsz].to(device) for k, v in test_enc.items()}
            with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
                logits = model(**batch).logits
            proba = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
            all_proba.append(proba)

    test_proba = np.concatenate(all_proba)
    test_pred  = (test_proba >= THRESHOLD).astype(int)

    sub_dir = Path("/kaggle/working") if IS_KAGGLE else (_root / "submissions")
    sub_dir.mkdir(exist_ok=True)
    sub_path = sub_dir / f"submission-{model_slug}-{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
    pd.DataFrame({"id": df_test["id"], "label": test_pred}).to_csv(sub_path, index=False)
    print(f"  Submission saved: {sub_path}  ({len(test_pred):,} rows)")

print(f"\n[DONE] Total time: {time()-_script_start:.1f}s  [{_now()}]")
