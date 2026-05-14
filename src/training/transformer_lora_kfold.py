"""
Transformer fine-tuning — Experiment 5: LoRA on Mistral-7B-v0.1.

Architecture
------------
- Base: mistralai/Mistral-7B-v0.1, loaded in 4-bit NF4 (~3.5 GB VRAM for weights)
- LoRA: r=8, alpha=16, target=q/k/v/o projections, ~4M trainable params
- AutoModelForSequenceClassification head (class 0=true, class 1=false)
- Gradient checkpointing + gradient accumulation (effective batch = 16)
- Same 5-fold k-fold / OOF / late-fusion structure as transformer_kfold_base.py

Expected runtime: ~15–25 min per fold, ~90–120 min total (RTX 5070).
Adapter checkpoints saved to models/transformer_lora_kfold/fold{k}-adapter/

Install (one-time):
  pip install peft bitsandbytes accelerate
"""
from datetime import datetime
from pathlib import Path
import sys
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)


# ============================================================
# Environment detection
# ============================================================
IS_KAGGLE = Path("/kaggle/input").exists()

if IS_KAGGLE:
    DATA_DIR   = Path("/kaggle/input/truth-classifier-nlp")
    OUTPUT_DIR = Path("/kaggle/working/models/transformer_lora_kfold")
else:
    def _find_root(start: Path) -> Path:
        for p in [start, *start.parents]:
            if (p / "data" / "train.csv").exists():
                return p
        raise FileNotFoundError("Cannot find project root")
    _root      = _find_root(Path.cwd())
    DATA_DIR   = _root / "data"
    OUTPUT_DIR = _root / "models" / "transformer_lora_kfold"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Config
# ============================================================
MODEL_NAME  = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH  = 128
BATCH_SIZE  = 4
GRAD_ACCUM  = 4     # effective batch = 16
EPOCHS      = 2
LR          = 5e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
K_FOLDS     = 5

# LoRA
LORA_R               = 8
LORA_ALPHA           = 16
LORA_DROPOUT         = 0.05
LORA_TARGET_MODULES  = ["q_proj", "v_proj", "k_proj", "o_proj"]

CLASS_WEIGHTS = [1.42, 0.77]
THRESHOLD     = 0.5
SEED          = 42

create_kaggle_csv = True
model_slug        = "mistral-7b-lora-kfold"

NUM_WORKERS = 0 if sys.platform == "win32" else 2

torch.manual_seed(SEED)
np.random.seed(SEED)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU  : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


# ============================================================
# Timing helper
# ============================================================
_script_start = time()
def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ============================================================
# Input formatter
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

tv_idx, ho_idx = train_test_split(
    all_idx, test_size=0.2, random_state=SEED, stratify=all_labels
)
tv_labels = all_labels[tv_idx]
ho_labels = all_labels[ho_idx]
X_ho      = [all_texts[i] for i in ho_idx]
print(f"  Trainval: {len(tv_idx):,}   Holdout: {len(ho_idx):,}")

if create_kaggle_csv:
    df_test    = pd.read_csv(DATA_DIR / "test_nolabel.csv")
    test_texts = df_test.apply(
        lambda r: format_input(r["speaker"], r["party_affiliation"],
                               r["subject"], r["statement"]),
        axis=1,
    ).tolist()
    print(f"  Test rows: {len(test_texts):,}")


# ============================================================
# Tokenizer  (left-pad for causal decoder model)
# ============================================================
print(f"\n[SECTION] Loading tokenizer  [{_now()}]")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"


class StatementDataset(Dataset):
    def __init__(self, texts: list[str], labels):
        self.enc    = tokenizer(texts, truncation=True, padding="max_length",
                                max_length=MAX_LENGTH, return_tensors="pt")
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.enc.items()}, self.labels[idx]


print(f"  Tokenizing holdout ({len(X_ho):,} rows)...")
holdout_ds     = StatementDataset(X_ho, ho_labels)
holdout_loader = DataLoader(holdout_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                            num_workers=NUM_WORKERS)


# ============================================================
# Model factory — loads quantized base + wraps with LoRA
# ============================================================
def _build_model() -> nn.Module:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
        pad_token_id=tokenizer.eos_token_id,
    )
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Near-zero init for the classification head. The default random init
    # produces large logits against Mistral's unnormalized hidden states,
    # causing batch-0 losses of 10-17 that destabilize the LoRA adapters
    # in the first epoch. std=0.01 keeps initial loss near log(2) while
    # leaving weights non-zero so LoRA params still receive gradients.
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "score" in name and param.requires_grad:
                nn.init.normal_(param, mean=0.0, std=0.01)
                break

    return model


# ============================================================
# Training helpers
# ============================================================
loss_weights = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32).to(device)
criterion    = nn.CrossEntropyLoss(weight=loss_weights)


def train_epoch(model, loader, optimizer, scheduler) -> float:
    model.train()
    total_loss    = 0.0
    running_steps = 0
    optimizer.zero_grad()

    for batch_idx, (inputs, labs) in enumerate(loader):
        inputs = {k: v.to(device) for k, v in inputs.items()}
        labs   = labs.to(device)
        logits = model(**inputs).logits
        loss   = criterion(logits, labs) / GRAD_ACCUM
        loss.backward()

        is_update = (batch_idx + 1) % GRAD_ACCUM == 0 or (batch_idx + 1) == len(loader)
        if is_update:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            running_steps += 1

        total_loss += loss.item() * GRAD_ACCUM
        if batch_idx == 0:
            print(f"    Batch 0 — loss={loss.item() * GRAD_ACCUM:.4f}")
        if (batch_idx + 1) % 100 == 0:
            print(f"    Batch {batch_idx+1}/{len(loader)}  avg_loss={total_loss/(batch_idx+1):.4f}")

    return total_loss / len(loader)


@torch.no_grad()
def predict_proba(model, loader) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_logits, all_labels_list, total_loss = [], [], 0.0
    for inputs, labs in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        loss   = criterion(logits, labs.to(device))
        total_loss     += loss.item()
        all_logits.append(logits.float().cpu())
        all_labels_list.append(labs)
    logits = torch.cat(all_logits)
    labels = torch.cat(all_labels_list).numpy()
    proba  = torch.softmax(logits, dim=-1)[:, 1].numpy()
    return total_loss / len(loader), proba, labels


@torch.no_grad()
def predict_proba_texts(model, texts: list[str]) -> np.ndarray:
    model.eval()
    enc = tokenizer(texts, truncation=True, padding="max_length",
                    max_length=MAX_LENGTH, return_tensors="pt")
    all_proba = []
    _bsz = BATCH_SIZE * 2
    for i in range(0, len(texts), _bsz):
        batch = {k: v[i:i + _bsz].to(device) for k, v in enc.items()}
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
        "model":            MODEL_NAME,
        "input_format":     "speaker | party | subject | statement",
        "k_folds":          K_FOLDS,
        "max_length":       MAX_LENGTH,
        "batch_size":       BATCH_SIZE,
        "grad_accum":       GRAD_ACCUM,
        "effective_batch":  BATCH_SIZE * GRAD_ACCUM,
        "epochs":           EPOCHS,
        "lr":               LR,
        "warmup_ratio":     WARMUP_RATIO,
        "weight_decay":     WEIGHT_DECAY,
        "lora_r":           LORA_R,
        "lora_alpha":       LORA_ALPHA,
        "lora_dropout":     LORA_DROPOUT,
        "lora_targets":     LORA_TARGET_MODULES,
        "class_weights":    CLASS_WEIGHTS,
        "seed":             SEED,
        "n_trainval":       len(tv_idx),
        "n_holdout":        len(ho_idx),
        "device":           device,
    },
)


# ============================================================
# K-Fold loop
# ============================================================
print(f"\n[SECTION] K-Fold training  [{_now()}]")
print(f"  K={K_FOLDS}  EPOCHS={EPOCHS}  BATCH={BATCH_SIZE}  GRAD_ACCUM={GRAD_ACCUM}")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED)

oof_proba        = np.zeros(len(tv_idx))
ho_proba_folds   = []
test_proba_folds = []

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

    _t0 = time()
    train_ds = StatementDataset(X_tr_f, y_tr_f)
    val_ds   = StatementDataset(X_val_f, y_val_f)
    print(f"  Tokenized in {time()-_t0:.1f}s")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,   shuffle=True,
                              num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE*2, shuffle=False,
                              num_workers=NUM_WORKERS)

    # Load fresh quantized model + LoRA for this fold
    print(f"  Loading model ({MODEL_NAME})...")
    _t0    = time()
    model  = _build_model()
    print(f"  Model loaded in {time()-_t0:.1f}s")

    # Steps are counted over optimizer updates, not raw batches
    update_steps_per_epoch = len(train_loader) // GRAD_ACCUM
    total_updates          = update_steps_per_epoch * EPOCHS
    warmup_updates         = int(total_updates * WARMUP_RATIO)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_updates, total_updates)
    print(f"  Optimizer: {len(trainable_params)} param tensors  "
          f"LR={LR:.1e}  total_updates={total_updates}")

    best_val_f1  = -1.0
    best_ckpt_pt = OUTPUT_DIR / f"fold{fold_k+1}-best.pt"

    for epoch in range(1, EPOCHS + 1):
        _t = time()
        print(f"\n  --- Epoch {epoch}/{EPOCHS} ---  [{_now()}]")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler)
        val_loss, val_proba, val_labels_ep = predict_proba(model, val_loader)

        val_pred = (val_proba >= 0.5).astype(int)
        val_f1   = f1_score(val_labels_ep, val_pred, average="macro", zero_division=0)
        val_auc  = roc_auc_score(val_labels_ep, val_proba)

        print(
            f"  Fold {fold_k+1} Epoch {epoch}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_macro_f1={val_f1:.4f}  val_roc_auc={val_auc:.4f}  "
            f"({time()-_t:.1f}s)"
        )
        wandb.log({
            "epoch": (fold_k * EPOCHS) + epoch,
            f"fold{fold_k+1}/train_loss":    train_loss,
            f"fold{fold_k+1}/val_loss":      val_loss,
            f"fold{fold_k+1}/val_macro_f1":  val_f1,
            f"fold{fold_k+1}/val_roc_auc":   val_auc,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            # Save only trainable params (LoRA adapters + classification head)
            torch.save(
                {n: p.data.cpu() for n, p in model.named_parameters() if p.requires_grad},
                best_ckpt_pt,
            )
            print(f"    New best val macro_f1={best_val_f1:.4f} — checkpoint saved")

    print(f"\n  Fold {fold_k+1} best val macro_f1={best_val_f1:.4f}")
    wandb.log({f"fold{fold_k+1}/best_val_macro_f1": best_val_f1})

    # Restore best adapter weights in-place
    best_state = torch.load(best_ckpt_pt, map_location="cpu")
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.requires_grad and n in best_state:
                p.data.copy_(best_state[n].to(p.device))

    # Save PEFT adapter (for extract script)
    adapter_dir = OUTPUT_DIR / f"fold{fold_k+1}-adapter"
    model.save_pretrained(str(adapter_dir))
    print(f"  Adapter saved: {adapter_dir.name}")

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

    del model, optimizer, scheduler, best_state
    torch.cuda.empty_cache()
    print(f"  GPU memory cleared")


# ============================================================
# OOF evaluation (full trainval)
# ============================================================
print(f"\n[SECTION] OOF evaluation (trainval, N={len(tv_idx):,})  [{_now()}]")

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

oof_pred = (oof_proba >= THRESHOLD).astype(int)
oof_f1   = f1_score(tv_labels, oof_pred, average="macro", zero_division=0)
oof_auc  = roc_auc_score(tv_labels, oof_proba)
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
# Save OOF probas (for late-fusion stacking)
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
