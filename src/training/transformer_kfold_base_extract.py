"""
Inference-only script — extract holdout and test probas from saved base k-fold checkpoints.

Run this once after transformer_kfold_base.py has completed. Outputs:
  models/transformer_kfold_base/ho_proba.npy   — shape (1790,)  ensemble holdout probas
  models/transformer_kfold_base/ho_idx.npy     — shape (1790,)  original df indices
  models/transformer_kfold_base/test_proba.npy — shape (3836,)  ensemble test probas

These files are auto-detected by stacking.py (base takes priority over small).
Takes ~4–5 min on RTX 5070 (inference only, no training).
"""
from pathlib import Path
import sys
from time import time

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


# ============================================================
# Paths  (must match transformer_kfold_base.py)
# ============================================================
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "train.csv").exists():
            return p
    raise FileNotFoundError("Cannot find project root")

_root      = _find_root(Path.cwd())
DATA_DIR   = _root / "data"
MODEL_DIR  = _root / "models" / "transformer_kfold_base"

if not MODEL_DIR.exists():
    raise FileNotFoundError(f"k-fold base model dir not found: {MODEL_DIR}\nRun transformer_kfold_base.py first.")


# ============================================================
# Config  (must match transformer_kfold_base.py exactly)
# ============================================================
MODEL_NAME  = "microsoft/deberta-v3-base"
MAX_LENGTH  = 128
BATCH_SIZE  = 32
K_FOLDS     = 5
SEED        = 42
CLS_DROPOUT = 0.3

NUM_WORKERS = 0 if sys.platform == "win32" else 2
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP     = False

print(f"Device: {device}")
if device.type == "cuda":
    print(f"  GPU : {torch.cuda.get_device_name(0)}")


# ============================================================
# Input formatter  (must be identical to transformer_kfold_base.py)
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
# Data — same splits as transformer_kfold_base.py
# ============================================================
print("\n[SECTION] Loading data")
df = pd.read_csv(DATA_DIR / "train.csv")
all_labels = np.array(df["label"].tolist())
all_idx    = np.arange(len(df))

tv_idx, ho_idx = train_test_split(
    all_idx, test_size=0.2, random_state=SEED, stratify=all_labels
)
print(f"  Trainval: {len(tv_idx):,}   Holdout: {len(ho_idx):,}")

X_ho = df.iloc[ho_idx].apply(
    lambda r: format_input(r["speaker"], r["party_affiliation"],
                           r["subject"], r["statement"]),
    axis=1,
).tolist()
y_ho = all_labels[ho_idx]

df_test    = pd.read_csv(DATA_DIR / "test_nolabel.csv")
test_texts = df_test.apply(
    lambda r: format_input(r["speaker"], r["party_affiliation"],
                           r["subject"], r["statement"]),
    axis=1,
).tolist()
print(f"  Test rows: {len(test_texts):,}")


# ============================================================
# Tokenizer
# ============================================================
print("\n[SECTION] Loading tokenizer")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


class TextDataset(Dataset):
    def __init__(self, texts: list[str]):
        self.enc = tokenizer(texts, truncation=True, padding="max_length",
                             max_length=MAX_LENGTH, return_tensors="pt")

    def __len__(self) -> int:
        return len(self.enc["input_ids"])

    def __getitem__(self, idx: int):
        return {k: v[idx] for k, v in self.enc.items()}


print("  Tokenizing holdout and test sets...")
_t0 = time()
ho_ds   = TextDataset(X_ho)
test_ds = TextDataset(test_texts)
ho_loader   = DataLoader(ho_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"  Tokenized in {time()-_t0:.1f}s")


# ============================================================
# Inference helpers
# ============================================================
@torch.no_grad()
def infer_proba(model, loader) -> np.ndarray:
    model.eval()
    all_proba = []
    for inputs in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=USE_AMP):
            logits = model(**inputs).logits
        proba = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba)


# ============================================================
# Load each fold checkpoint and collect probas
# ============================================================
print("\n[SECTION] Running inference across all folds")
ho_proba_folds   = []
test_proba_folds = []

for fold_k in range(1, K_FOLDS + 1):
    ckpt_path = MODEL_DIR / f"fold{fold_k}-best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\n  Fold {fold_k}/{K_FOLDS} — loading {ckpt_path.name}")
    _cfg = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    _cfg.cls_dropout = CLS_DROPOUT
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, config=_cfg, torch_dtype=torch.float32,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)

    _t = time()
    ho_proba_folds.append(infer_proba(model, ho_loader))
    test_proba_folds.append(infer_proba(model, test_loader))
    print(f"  Fold {fold_k} done in {time()-_t:.1f}s")

    del model
    torch.cuda.empty_cache()


# ============================================================
# Average and save
# ============================================================
print("\n[SECTION] Saving outputs")
ho_proba_ensemble   = np.mean(ho_proba_folds,   axis=0)
test_proba_ensemble = np.mean(test_proba_folds, axis=0)

np.save(MODEL_DIR / "ho_proba.npy",   ho_proba_ensemble)
np.save(MODEL_DIR / "ho_idx.npy",     ho_idx)
np.save(MODEL_DIR / "test_proba.npy", test_proba_ensemble)

print(f"  ho_proba.npy   : shape={ho_proba_ensemble.shape}  range=[{ho_proba_ensemble.min():.3f}, {ho_proba_ensemble.max():.3f}]")
print(f"  ho_idx.npy     : shape={ho_idx.shape}")
print(f"  test_proba.npy : shape={test_proba_ensemble.shape}  range=[{test_proba_ensemble.min():.3f}, {test_proba_ensemble.max():.3f}]")
print(f"\n  All saved to: {MODEL_DIR}")
print(f"\n[DONE]")
