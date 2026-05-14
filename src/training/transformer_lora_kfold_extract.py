"""
Inference-only script — extract holdout and test probas from saved LoRA adapters.

Run this once after transformer_lora_kfold.py has completed. Outputs:
  models/transformer_lora_kfold/ho_proba.npy   — shape (1790,)
  models/transformer_lora_kfold/ho_idx.npy     — shape (1790,)
  models/transformer_lora_kfold/test_proba.npy — shape (3836,)

These files are auto-detected by stacking.py (LoRA > base > small priority).
Expected runtime: ~20–30 min on RTX 5070.
"""
from pathlib import Path
import gc
import sys
from time import time

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)


# ============================================================
# Paths  (must match transformer_lora_kfold.py)
# ============================================================
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "data" / "train.csv").exists():
            return p
    raise FileNotFoundError("Cannot find project root")

_root      = _find_root(Path.cwd())
DATA_DIR   = _root / "data"
MODEL_DIR  = _root / "models" / "transformer_lora_kfold"

if not MODEL_DIR.exists():
    raise FileNotFoundError(
        f"LoRA model dir not found: {MODEL_DIR}\nRun transformer_lora_kfold.py first."
    )


# ============================================================
# Config  (must match transformer_lora_kfold.py exactly)
# ============================================================
MODEL_NAME  = "mistralai/Mistral-7B-v0.1"
MAX_LENGTH  = 128
BATCH_SIZE  = 8      # inference — larger batch is fine
K_FOLDS     = 5
SEED        = 42

NUM_WORKERS = 0 if sys.platform == "win32" else 2
device      = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU : {torch.cuda.get_device_name(0)}")


# ============================================================
# Input formatter  (must be identical to training script)
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
# Data — same splits as transformer_lora_kfold.py
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
tokenizer.pad_token    = tokenizer.eos_token
tokenizer.padding_side = "left"


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
# Inference helper
# ============================================================
@torch.no_grad()
def infer_proba(model, loader) -> np.ndarray:
    model.eval()
    all_proba = []
    for inputs in loader:
        inputs = {k: v.to(device) for k, v in inputs.items()}
        logits = model(**inputs).logits
        proba  = torch.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()
        all_proba.append(proba)
    return np.concatenate(all_proba)


# ============================================================
# Load each fold adapter and collect probas
# ============================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

print("\n[SECTION] Running inference across all folds")
ho_proba_folds   = []
test_proba_folds = []

for fold_k in range(1, K_FOLDS + 1):
    adapter_dir = MODEL_DIR / f"fold{fold_k}-adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_dir}")

    print(f"\n  Fold {fold_k}/{K_FOLDS} — loading adapter from {adapter_dir.name}")
    _t = time()

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map={"": 0},
        pad_token_id=tokenizer.eos_token_id,
    )
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    ho_proba_folds.append(infer_proba(model, ho_loader))
    test_proba_folds.append(infer_proba(model, test_loader))
    print(f"  Fold {fold_k} done in {time()-_t:.1f}s")

    del model, base_model
    gc.collect()
    torch.cuda.synchronize()
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
