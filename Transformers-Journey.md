# Transformers Journey

Experiment log for fine-tuned transformer models. Covers the rationale for moving beyond gradient boosting + stacking, the two execution paths (Kaggle GPU vs. PC CPU), model choices, and the hybrid architecture plan.

---

## Why Transformers After Stacking

The stacking ensemble squeezed the most signal available from the current feature representation — 384-dim sentence embeddings + metadata. The ceiling on that approach is roughly set by `all-MiniLM-L6-v2`, a small, fast embedding model that was not trained on political fact-checking content.

Fine-tuned transformers break through that ceiling by:

1. **Reading the raw statement directly.** Instead of compressing a statement into a fixed 384-dim vector before training, the transformer attends to every token and learns which words, phrases, and co-occurrence patterns predict truthfulness — within this specific dataset, during training.

2. **Task-specific representation learning.** Pre-trained LLMs already encode world knowledge, political language patterns, and hedging/certainty markers (e.g. *"claims that", "no evidence", "misrepresents"*). Fine-tuning shifts those representations toward the binary true/false signal in LIAR-style data.

3. **Complementary errors to the stacking ensemble.** The transformer's mistakes will correlate less with tree-based model mistakes than the LR/RFC/CatBoost/LGBM base models correlate with each other — making it a strong candidate for a late-fusion ensemble on top of stacking.

---

## Model Options

### DistilBERT (`distilbert-base-uncased`)

- 66M parameters, 40% smaller and 60% faster than BERT-base with ~97% of its performance on GLUE benchmarks.
- Fits in Kaggle's free T4 GPU with a batch size of 32, fine-tunes in ~15–20 min for 3 epochs on 8,950 samples.
- Good baseline: well-understood, stable training, lots of reference code.
- **Weakness:** it was distilled from BERT-base, so it inherits BERT's vocabulary and pre-training domain. No specific training on political or factual text.

### DeBERTa-v3-small (`microsoft/deberta-v3-small`)

- 86M parameters. Uses disentangled attention (separate position and content embeddings) and enhanced mask decoder.
- Consistently outperforms DistilBERT on NLI and fact-checking benchmarks — relevant because fake news detection is structurally close to NLI (statement vs. claim plausibility).
- Fine-tunes in ~20–25 min on T4.
- **Recommended starting point** for Kaggle runs. The performance gap over DistilBERT on short political statements is typically 1–3 F1 points.

### PC without GPU — Embedding-only mode

Full fine-tuning on CPU is prohibitively slow (~4–8 hours per epoch for these model sizes). The practical alternative:

- Use `all-mpnet-base-v2` (768-dim) as a drop-in upgrade in the existing preprocessing pipeline. This is Tier 4 from Planning.md and can be done immediately.
- Or extract frozen DeBERTa embeddings (mean-pool the last hidden state) and feed them to the stacking ensemble. This gives richer text representation without GPU fine-tuning.

---

## Architecture Plan

### Option A — Text-only fine-tuning (simplest, Kaggle)

```
[statement tokens] → DeBERTa encoder → [CLS] → dropout → Linear(768→2)
```

- Cross-entropy loss with class weights `{0: 1.42, 1: 0.77}`.
- Train with AdamW, lr=2e-5, linear warmup + decay, 3 epochs.
- Expected Macro F1: 0.64–0.68 based on similar LIAR-style benchmarks.

### Option B — Hybrid: transformer text head + metadata MLP (best ceiling)

```
[statement tokens] → DeBERTa encoder → [CLS] repr (768-dim)  ─┐
                                                                 ├→ concat → MLP → sigmoid
[metadata features] → BatchNorm → Linear → ReLU → Linear ──────┘
```

Metadata features to include: `speaker_primary_true_rate`, `subject_primary_true_rate`, `party_affiliation` (encoded), `is_major_party`, OOF probas from stacking ensemble (if available).

- Metadata path is a 2-layer MLP: `n_meta → 64 → 32` with ReLU and dropout.
- Combined head: `concat(768, 32) → 128 → 1` with sigmoid.
- Freeze DeBERTa for the first epoch, then unfreeze all layers.
- **Why this is the best ceiling:** metadata true-rate features are powerful predictors that pure text models can't access. The hybrid lets the transformer focus on language signals while the MLP routes speaker/party credibility.

### Option C — Late fusion (minimal code change)

Train the transformer text-only (Option A), then add its predicted probas as a new base model into the existing stacking ensemble. No hybrid architecture needed — the meta-LR in `stacking.py` absorbs the transformer signal.

This is the lowest-risk path: reuses all existing infrastructure and degrades gracefully if the transformer underperforms.

---

## Execution Plan

| Step | What | Where |                                                                                               
|------|------|--------|                                                                                              
| 1 | Upgrade embeddings to `all-mpnet-base-v2` in preprocessing | PC, now |                                          
| 2 | Write `transformer.py` — Option A text-only fine-tuning | Kaggle notebook |                                     
| 3 | Evaluate on holdout, compare to stacking F1 | Kaggle |                                                          
| 4 | If > stacking: add transformer proba to meta-LR (Option C) | Kaggle |                                           
| 5 | If time allows: implement Option B hybrid | Kaggle | and then  start to scaffold the fine-tuning script   
---

## What to Watch

- **Overfitting risk:** 8,950 samples is small for full fine-tuning. Monitor train vs. val loss per epoch. If val loss rises after epoch 2, stop early.
- **Class imbalance:** apply class weights in `CrossEntropyLoss(weight=torch.tensor([1.42, 0.77]))`. The imbalance is mild but consistent with prior experiments.
- **Threshold tuning:** same as tree models — the default 0.5 threshold will favour the majority class. Tune on the validation fold to maximise Macro F1.
- **Tokenization cutoff:** political statements in this dataset are short (median ~20 tokens). `max_length=128` is sufficient and halves memory vs. the default 512.

---

## Hardware

| Component | Spec | Notes |
|-----------|------|-------|
| CPU | AMD Ryzen 7 7700X, 8-core @ 4.5 GHz | — |
| RAM | 32 GB (31.1 GB usable) | — |
| GPU | NVIDIA GeForce RTX 5070 — **12 GB VRAM** | Blackwell (sm_120); needs PyTorch ≥ 2.6 |
| Storage | 1.82 TB total, ~150 GB used | Plenty for model checkpoints |

BERT-base / DeBERTa-v3-small at `batch_size=16, max_length=128` uses ~4–5 GB VRAM — comfortable on 12 GB.
Steps 2–5 can run locally; Kaggle is optional (T4 16 GB = larger batch size possible).

Hardware assessment

  ┌───────────┬──────────────────┬────────────────────────────┬─────────────┐
  │ Component │  What you have   │      What BERT needs       │   Verdict   │
  ├───────────┼──────────────────┼────────────────────────────┼─────────────┤
  │ VRAM      │ 12 GB (RTX 5070) │ 4–6 GB (BERT-base, bs=16)  │ Comfortable │
  ├───────────┼──────────────────┼────────────────────────────┼─────────────┤
  │ RAM       │ 32 GB            │ 8–16 GB                    │ Fine        │
  ├───────────┼──────────────────┼────────────────────────────┼─────────────┤
  │ CPU       │ Ryzen 7 7700X    │ Any modern 8-core          │ Fine        │
  ├───────────┼──────────────────┼────────────────────────────┼─────────────┤
  │ Storage   │ ~1.67 TB free    │ 5–20 GB for models + cache │ Fine        │
  └───────────┴──────────────────┴────────────────────────────┴─────────────┘

  ---
  One important gotcha: RTX 5070 is Blackwell (sm_120)

  This is a brand-new GPU architecture. You need PyTorch ≥ 2.6 with the matching CUDA 12.6+ build — older PyTorch
  versions won't see it as a CUDA device and will silently fall back to CPU. Check first:

Bash(nvidia-smi)
  ⎿  Tue May 12 13:02:48 2026
     +-----------------------------------------------------------------------------------------+
     | NVIDIA-SMI 581.42                 Driver Version: 581.42         CUDA Version: 13.0     |
GPU is confirmed — RTX 5070, 12 GB, CUDA 13.0 driver.

  CUDA drivers are backwards compatible, so install the cu128 PyTorch build (the highest available on the PyTorch
  index). Run this:

  conda run -n sitc pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall
  
  import torch
  print(torch.__version__)          # needs 2.6+
  print(torch.cuda.is_available())  # must be True
  print(torch.cuda.get_device_name(0))  # should say RTX 5070

  conda run -n sitc python -c "import os; os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'; import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA device')"
2.11.0+cu128
True
NVIDIA GeForce RTX 5070



  ---
  Recommended models for your task (small dataset, political text)

  Model: deberta-v3-base
  Size: 86M params, ~340 MB
  Why good here: Best-in-class on classification benchmarks; disentangled attention is great for nuanced political
    language
  ────────────────────────────────────────
  Model: roberta-base
  Size: 125M params, ~500 MB
  Why good here: Robust BERT, strong on short text, well-understood
  ────────────────────────────────────────
  Model: bert-base-uncased
  Size: 110M params, ~440 MB
  Why good here: Classic baseline, extensive literature
  ────────────────────────────────────────
  ────────────────────────────────────────
  Model: distilbert-base-uncased
  Size: 66M params, ~260 MB
  Why good here: Half the size/speed of BERT-base, good for fast iteration
  Model: distilbert-base-uncased
  Size: 66M params, ~260 MB
  Why good here: Half the size/speed of BERT-base, good for fast iteration

  With 8,950 training samples, DeBERTa-v3-base is the best first bet — it's particularly strong on low-to-medium data
  Why good here: Half the size/speed of BERT-base, good for fast iteration

  With 8,950 training samples, DeBERTa-v3-base is the best first bet — it's particularly strong on low-to-medium data
  regimes and consistently outperforms RoBERTa on GLUE/SuperGLUE benchmarks.
  regimes and consistently outperforms RoBERTa on GLUE/SuperGLUE benchmarks.

  ---
  Packages you'll need

  Packages you'll need

  pip install transformers datasets accelerate torch
---

## Kaggle setup

1. Upload your data as a Kaggle Dataset

  - kaggle.com → Datasets → New Dataset
  - Upload data/train.csv and data/test_nolabel.csv
  - Name it exactly truth-classifier-nlp (must match the path in transformer.py)
  - Set to Private

  ---
  2. Create a new Notebook

  - kaggle.com → Code → New Notebook
  - Settings → Accelerator → GPU T4 x2 (free, ~30 hrs/week)
  - Add your dataset: Add Data → search "truth-classifier-nlp" → Add

  ---
  3. Add your W&B API key as a Kaggle Secret

  - kaggle.com → Settings → Connected Services (or Account) → Secrets → Add
  - Name: WANDB_API_KEY, Value: your W&B key
  - In the notebook, expose it before wandb.login():
  from kaggle_secrets import UserSecretsClient
  import os
  os.environ["WANDB_API_KEY"] = UserSecretsClient().get_secret("WANDB_API_KEY")

  ---
  4. First cell of the notebook — installs

  !pip install -q transformers accelerate wandb
  (datasets and torch are already on Kaggle kernels)

  ---
  5. Paste transformer.py

  Copy the full contents of src/training/transformer.py into the next cell. The IS_KAGGLE flag at the top will
  automatically set the right paths — no edits needed.

  ---
  6. One thing to tweak for T4

  In the config section, bump the batch size since T4 has 16 GB vs your 12 GB:
  BATCH_SIZE = 32

  ---
  That's it. The script handles the rest — tokenizing, training, threshold tuning, holdout eval, W&B logging, and saving
   submission-deberta-v3-small-YYYYMMDD-HHMM.csv to /kaggle/working/.



