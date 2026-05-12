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

-----------------------------------------------------------------------


---> Run 1 - 
[SECTION] Training  [13:36:50]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/3 ---  [13:36:50]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6651
    Batch 50/403  avg_loss=0.7040
    Batch 100/403  avg_loss=0.6986
    Batch 150/403  avg_loss=0.6928
    Batch 200/403  avg_loss=0.6883
    Batch 250/403  avg_loss=0.6841
    Batch 300/403  avg_loss=0.6819
    Batch 350/403  avg_loss=0.6767
    Batch 400/403  avg_loss=0.6767
  Train loss: 0.6767  — starting val evaluation
  Val proba range: [0.2699, 0.6893]  NaNs: 0
  Epoch 1/3  train_loss=0.6767  val_loss=0.6539  val_macro_f1=0.6128  val_roc_auc=0.6700  (29.7s)
    New best val macro_f1=0.6128 — checkpoint saved

  --- Epoch 2/3 ---  [13:37:21]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.5945
    Batch 50/403  avg_loss=0.6278
    Batch 100/403  avg_loss=0.6316
    Batch 150/403  avg_loss=0.6273
    Batch 200/403  avg_loss=0.6283
    Batch 250/403  avg_loss=0.6252
    Batch 300/403  avg_loss=0.6254
    Batch 350/403  avg_loss=0.6214
    Batch 400/403  avg_loss=0.6223
  Train loss: 0.6227  — starting val evaluation
  Val proba range: [0.1644, 0.9476]  NaNs: 0
  Epoch 2/3  train_loss=0.6227  val_loss=0.6692  val_macro_f1=0.6005  val_roc_auc=0.6635  (29.4s)

  --- Epoch 3/3 ---  [13:37:50]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6072
    Batch 50/403  avg_loss=0.5313
    Batch 100/403  avg_loss=0.5432
    Batch 150/403  avg_loss=0.5448
    Batch 200/403  avg_loss=0.5393
    Batch 250/403  avg_loss=0.5373
    Batch 300/403  avg_loss=0.5315
    Batch 350/403  avg_loss=0.5299
    Batch 400/403  avg_loss=0.5334
  Train loss: 0.5329  — starting val evaluation
  Val proba range: [0.0957, 0.9819]  NaNs: 0
  Epoch 3/3  train_loss=0.5329  val_loss=0.7601  val_macro_f1=0.5989  val_roc_auc=0.6597  (29.4s)

[SECTION] Loading best checkpoint  [13:38:19]

[SECTION] Threshold tuning on val set  [13:38:20]
   threshold   macro_f1
        0.20   0.3932
        0.21   0.3932
        0.22   0.3932
        0.23   0.3932
        0.24   0.3932
        0.25   0.3932
        0.26   0.3932
        0.27   0.3927
        0.28   0.3964
        0.29   0.3996
        0.30   0.4222
        0.31   0.4359
        0.32   0.4678
        0.33   0.4733
        0.34   0.4839
        0.35   0.5096
        0.36   0.5352
        0.37   0.5459
        0.38   0.5462
        0.39   0.5569
        0.40   0.5823
        0.41   0.5903
        0.42   0.6047
        0.43   0.6081
        0.44   0.6122
        0.45   0.6094
        0.46   0.6100
        0.47   0.6102
        0.48   0.6227  ←
        0.49   0.6165
        0.50   0.6128
        0.51   0.6140
        0.52   0.6157
        0.53   0.6041
        0.54   0.6075
        0.55   0.5989
        0.56   0.5865
        0.57   0.5672
        0.58   0.5655
        0.59   0.5441
        0.60   0.5129
        0.61   0.4782
        0.62   0.4361
        0.63   0.3944
        0.64   0.3536
        0.65   0.3023
        0.66   0.2857
        0.67   0.2692
        0.68   0.2668
        0.69   0.2603
        0.70   0.2603
        0.71   0.2603
        0.72   0.2603
        0.73   0.2603
        0.74   0.2603
        0.75   0.2603
        0.76   0.2603

  Best threshold: 0.48  (val macro_f1=0.6227)

[SECTION] Holdout evaluation  [13:38:20]
  Threshold: 0.48

Holdout results:
  roc_auc: 0.6552
  pr_auc: 0.7566
  macro_f1: 0.6127
  f1: 0.7306
  precision: 0.7253
  recall: 0.7360
  accuracy: 0.6486
  mcc: 0.2255
  balanced_acc: 0.6120

              precision    recall  f1-score   support

           0       0.50      0.49      0.49       631
           1       0.73      0.74      0.73      1159

    accuracy                           0.65      1790
   macro avg       0.61      0.61      0.61      1790
weighted avg       0.65      0.65      0.65      1790

**Script config at time of Run 1**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `microsoft/deberta-v3-small` | 86M params, FP32 |
| MAX_LENGTH | 128 | Covers 99%+ of statements |
| BATCH_SIZE | 16 | Safe for 12 GB VRAM |
| EPOCHS | 3 | Best checkpoint saved by val macro_f1 |
| LR | 2e-5 | Too aggressive — model peaked at epoch 1 |
| WARMUP_RATIO | 0.1 | 10% of total steps |
| WEIGHT_DECAY | 0.01 | — |
| CLASS_WEIGHTS | [1.42, 0.77] | {0: true, 1: false} |
| AMP | Disabled | DeBERTa-v3 unstable in BF16/FP16 |
| Split | 72% train / 8% val / 20% holdout | Val used for checkpoint + threshold tuning |
| Threshold tuning | On val set | Best threshold: 0.48 |
| Optimizer | AdamW | — |
| Scheduler | Linear warmup + decay | — |

**Key observations**
- Overfitting after epoch 1: val loss rose epoch 2→3 while train loss kept falling
- Holdout macro_f1 0.6127 — just below stacking R5 (0.6168)
- ROC-AUC 0.6552 — notably below stacking (0.6835)
- Class 1 recall improved over stacking (0.74 vs 0.64) — different error profile, promising for late fusion

**Next: Run 2** — lower LR to `1e-5`, increase WARMUP_RATIO to `0.2` to spread learning across all 3 epochs.

-------------------------------

---> Run 2 Output
[SECTION] Training  [18:46:08]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/3 ---  [18:46:08]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6651
    Batch 50/403  avg_loss=0.7046
    Batch 100/403  avg_loss=0.6994
    Batch 150/403  avg_loss=0.6981
    Batch 200/403  avg_loss=0.6972
    Batch 250/403  avg_loss=0.6928
    Batch 300/403  avg_loss=0.6899
    Batch 350/403  avg_loss=0.6837
    Batch 400/403  avg_loss=0.6828
  Train loss: 0.6828  — starting val evaluation
  Val proba range: [0.2696, 0.7101]  NaNs: 0
  Epoch 1/3  train_loss=0.6828  val_loss=0.6580  val_macro_f1=0.5993  val_roc_auc=0.6669  (29.9s)
    New best val macro_f1=0.5993 — checkpoint saved

  --- Epoch 2/3 ---  [18:46:38]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6302
    Batch 50/403  avg_loss=0.6369
    Batch 100/403  avg_loss=0.6458
    Batch 150/403  avg_loss=0.6418
    Batch 200/403  avg_loss=0.6436
    Batch 250/403  avg_loss=0.6430
    Batch 300/403  avg_loss=0.6439
    Batch 350/403  avg_loss=0.6405
    Batch 400/403  avg_loss=0.6405
  Train loss: 0.6409  — starting val evaluation
  Val proba range: [0.1770, 0.9000]  NaNs: 0
  Epoch 2/3  train_loss=0.6409  val_loss=0.6611  val_macro_f1=0.5948  val_roc_auc=0.6638  (29.4s)

  --- Epoch 3/3 ---  [18:47:08]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.7002
    Batch 50/403  avg_loss=0.5867
    Batch 100/403  avg_loss=0.5968
    Batch 150/403  avg_loss=0.5975
    Batch 200/403  avg_loss=0.5959
    Batch 250/403  avg_loss=0.5936
    Batch 300/403  avg_loss=0.5901
    Batch 350/403  avg_loss=0.5878
    Batch 400/403  avg_loss=0.5916
  Train loss: 0.5913  — starting val evaluation
  Val proba range: [0.1204, 0.9576]  NaNs: 0
  Epoch 3/3  train_loss=0.5913  val_loss=0.7011  val_macro_f1=0.5973  val_roc_auc=0.6556  (29.4s)

[SECTION] Loading best checkpoint  [18:47:37]

[SECTION] Threshold tuning on val set  [18:47:38]
   threshold   macro_f1
        0.20   0.3932
        0.21   0.3932
        0.22   0.3932
        0.23   0.3932
        0.24   0.3932
        0.25   0.3932
        0.26   0.3932
        0.27   0.3927
        0.28   0.3912
        0.29   0.3954
        0.30   0.4189
        0.31   0.4183
        0.32   0.4295
        0.33   0.4420
        0.34   0.4670
        0.35   0.4795
        0.36   0.4998
        0.37   0.5110
        0.38   0.5146
        0.39   0.5303
        0.40   0.5491
        0.41   0.5683
        0.42   0.5701
        0.43   0.5736
        0.44   0.5837
        0.45   0.5925
        0.46   0.5929
        0.47   0.5944
        0.48   0.6013
        0.49   0.6000
        0.50   0.5993
        0.51   0.6061
        0.52   0.6086
        0.53   0.6091
        0.54   0.6048
        0.55   0.6106
        0.56   0.6193  ←
        0.57   0.6112
        0.58   0.5991
        0.59   0.5921
        0.60   0.5793
        0.61   0.5556
        0.62   0.5057
        0.63   0.4623
        0.64   0.4047
        0.65   0.3719
        0.66   0.3283
        0.67   0.2870
        0.68   0.2810
        0.69   0.2739
        0.70   0.2628
        0.71   0.2628
        0.72   0.2603
        0.73   0.2603
        0.74   0.2603
        0.75   0.2603
        0.76   0.2603

  Best threshold: 0.56  (val macro_f1=0.6193)

[SECTION] Holdout evaluation  [18:47:38]
  Threshold: 0.56

Holdout results:
  roc_auc: 0.6508
  pr_auc: 0.7510
  macro_f1: 0.6039
  f1: 0.6840
  precision: 0.7412
  recall: 0.6350
  accuracy: 0.6201
  mcc: 0.2189
  balanced_acc: 0.6139

              precision    recall  f1-score   support

           0       0.47      0.59      0.52       631
           1       0.74      0.64      0.68      1159

    accuracy                           0.62      1790
   macro avg       0.61      0.61      0.60      1790
weighted avg       0.65      0.62      0.63      1790


**Script config at time of Run 2**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `microsoft/deberta-v3-small` | 86M params, FP32 |
| EPOCHS | 3 | Best checkpoint saved by val macro_f1 |
| LR | 1e-5 | Half of Run 1 |
| WARMUP_RATIO | 0.2 | 20% of total steps |
| WEIGHT_DECAY | 0.01 | — |
| LABEL_SMOOTHING | none | — |
| Scheduler | Linear warmup + decay | — |
| Threshold tuning | On val set | Best threshold: 0.56 |

**Key observations — Run 2 was a regression**

| Metric | Run 1 (2e-5, wu=0.1) | Run 2 (1e-5, wu=0.2) |
|--------|----------------------|----------------------|
| Best checkpoint epoch | 1 | 1 |
| Best val macro_f1 (ep1) | 0.6122 | 0.5993 |
| Tuned val macro_f1 | 0.6227 (thr=0.48) | 0.6193 (thr=0.56) |
| Holdout macro_f1 | 0.6127 | 0.6039 |
| Holdout ROC-AUC | 0.6552 | 0.6508 |

- Epoch 1 still the best checkpoint in Run 2, but with even less signal than Run 1
- Lower LR extracted *less* signal in epoch 1 (val F1: 0.6122 → 0.5993)
- Val_loss at epoch 3 jumped to 0.7011 — more severe collapse than Run 1
- Proba range blew out to [0.12, 0.96] by epoch 3 — overconfident collapse
- Threshold shifted to 0.56 vs 0.48 — model skewed harder toward predicting false
- Root cause: ~6,400 training samples is too small for 86M params; the model saturates useful learning in one pass — LR tuning alone does not fix this

**Next: Run 3** — one-epoch approach: `EPOCHS=1`, `LR=3e-5`, cosine schedule, `label_smoothing=0.1`. Extract max signal from one clean pass; skip overfitting entirely.

--> Run 3 One epoch approach output

[SECTION] Training  [19:12:58]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/1 ---  [19:12:58]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6684
    Batch 50/403  avg_loss=0.7093
    Batch 100/403  avg_loss=0.7017
    Batch 150/403  avg_loss=0.6999
    Batch 200/403  avg_loss=0.6950
    Batch 250/403  avg_loss=0.6914
    Batch 300/403  avg_loss=0.6897
    Batch 350/403  avg_loss=0.6864
    Batch 400/403  avg_loss=0.6870
  Train loss: 0.6872  — starting val evaluation
  Val proba range: [0.2577, 0.7147]  NaNs: 0
  Epoch 1/1  train_loss=0.6872  val_loss=0.6716  val_macro_f1=0.5993  val_roc_auc=0.6509  (30.0s)
    New best val macro_f1=0.5993 — checkpoint saved

[SECTION] Loading best checkpoint  [19:13:29]

[SECTION] Threshold tuning on val set  [19:13:29]
   threshold   macro_f1
        0.20   0.3932
        0.21   0.3932
        0.22   0.3932
        0.23   0.3932
        0.24   0.3932
        0.25   0.3932
        0.26   0.3970
        0.27   0.3959
        0.28   0.4167
        0.29   0.4411
        0.30   0.4725
        0.31   0.4980
        0.32   0.5192
        0.33   0.5544
        0.34   0.5768
        0.35   0.5848
        0.36   0.6020  ←
        0.37   0.5989
        0.38   0.5979
        0.39   0.5939
        0.40   0.5969
        0.41   0.5941
        0.42   0.5928
        0.43   0.5932
        0.44   0.5888
        0.45   0.5921
        0.46   0.5997
        0.47   0.5965
        0.48   0.5966
        0.49   0.5963
        0.50   0.5993
        0.51   0.5905
        0.52   0.5913
        0.53   0.5913
        0.54   0.5909
        0.55   0.5878
        0.56   0.5804
        0.57   0.5730
        0.58   0.5712
        0.59   0.5664
        0.60   0.5599
        0.61   0.5419
        0.62   0.5332
        0.63   0.5227
        0.64   0.5118
        0.65   0.4896
        0.66   0.4590
        0.67   0.4304
        0.68   0.3796
        0.69   0.3264
        0.70   0.2848
        0.71   0.2652
        0.72   0.2603
        0.73   0.2603
        0.74   0.2603
        0.75   0.2603
        0.76   0.2603

  Best threshold: 0.36  (val macro_f1=0.6020)

[SECTION] Holdout evaluation  [19:13:29]
  Threshold: 0.36

Holdout results:
  roc_auc: 0.6416
  pr_auc: 0.7406
  macro_f1: 0.5851
  f1: 0.7631
  precision: 0.6976
  recall: 0.8421
  accuracy: 0.6615
  mcc: 0.1986
  balanced_acc: 0.5859

              precision    recall  f1-score   support

           0       0.53      0.33      0.41       631
           1       0.70      0.84      0.76      1159

    accuracy                           0.66      1790
   macro avg       0.61      0.59      0.59      1790
weighted avg       0.64      0.66      0.64      1790

**Script config at time of Run 3**

| Parameter | Value | Notes |
|-----------|-------|-------|
| EPOCHS | 1 | One-epoch approach |
| LR | 3e-5 | Higher than Run 1 |
| WARMUP_RATIO | 0.1 | — |
| LABEL_SMOOTHING | 0.1 | New — turned out to conflict with class weights |
| Scheduler | Cosine | New |
| Threshold tuning | On val set | Best threshold: 0.36 |

**Key observations — Run 3 was the worst of all three**

| Metric | Run 1 (2e-5, 3ep) | Run 2 (1e-5, 3ep) | Run 3 (3e-5, 1ep, cosine, ls) |
|--------|-------------------|-------------------|-------------------------------|
| Val macro_f1 (thr=0.5) | **0.6122** | 0.5993 | 0.5993 |
| Tuned val macro_f1 | **0.6227** (thr=0.48) | 0.6193 (thr=0.56) | 0.6020 (thr=0.36) |
| Holdout macro_f1 | **0.6127** | 0.6039 | 0.5851 |
| Holdout ROC-AUC | **0.6552** | 0.6508 | 0.6416 |
| Class 0 recall | 0.49 | 0.59 | 0.33 |

- **Label smoothing + class weights conflicted**: opposing forces on the loss surface; model compensated by skewing probabilities low → threshold dropped to 0.36, class 0 recall collapsed to 0.33
- **Generalization gap widening**: −0.010 Run 1, −0.015 Run 2, −0.017 Run 3 — threshold tuning overfitting to the small 716-sample val set
- **2e-5 was the validated sweet spot**: Run 1 epoch 1 (0.6122) was clearly better than both 1e-5 and 3e-5
- **Root cause**: the backbone updates too aggressively at any uniform LR for 3 epochs; LR/schedule tuning alone can't fix this

**Next: Run 4** — Layer-wise LR decay (LLRD): head gets `LR=2e-5`, each encoder layer multiplied by `0.9` going down (layer 11 ≈ 1.8e-5, layer 0 ≈ 0.43e-5, embeddings ≈ 0.39e-5). No label smoothing. Linear warmup. `EPOCHS=3`.


--------------------------------------

--> Output Run 4** — Layer-wise LR decay (LLRD): head gets `LR=2e-5`, each encoder layer multiplied by `0.9` going down (layer 11 ≈ 1.8e-5, layer 0 ≈ 0.43e-5, embeddings ≈ 0.39e-5). No label smoothing. Linear warmup. `EPOCHS=3`.

[SECTION] Training  [19:31:23]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/3 ---  [19:31:23]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6651
    Batch 50/403  avg_loss=0.7041
    Batch 100/403  avg_loss=0.6987
    Batch 150/403  avg_loss=0.6954
    Batch 200/403  avg_loss=0.6887
    Batch 250/403  avg_loss=0.6833
    Batch 300/403  avg_loss=0.6811
    Batch 350/403  avg_loss=0.6767
    Batch 400/403  avg_loss=0.6765
  Train loss: 0.6766  — starting val evaluation
  Val proba range: [0.2764, 0.6933]  NaNs: 0
  Epoch 1/3  train_loss=0.6766  val_loss=0.6580  val_macro_f1=0.6145  val_roc_auc=0.6624  (29.1s)
    New best val macro_f1=0.6145 — checkpoint saved

  --- Epoch 2/3 ---  [19:31:53]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6160
    Batch 50/403  avg_loss=0.6300
    Batch 100/403  avg_loss=0.6375
    Batch 150/403  avg_loss=0.6351
    Batch 200/403  avg_loss=0.6374
    Batch 250/403  avg_loss=0.6356
    Batch 300/403  avg_loss=0.6357
    Batch 350/403  avg_loss=0.6324
    Batch 400/403  avg_loss=0.6329
  Train loss: 0.6334  — starting val evaluation
  Val proba range: [0.1551, 0.9043]  NaNs: 0
  Epoch 2/3  train_loss=0.6334  val_loss=0.6637  val_macro_f1=0.5931  val_roc_auc=0.6632  (28.8s)

  --- Epoch 3/3 ---  [19:32:21]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6594
    Batch 50/403  avg_loss=0.5685
    Batch 100/403  avg_loss=0.5801
    Batch 150/403  avg_loss=0.5803
    Batch 200/403  avg_loss=0.5794
    Batch 250/403  avg_loss=0.5772
    Batch 300/403  avg_loss=0.5733
    Batch 350/403  avg_loss=0.5716
    Batch 400/403  avg_loss=0.5758
  Train loss: 0.5754  — starting val evaluation
  Val proba range: [0.1025, 0.9738]  NaNs: 0
  Epoch 3/3  train_loss=0.5754  val_loss=0.7104  val_macro_f1=0.5948  val_roc_auc=0.6593  (28.8s)

[SECTION] Loading best checkpoint  [19:32:50]

[SECTION] Threshold tuning on val set  [19:32:51]
   threshold   macro_f1
        0.20   0.3932
        0.21   0.3932
        0.22   0.3932
        0.23   0.3932
        0.24   0.3932
        0.25   0.3932
        0.26   0.3932
        0.27   0.3932
        0.28   0.3927
        0.29   0.4012
        0.30   0.4026
        0.31   0.4136
        0.32   0.4320
        0.33   0.4488
        0.34   0.4646
        0.35   0.4839
        0.36   0.5013
        0.37   0.5221
        0.38   0.5383
        0.39   0.5470
        0.40   0.5597
        0.41   0.5696
        0.42   0.5773
        0.43   0.5935
        0.44   0.5857
        0.45   0.6013
        0.46   0.6045
        0.47   0.6083
        0.48   0.6075
        0.49   0.6099
        0.50   0.6145  ←
        0.51   0.6106
        0.52   0.6102
        0.53   0.6016
        0.54   0.6032
        0.55   0.6050
        0.56   0.5922
        0.57   0.5845
        0.58   0.5667
        0.59   0.5499
        0.60   0.5488
        0.61   0.5374
        0.62   0.4855
        0.63   0.4448
        0.64   0.3986
        0.65   0.3528
        0.66   0.3023
        0.67   0.2818
        0.68   0.2700
        0.69   0.2628
        0.70   0.2603
        0.71   0.2603
        0.72   0.2603
        0.73   0.2603
        0.74   0.2603
        0.75   0.2603
        0.76   0.2603

  Best threshold: 0.50  (val macro_f1=0.6145)

[SECTION] Holdout evaluation  [19:32:51]
  Threshold: 0.50

Holdout results:
  roc_auc: 0.6538
  pr_auc: 0.7537
  macro_f1: 0.6133
  f1: 0.7319
  precision: 0.7254
  recall: 0.7386
  accuracy: 0.6497
  mcc: 0.2269
  balanced_acc: 0.6125

              precision    recall  f1-score   support

           0       0.50      0.49      0.49       631
           1       0.73      0.74      0.73      1159

    accuracy                           0.65      1790
   macro avg       0.61      0.61      0.61      1790
weighted avg       0.65      0.65      0.65      1790

**Script config at time of Run 4**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `microsoft/deberta-v3-small` | 86M params, FP32 |
| EPOCHS | 3 | Best checkpoint saved by val macro_f1 |
| LR | 2e-5 | Same as Run 1 |
| LLRD_FACTOR | 0.9 | New — head=2e-5, layer 11≈1.8e-5, layer 0≈0.43e-5, embeddings≈0.39e-5 |
| WARMUP_RATIO | 0.1 | — |
| WEIGHT_DECAY | 0.01 | — |
| Scheduler | Linear warmup + decay | — |
| Threshold tuning | On val set | Best threshold: 0.50 (no tuning benefit — natural threshold won) |

**Key observations — Run 4 confirmed LLRD helps marginally; epoch 1 pattern unchanged**

| Metric | R1 (2e-5) | R2 (1e-5) | R3 (3e-5, 1ep) | R4 (+LLRD) |
|--------|-----------|-----------|-----------------|------------|
| Ep1 val_macro_f1 | 0.6122 | 0.5993 | 0.5993 | **0.6145** |
| Tuned val macro_f1 | 0.6227 | 0.6193 | 0.6020 | 0.6145 |
| Holdout macro_f1 | 0.6127 | 0.6039 | 0.5851 | **0.6133** |
| Holdout ROC-AUC | 0.6552 | 0.6508 | 0.6416 | 0.6538 |
| Best epoch | 1 | 1 | 1 | 1 |

- LLRD gave a marginal but consistent gain: epoch 1 val F1 0.6122 → 0.6145, holdout 0.6127 → 0.6133
- Threshold collapsed to 0.50 — first time threshold tuning gave zero benefit; probabilities better calibrated by LLRD
- Val loss trajectory: epoch 2 (0.6637) → epoch 3 (0.7104) — same overfitting curve as all prior runs
- Epoch 1 is the best checkpoint across all 4 runs without exception — LR, warmup, schedule, and LLRD cannot fix this
- Root cause confirmed: classifier head and backbone fine-tune simultaneously from a random head; they destabilize each other after one pass, and the model memorizes noise from epoch 2 onward

**Next: Run 5 — Freeze-then-unfreeze (gradual unfreezing)**

**Changes from Run 4:**
- `FREEZE_EPOCHS = 1` — DeBERTa backbone frozen for epoch 1; only the ~1,500-param classifier head trains
- Phase 1 optimizer: classifier params only, flat `LR=2e-5`, linear warmup over epoch 1 steps only
- Phase 2 optimizer: rebuilt fresh at epoch 2 with LLRD across all layers, with its own linear warmup over the remaining 2-epoch steps
- All other config unchanged: LR=2e-5, LLRD_FACTOR=0.9, WARMUP_RATIO=0.1, EPOCHS=3

**Decision rationale:** Every run peaks at epoch 1 because the random classifier head destabilizes the backbone from the first step. Freezing the backbone in epoch 1 lets the head converge on top of frozen pretrained DeBERTa features. When the backbone unfreezes at epoch 2, it fine-tunes starting from a stable, task-adapted head — giving epochs 2–3 a real signal to follow instead of noise from a still-random head.

**Expected behavior:**
- Epoch 1 val F1 will be lower than R4's 0.6145 (classifier-only learning is weaker than full fine-tuning in the first pass)
- Epoch 2 should show genuine improvement for the first time — backbone fine-tunes on top of a warm head
- Best checkpoint should shift to epoch 2 or 3 rather than epoch 1
- Val loss should not spike at epoch 2 — the transition from frozen to unfrozen should be smoother

-----------------------------------------------------

--> Run5 Output Run 5 — Freeze-then-unfreeze (gradual unfreezing)
[SECTION] Tokenizing  [20:00:38]
  Tokenized in 0.7s

[SECTION] Loading model: microsoft/deberta-v3-small  [20:00:40]
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 5638.62it/s]
[transformers] DebertaV2ForSequenceClassification LOAD REPORT from: microsoft/deberta-v3-small
Key                                     | Status     | 
----------------------------------------+------------+-
mask_predictions.LayerNorm.bias         | UNEXPECTED | 
lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED | 
mask_predictions.classifier.bias        | UNEXPECTED | 
lm_predictions.lm_head.dense.bias       | UNEXPECTED | 
mask_predictions.dense.weight           | UNEXPECTED | 
lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED | 
mask_predictions.classifier.weight      | UNEXPECTED | 
mask_predictions.dense.bias             | UNEXPECTED | 
mask_predictions.LayerNorm.weight       | UNEXPECTED | 
lm_predictions.lm_head.dense.weight     | UNEXPECTED | 
lm_predictions.lm_head.bias             | UNEXPECTED | 
pooler.dense.bias                       | MISSING    | 
classifier.bias                         | MISSING    | 
classifier.weight                       | MISSING    | 
pooler.dense.weight                     | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
  Parameters: 141,896,450
  Backbone frozen — frozen=141,304,320  trainable=592,130
  Phase 1 optimizer — 4 param tensors  lr=2.0e-05


[SECTION] Training  [20:00:45]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/3 ---  [20:00:45]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6651
    Batch 50/403  avg_loss=0.7038
    Batch 100/403  avg_loss=0.6990
    Batch 150/403  avg_loss=0.6981
    Batch 200/403  avg_loss=0.6977
    Batch 250/403  avg_loss=0.6966
    Batch 300/403  avg_loss=0.6965
    Batch 350/403  avg_loss=0.6960
    Batch 400/403  avg_loss=0.6957
  Train loss: 0.6957  — starting val evaluation
  Val proba range: [0.5020, 0.5138]  NaNs: 0
  Epoch 1/3  train_loss=0.6957  val_loss=0.6927  val_macro_f1=0.3932  val_roc_auc=0.5709  (9.1s)
    New best val macro_f1=0.3932 — checkpoint saved

  [Phase 2] Unfreezing backbone + rebuilding optimizer with LLRD
  Backbone unfrozen — trainable=141,896,450
  LLRD groups: 16  LR range: [0.00e+00, 0.00e+00]

  --- Epoch 2/3 ---  [20:00:55]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.7023
    Batch 50/403  avg_loss=0.6955
    Batch 100/403  avg_loss=0.6952
    Batch 150/403  avg_loss=0.6904
    Batch 200/403  avg_loss=0.6880
    Batch 250/403  avg_loss=0.6845
    Batch 300/403  avg_loss=0.6823
    Batch 350/403  avg_loss=0.6776
    Batch 400/403  avg_loss=0.6766
  Train loss: 0.6771  — starting val evaluation
  Val proba range: [0.2213, 0.7682]  NaNs: 0
  Epoch 2/3  train_loss=0.6771  val_loss=0.6530  val_macro_f1=0.5961  val_roc_auc=0.6618  (28.8s)
    New best val macro_f1=0.5961 — checkpoint saved

  --- Epoch 3/3 ---  [20:01:24]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.7680
    Batch 50/403  avg_loss=0.6249
    Batch 100/403  avg_loss=0.6328
    Batch 150/403  avg_loss=0.6327
    Batch 200/403  avg_loss=0.6325
    Batch 250/403  avg_loss=0.6307
    Batch 300/403  avg_loss=0.6284
    Batch 350/403  avg_loss=0.6272
    Batch 400/403  avg_loss=0.6295
  Train loss: 0.6294  — starting val evaluation
  Val proba range: [0.1456, 0.9374]  NaNs: 0
  Epoch 3/3  train_loss=0.6294  val_loss=0.6622  val_macro_f1=0.6129  val_roc_auc=0.6696  (28.8s)
    New best val macro_f1=0.6129 — checkpoint saved

[SECTION] Loading best checkpoint  [20:01:53]

[SECTION] Threshold tuning on val set  [20:01:54]
   threshold   macro_f1
        0.20   0.4465
        0.21   0.4784
        0.22   0.4754
        0.23   0.5069
        0.24   0.5301
        0.25   0.5413
        0.26   0.5577
        0.27   0.5670
        0.28   0.5760
        0.29   0.5838
        0.30   0.5936
        0.31   0.5853
        0.32   0.5883
        0.33   0.6024
        0.34   0.6013
        0.35   0.6135
        0.36   0.6107
        0.37   0.6130
        0.38   0.6165  ←
        0.39   0.6159
        0.40   0.6124
        0.41   0.6102
        0.42   0.6165
        0.43   0.6156
        0.44   0.6097
        0.45   0.6137
        0.46   0.6080
        0.47   0.6088
        0.48   0.6058
        0.49   0.6074
        0.50   0.6129
        0.51   0.6090
        0.52   0.6055
        0.53   0.5940
        0.54   0.5929
        0.55   0.5929
        0.56   0.5932
        0.57   0.5927
        0.58   0.5952
        0.59   0.5979
        0.60   0.5956
        0.61   0.5825
        0.62   0.5789
        0.63   0.5724
        0.64   0.5697
        0.65   0.5614
        0.66   0.5586
        0.67   0.5513
        0.68   0.5424
        0.69   0.5350
        0.70   0.5196
        0.71   0.5111
        0.72   0.5022
        0.73   0.4847
        0.74   0.4730
        0.75   0.4576
        0.76   0.4448

  Best threshold: 0.38  (val macro_f1=0.6165)

[SECTION] Holdout evaluation  [20:01:54]
  Threshold: 0.38

Holdout results:
  roc_auc: 0.6679
  pr_auc: 0.7699
  macro_f1: 0.6203
  f1: 0.7535
  precision: 0.7236
  recall: 0.7860
  accuracy: 0.6670
  mcc: 0.2453
  balanced_acc: 0.6173

              precision    recall  f1-score   support

           0       0.53      0.45      0.49       631
           1       0.72      0.79      0.75      1159

    accuracy                           0.67      1790
   macro avg       0.63      0.62      0.62      1790
weighted avg       0.66      0.67      0.66      1790

**Script config at time of Run 5**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `microsoft/deberta-v3-small` | 86M params, FP32 |
| EPOCHS | 3 | — |
| FREEZE_EPOCHS | 1 | New — backbone frozen for epoch 1 |
| LR | 2e-5 | Phase 1 (head) and Phase 2 (head, LLRD base) |
| LLRD_FACTOR | 0.9 | Applied only in Phase 2 |
| WARMUP_RATIO | 0.1 | Applied independently per phase |
| Scheduler | Linear warmup + decay (×2) | Phase 1: 40-step warmup / 403 steps; Phase 2: 80-step warmup / 806 steps |
| Threshold tuning | On val set | Best threshold: 0.38 |

**Known display bug (not a training bug):** `LLRD groups: 16  LR range: [0.00e+00, 0.00e+00]`
`get_linear_schedule_with_warmup` calls `step()` during init, which sets all `optimizer.param_groups[i]['lr']` to `base_lr × 0 = 0` (warmup step 0). The print reads from the already-stepped param groups. Actual training ran correctly — train loss descended normally from 0.6957 → 0.6771 → 0.6294. Fix in Run 6: capture the base LR from LLRD groups before scheduler construction.

**Key observations — freeze-then-unfreeze hypothesis confirmed; new best holdout**

| Metric | R1 | R2 | R3 | R4 (+LLRD) | R5 (freeze) |
|--------|-----|-----|-----|------------|-------------|
| Best epoch | 1 | 1 | 1 | 1 | **3** |
| Best val macro_f1 | 0.6128 | 0.5993 | 0.5993 | 0.6145 | 0.6129 |
| Tuned val macro_f1 | 0.6227 | 0.6193 | 0.6020 | 0.6145 | 0.6165 |
| Holdout macro_f1 | 0.6127 | 0.6039 | 0.5851 | 0.6133 | **0.6203** |
| Holdout ROC-AUC | 0.6552 | 0.6508 | 0.6416 | 0.6538 | **0.6679** |

- **Best checkpoint is epoch 3 for the first time** — multi-epoch fine-tuning now provides a real benefit
- Val F1 trend reversed: 0.3932 (ep1, frozen) → 0.5961 (ep2) → **0.6129 (ep3)** — ascending across all unfrozen epochs
- Holdout macro_f1 improved from 0.6133 → **0.6203** (+0.0070 over R4)
- Holdout ROC-AUC jumped from 0.6538 → **0.6679** (+0.0141) — strongest single-run AUC gain so far
- Val loss at epoch 3 (0.6622) is slightly above epoch 2 (0.6530) — there's a mild overfitting signal on loss, but F1 still improved, meaning the threshold adjustment captured real signal improvement
- The val F1 trend at epoch 3 (0.6129) was still rising — we stopped too early; epoch 4 may have continued to improve
- Threshold dropped to 0.38 — model still slightly biased toward predicting false; threshold tuning remains necessary

**Next: Run 6 — More epochs (EPOCHS=5, FREEZE_EPOCHS=1)**

**Changes from Run 5:**
- `EPOCHS = 5` (was 3) — backbone now gets 4 full fine-tuning epochs instead of 2
- `FREEZE_EPOCHS = 1` unchanged — epoch 1 still freezes backbone for head warmup
- Fix LR display bug: capture base LR range from LLRD param groups before scheduler construction
- All other config unchanged

**Decision rationale:** Run 5 showed val F1 still rising at epoch 3 with no sign of collapse — the freeze-then-unfreeze approach changed the training dynamics enough that more epochs are now beneficial. With FREEZE_EPOCHS=1, the backbone gets `(EPOCHS - 1)` fine-tuning epochs. Going from 2 to 4 backbone epochs lets us find where val F1 actually peaks under this regime rather than cutting off early.

**Expected behavior:**
- Epoch 1 will be identical to R5 (same frozen head training)
- Epochs 2–4 should continue the ascending val F1 trend seen in R5
- Epoch 5 is the "will it collapse?" test — val loss may rise while F1 continues or plateaus
- Best checkpoint will likely be epoch 3 or 4; if still improving at epoch 5, R7 would extend further
- Total runtime ~2× longer than R5 (~75–80 seconds for 5 epochs of backbone fine-tuning)

--------------------------------------

--> Run 6 Output - Run 6 — More epochs (EPOCHS=5, FREEZE_EPOCHS=1)

[SECTION] Loading data  [20:25:50]
  Rows: 8,950  |  Labels: {1: 5795, 0: 3155}

[SECTION] Dataset features
  Used       : ['statement']  (raw text, tokenized)
  Not used   : ['subject', 'speaker', 'speaker_job', 'state_info', 'party_affiliation']
  Token len  : min=2  median=17  p99=41  max=467
  subject               : 3409 unique  top-3=['health-care', 'taxes', 'elections']
  speaker               : 2634 unique  top-3=['barack-obama', 'donald-trump', 'hillary-clinton']
  speaker_job           : 1090 unique  top-3=['President', 'U.S. Senator', 'Governor']
  state_info            :   78 unique  top-3=['Texas', 'Florida', 'Wisconsin']
  party_affiliation     :   24 unique  top-3=['republican', 'democrat', 'none']
  Train: 6,444   Val: 716   Holdout: 1,790

[SECTION] Tokenizing  [20:25:50]
  Tokenized in 0.7s

[SECTION] Loading model: microsoft/deberta-v3-small  [20:25:52]
[transformers] `torch_dtype` is deprecated! Use `dtype` instead!
Loading weights: 100%|████████████████████████████████████████████████████████████| 102/102 [00:00<00:00, 6147.88it/s]
[transformers] DebertaV2ForSequenceClassification LOAD REPORT from: microsoft/deberta-v3-small
Key                                     | Status     | 
----------------------------------------+------------+-
lm_predictions.lm_head.LayerNorm.weight | UNEXPECTED | 
lm_predictions.lm_head.LayerNorm.bias   | UNEXPECTED | 
mask_predictions.classifier.weight      | UNEXPECTED | 
lm_predictions.lm_head.bias             | UNEXPECTED | 
mask_predictions.LayerNorm.weight       | UNEXPECTED | 
mask_predictions.dense.weight           | UNEXPECTED | 
mask_predictions.classifier.bias        | UNEXPECTED | 
mask_predictions.LayerNorm.bias         | UNEXPECTED | 
lm_predictions.lm_head.dense.bias       | UNEXPECTED | 
mask_predictions.dense.bias             | UNEXPECTED | 
lm_predictions.lm_head.dense.weight     | UNEXPECTED | 
pooler.dense.bias                       | MISSING    | 
pooler.dense.weight                     | MISSING    | 
classifier.bias                         | MISSING    | 
classifier.weight                       | MISSING    | 

Notes:
- UNEXPECTED:   can be ignored when loading from different task/architecture; not ok if you expect identical arch.
- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.
  Parameters: 141,896,450
  Backbone frozen — frozen=141,304,320  trainable=592,130
  Phase 1 optimizer — 4 param tensors  lr=2.0e-05

[SECTION] Training  [20:25:56]
  Model dtype     : torch.float32
  loss_weights    : tensor([1.4200, 0.7700], device='cuda:0')
  Train batches   : 403  Val batches: 23

  --- Epoch 1/5 ---  [20:25:56]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6651
    Batch 50/403  avg_loss=0.7038
    Batch 100/403  avg_loss=0.6990
    Batch 150/403  avg_loss=0.6981
    Batch 200/403  avg_loss=0.6977
    Batch 250/403  avg_loss=0.6966
    Batch 300/403  avg_loss=0.6965
    Batch 350/403  avg_loss=0.6960
    Batch 400/403  avg_loss=0.6957
  Train loss: 0.6957  — starting val evaluation
  Val proba range: [0.5020, 0.5138]  NaNs: 0
  Epoch 1/5  train_loss=0.6957  val_loss=0.6927  val_macro_f1=0.3932  val_roc_auc=0.5709  (9.1s)
    New best val macro_f1=0.3932 — checkpoint saved

  [Phase 2] Unfreezing backbone + rebuilding optimizer with LLRD
  Backbone unfrozen — trainable=141,896,450
  LLRD groups: 16  LR range: [9.57e-06, 2.00e-05]

  --- Epoch 2/5 ---  [20:26:06]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.7023
    Batch 50/403  avg_loss=0.6955
    Batch 100/403  avg_loss=0.6956
    Batch 150/403  avg_loss=0.6957
    Batch 200/403  avg_loss=0.6938
    Batch 250/403  avg_loss=0.6898
    Batch 300/403  avg_loss=0.6866
    Batch 350/403  avg_loss=0.6817
    Batch 400/403  avg_loss=0.6810
  Train loss: 0.6815  — starting val evaluation
  Val proba range: [0.2088, 0.7831]  NaNs: 0
  Epoch 2/5  train_loss=0.6815  val_loss=0.6525  val_macro_f1=0.5945  val_roc_auc=0.6651  (28.9s)
    New best val macro_f1=0.5945 — checkpoint saved

  --- Epoch 3/5 ---  [20:26:35]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.7886
    Batch 50/403  avg_loss=0.6206
    Batch 100/403  avg_loss=0.6293
    Batch 150/403  avg_loss=0.6307
    Batch 200/403  avg_loss=0.6296
    Batch 250/403  avg_loss=0.6278
    Batch 300/403  avg_loss=0.6249
    Batch 350/403  avg_loss=0.6243
    Batch 400/403  avg_loss=0.6269
  Train loss: 0.6269  — starting val evaluation
  Val proba range: [0.1172, 0.9636]  NaNs: 0
  Epoch 3/5  train_loss=0.6269  val_loss=0.6557  val_macro_f1=0.6002  val_roc_auc=0.6709  (28.8s)
    New best val macro_f1=0.6002 — checkpoint saved

  --- Epoch 4/5 ---  [20:27:05]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.6381
    Batch 50/403  avg_loss=0.5802
    Batch 100/403  avg_loss=0.5611
    Batch 150/403  avg_loss=0.5507
    Batch 200/403  avg_loss=0.5459
    Batch 250/403  avg_loss=0.5444
    Batch 300/403  avg_loss=0.5463
    Batch 350/403  avg_loss=0.5476
    Batch 400/403  avg_loss=0.5446
  Train loss: 0.5433  — starting val evaluation
  Val proba range: [0.0396, 0.9930]  NaNs: 0
  Epoch 4/5  train_loss=0.5433  val_loss=0.7386  val_macro_f1=0.5793  val_roc_auc=0.6624  (28.9s)

  --- Epoch 5/5 ---  [20:27:33]
    Batch 0 — logits dtype=torch.float32  labs dtype=torch.int64  loss=0.4235
    Batch 50/403  avg_loss=0.4750
    Batch 100/403  avg_loss=0.4660
    Batch 150/403  avg_loss=0.4652
    Batch 200/403  avg_loss=0.4624
    Batch 250/403  avg_loss=0.4649
    Batch 300/403  avg_loss=0.4634
    Batch 350/403  avg_loss=0.4672
    Batch 400/403  avg_loss=0.4637
  Train loss: 0.4631  — starting val evaluation
  Val proba range: [0.0246, 0.9972]  NaNs: 0
  Epoch 5/5  train_loss=0.4631  val_loss=0.8224  val_macro_f1=0.5942  val_roc_auc=0.6501  (28.9s)

[SECTION] Loading best checkpoint  [20:28:02]

[SECTION] Threshold tuning on val set  [20:28:03]
   threshold   macro_f1
        0.20   0.4590
        0.21   0.4697
        0.22   0.4821
        0.23   0.4989
        0.24   0.5110
        0.25   0.5251
        0.26   0.5403
        0.27   0.5554
        0.28   0.5610
        0.29   0.5690
        0.30   0.5758
        0.31   0.5806
        0.32   0.5886
        0.33   0.6010
        0.34   0.6047
        0.35   0.5989
        0.36   0.6113  ←
        0.37   0.6067
        0.38   0.6061
        0.39   0.6113
        0.40   0.5990
        0.41   0.5952
        0.42   0.6001
        0.43   0.6038
        0.44   0.6094
        0.45   0.5998
        0.46   0.6095
        0.47   0.6078
        0.48   0.6100
        0.49   0.6059
        0.50   0.6002
        0.51   0.6021
        0.52   0.5991
        0.53   0.5926
        0.54   0.5934
        0.55   0.5887
        0.56   0.5714
        0.57   0.5675
        0.58   0.5652
        0.59   0.5558
        0.60   0.5475
        0.61   0.5472
        0.62   0.5397
        0.63   0.5337
        0.64   0.5301
        0.65   0.5280
        0.66   0.5139
        0.67   0.5056
        0.68   0.4980
        0.69   0.4925
        0.70   0.4856
        0.71   0.4760
        0.72   0.4564
        0.73   0.4470
        0.74   0.4393
        0.75   0.4296
        0.76   0.4126

  Best threshold: 0.36  (val macro_f1=0.6113)

[SECTION] Holdout evaluation  [20:28:03]
  Threshold: 0.36

Holdout results:
  roc_auc: 0.6757
  pr_auc: 0.7825
  macro_f1: 0.6126
  f1: 0.7509
  precision: 0.7181
  recall: 0.7869
  accuracy: 0.6620
  mcc: 0.2310
  balanced_acc: 0.6098

              precision    recall  f1-score   support

           0       0.53      0.43      0.47       631
           1       0.72      0.79      0.75      1159

    accuracy                           0.66      1790
   macro avg       0.62      0.61      0.61      1790
weighted avg       0.65      0.66      0.65      1790

**Script config at time of Run 6**

| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | `microsoft/deberta-v3-small` | 86M params, FP32 |
| EPOCHS | 5 | — |
| FREEZE_EPOCHS | 1 | Backbone frozen for epoch 1 |
| LR | 2e-5 | Phase 2 LLRD base |
| LLRD_FACTOR | 0.9 | — |
| WARMUP_RATIO | 0.1 | Phase 2 warmup: 161 steps over 1,612 total |
| Threshold tuning | On val set | Best threshold: 0.36 |

**Key observations — epoch 3 is the confirmed peak; R5 schedule beats R6 at epoch 3**

| Metric | R4 (+LLRD) | R5 (3ep, freeze) | R6 (5ep, freeze) |
|--------|------------|-----------------|-----------------|
| Best epoch | 1 | **3** | **3** |
| Best val macro_f1 | 0.6145 | 0.6129 | 0.6002 |
| Tuned val macro_f1 | 0.6145 | 0.6165 | 0.6113 |
| Holdout macro_f1 | 0.6133 | **0.6203** | 0.6126 |
| Holdout ROC-AUC | 0.6538 | 0.6679 | **0.6757** |

- **Best checkpoint is epoch 3 in both R5 and R6** — 1 frozen epoch + 2 fine-tuning epochs is the confirmed optimum
- **R6 epoch 3 is weaker than R5 epoch 3** (val F1: 0.6002 vs 0.6129) despite identical frozen epoch and identical training batches — cause is the LR schedule: R5's phase-2 schedule covers 806 steps (2 epochs) with warmup=80; R6's covers 1,612 steps (4 epochs) with warmup=161. At epoch 3 (step 806), R5 has completed its full decay curve while R6 is still mid-trajectory at ~1.6e-5 — a different and less optimal LR profile for 2 backbone epochs of training
- **Epoch 4 is the overfitting cliff**: train_loss drops sharply 0.6269 → 0.5433 while val_loss spikes 0.6557 → 0.7386 (+0.083), proba range blows out to [0.04, 0.99]
- **R6 ROC-AUC 0.6757 > R5 ROC-AUC 0.6679** even though F1 is lower — the epoch 3 checkpoint under the 4-epoch LR schedule ranks examples better but sets a slightly worse decision boundary; threshold tuning compensates less well on the 716-sample val set
- Root cause of R5 > R6: the phase-2 LR schedule should always be calibrated to the actual number of backbone epochs being trained, not the total EPOCHS count; with FREEZE_EPOCHS=1 and EPOCHS=3, R5 had the correctly sized schedule

**Confirmed best Option A config: EPOCHS=3, FREEZE_EPOCHS=1 (Run 5)**

**Next: Run 7 — Classifier dropout increase (`cls_dropout=0.3`)**

**Changes from Run 5 (the confirmed best base):**
- `EPOCHS = 3`, `FREEZE_EPOCHS = 1` — unchanged (R5 schedule)
- `CLS_DROPOUT = 0.3` (from default ~0.1) — passed to `AutoModelForSequenceClassification.from_pretrained` via the `cls_dropout` config key
- Everything else unchanged

**Decision rationale:** R5 established the right training regime. The remaining overfitting signal — val F1 slightly below holdout, val loss already rising at epoch 3, and all models topping out ~0.62 — suggests the classifier head is memorizing the 716-sample val distribution. Increasing `cls_dropout` from ~0.1 to 0.3 adds direct regularization at the head during fine-tuning without changing any other dynamics. This is the standard next lever after epoch count is confirmed.

**Expected behavior:**
- Epoch 1 val F1 slightly lower (more dropout = noisier head-only training)
- Epochs 2–3 val F1 should be more stable — higher dropout smooths the loss surface, potentially letting epoch 3 exceed 0.6129
- Holdout gap to val should narrow (less overfitting at the head)
- If 0.3 hurts, try 0.2 as a midpoint

