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


Here's the diagnosis:

  ┌─────────────────────────┬──────────────────────┬──────────────────────┐
  │         Metric          │ Run 1 (2e-5, wu=0.1) │ Run 2 (1e-5, wu=0.2) │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Best checkpoint epoch   │ 1                    │ 1                    │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Best val macro_f1 (ep1) │ 0.6122               │ 0.5993               │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Tuned val macro_f1      │ 0.6227 (thr=0.48)    │ 0.6193 (thr=0.56)    │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Holdout macro_f1        │ 0.6127               │ 0.6039               │
  ├─────────────────────────┼──────────────────────┼──────────────────────┤
  │ Holdout ROC-AUC         │ 0.6552               │ 0.6508               │
  └─────────────────────────┴──────────────────────┴──────────────────────┘

  What happened: Lowering LR to 1e-5 extracted less signal in epoch 1 (val F1 dropped from 0.6122 →
  0.5993), and the model still overfits from epoch 2 regardless. The val_loss spike at epoch 3 (0.6580
   → 0.7011) is even more severe than Run 1. The proba range blowing out to [0.12, 0.96] by epoch 3
  confirms overconfident collapse.

  Root cause: ~6,400 training samples is genuinely small for 86M params. The model saturates useful
  learning in one epoch — LR tuning doesn't fix that, regularization does.



  Option B — Layer-wise LR decay (LLRD) (principled): Apply 0.9× LR decay per DeBERTa layer so
──────────────────────────────────────────────────────────────────────────────────────────────────────
Which direction for Run 3?
     Simplest: extract max signal in one clean pass, avoid overfitting entirely. Easy to implement,
     fast run.
  0.5993), and the model still overfits from epoch 2 regardless. The val_loss spike at epoch 3 (0.6580
   → 0.7011) is even more severe than Run 1. The proba range blowing out to [0.12, 0.96] by epoch 3
  confirms overconfident collapse.

  Root cause: ~6,400 training samples is genuinely small for 86M params. The model saturates useful
  learning in one epoch — LR tuning doesn't fix that, regularization does.

  Run 3 proposal — one of two directions:

  Option A — "One-epoch wonder" (simplest, lowest risk): Set EPOCHS=1, LR=3e-5, cosine schedule, add
  label_smoothing=0.1. Take full advantage of the clean first pass, skip overfitting entirely.

  Option B — Layer-wise LR decay (LLRD) (principled): Apply 0.9× LR decay per DeBERTa layer so
  backbone gets ~3e-6 while the classifier head gets full 1e-5. Standard trick for preventing backbone

  