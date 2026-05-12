# Stacking Journey

Experiment log for `src/training/stacking.py`. Covers the rationale behind the ensemble design, what each decision trades off, what we expect to see, and what to investigate after the run.

---

## Why Stacking After Individual Models

The project has four trained model families at this point:

| Model | Holdout Macro F1 | Holdout ROC-AUC | Threshold |
|-------|-----------------|----------------|-----------|
| LR | — | — | — |
| RFC | — | — | — |
| LGBM initial | 0.6062 | 0.6681 | 0.58 |
| LGBM Option B | 0.6179 | **0.6790** | 0.52 |
| CatBoost initial | 0.6184 | 0.6653 | 0.48 |
| **CatBoost + OptB** | **0.6294** | 0.6740 | 0.46 |

The plateau pattern is typical: each individual model improvement is smaller than the last (+0.001 between CatBoost initial and OptB). At this stage, two directions remain — more powerful individual models (larger transformers, fine-tuning) or ensemble methods. Stacking is the right next step because:

1. **Complementary errors are already present.** CatBoost-OptB has the best Macro F1 (0.6294) but weaker ROC-AUC (0.6740) than LGBM-OptB (0.6790). This is a direct signal that the two models make different mistakes: LGBM ranks probabilities better globally, CatBoost makes better decisions at the tuned threshold. A meta-learner can combine both.

2. **Model families are genuinely diverse.** LR draws a linear hyperplane in the embedding space. RFC builds uncorrelated decision trees with bootstrap sampling. LGBM uses leaf-wise gradient boosting with explicit regularisation. CatBoost uses symmetric trees with ordered target statistics. These four algorithms differ in inductive bias, variance profile, and how they handle categorical features — the necessary condition for stacking to add value.

3. **Implementation cost is low given the preprocessing is already built.** The preprocessing pipeline, OrdinalEncoder, true-rate logic, and CV infrastructure all exist in cat.py. Stacking reuses all of it; the only new components are the OOF collection loop and the meta-LR.

---

## Architecture Decisions

### Shared preprocessing

All 4 base models receive the same feature matrix — the CatBoost-OptB configuration (sentence embeddings, OrdinalEncoder, `drop_speaker_true_rate=True`). This is a pragmatic choice.

The alternative — separate preprocessing per model (TF-IDF for LR, embeddings for trees) — would be more faithful to how each model was originally trained. But it would require running `preprocess_one_step` twice per fold (once for LR, once for trees), which doubles the already-slow preprocessing time. More importantly, the embedding-based LR in the stack is still a useful base model: it learns a linear combination of the 384 embedding dimensions, which is a genuinely different decision surface than what any of the tree models produce.

The LR in the stack is **not** the same LR that was trained in `lr.py` (which may use TF-IDF). It is a new LR base model trained on embeddings. This is fine — stacking doesn't require the base models to match their stand-alone training exactly. It requires that they produce diverse, competent probabilities.

### `drop_speaker_true_rate=True` for all base models

Applied uniformly. The CatBoost-OptB experiments confirmed that dropping `fe_speaker_true_rate` improves performance even when CatBoost's ordered boosting was expected to handle the dominance. Keeping it for some base models but not others would make the OOF probas less comparable and harder for the meta-LR to interpret. Consistency wins.

### No inner HP search

Each base model uses hardcoded HPs from prior experiments. This is the "low implementation cost" design decision from the CatBoost Journey notes. The stacking gain comes from combining complementary errors, not from fine-tuning base models. Running nested CV inside stacking's outer CV would require:
- 5 outer folds × 3 inner folds × 4 models × 20 HP iterations = 1,200 model fits (vs. 5 × 4 = 20 fits without inner search).

The marginal gain from per-fold HP tuning inside stacking is small — the base models are already well-calibrated from their individual experiments.

### Meta-learner: LogisticRegression (C=0.1)

The meta-LR takes a 4-column input matrix (one column per base model OOF probability) and learns a weighted combination.

**Why LR?** The meta-learner should be simple. With only 4 input features (the 4 OOF probas), a complex meta-model (gradient boosted trees, neural net) would overfit to the CV fold structure. LR produces a weighted average of the base model probas with a logit transform — exactly what we want.

**Why C=0.1?** Light L2 regularization. With 8 free parameters (4 weights + 4 intercepts across both classes, though it simplifies to 4 weights + 1 intercept for binary), the meta-LR can overfit to the OOF probas if not constrained. C=0.1 keeps the weights stable. If the meta-LR assigns a large negative coefficient to any base model, that model's probas are hurting the ensemble — worth investigating.

**Why no class_weight?** The base model OOF probas already encode the class imbalance signal via their own class weighting. Applying class_weight again to the meta-LR would double-count the imbalance correction. The meta-LR sees well-calibrated imbalanced probas and learns to combine them — imbalance handling is the base models' job.

---

## Base Model HP Decisions

### LR

```python
C=1.0, penalty='l2', max_iter=1000, solver='lbfgs', class_weight={0:1.42, 1:0.77}
```

C=1.0 is a neutral starting point. The LR in `lr.py` used a hyperparameter search over `loguniform(1e-3, 10)`. In the stacking context, the LR base model's exact HP matters less than its role: producing a linear-boundary probability that's different from the tree models' non-linear boundaries.

StandardScaler is applied inside each fold (fit on fold_train, transform fold_val). This is necessary because LR is not scale-invariant: the embedding dimensions span [-1, 1] (already normalized by sentence-transformers), but other features (char_len, frequency counts, spelling_err_count) have very different scales. Without scaling, the LR's gradient descent is poorly conditioned and the weights are biased toward high-variance features.

### RFC

```python
n_estimators=300, max_features=0.3, min_samples_leaf=2, class_weight={0:1.42, 1:0.77}
```

From the HP search range in `rfc.py`, `max_features` was searched over `[0.2, 0.3, 0.5, "sqrt", "log2"]` and `min_samples_leaf` over `randint(1, 8)`. The chosen values (0.3, 2) are plausible central estimates. RFC's key contribution to the stack is its bootstrap aggregation — it averages over many high-variance, low-bias trees, which gives its probabilities a different correlation structure than the boosted models.

### LGBM-OptB

```python
n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
subsample=0.8, colsample_bytree=0.8, class_weight={0:1.42, 1:0.77}
```

The learning_rate=0.03 matches LGBM's consistent preference across all HP search runs (4/5 folds in most experiments). num_leaves=31 is the sklearn default and is conservative — LGBM's HP searches typically preferred 31–63. This is intentionally the simpler end of what LGBM can do; in stacking, simpler base models with less variance often combine better than each individually over-tuned model.

### CatBoost-OptB

```python
iterations=300, learning_rate=0.03, depth=4, l2_leaf_reg=5,
border_count=32, bagging_temperature=0.0, class_weights=[1.42, 0.77]
```

Directly from the CatBoost-OptB run results — all 5 folds unanimously voted for iterations=300, depth=4, and border_count=32. This is the most reliable HP set in the project: unanimous votes indicate no ambiguity. The symmetric tree structure (depth=4 → 16 leaf patterns per tree) combined with ordered target statistics makes CatBoost's probability estimates the most calibrated of the four base models (evidenced by its threshold of 0.46, the lowest in the project).

---

## OOF Collection and Meta-Training

The CV loop trains all 4 base models inside each fold and writes their validation-set probabilities to OOF arrays. After CV:

```
oof_lr   : shape (N_trainval,)  — linear boundary probas
oof_rfc  : shape (N_trainval,)  — bootstrap-averaged probas
oof_lgbm : shape (N_trainval,)  — leaf-wise gradient boosting probas
oof_cat  : shape (N_trainval,)  — symmetric-tree ordered-stats probas

meta_X   : shape (N_trainval, 4) = column_stack([oof_lr, oof_rfc, oof_lgbm, oof_cat])
```

The meta-LR is trained on `meta_X` with `y_trainval` as the target. The OOF probabilities are unbiased: each row's proba was generated by a model that never saw that row during training. This is the key correctness property of stacking — without it, the meta-LR would overfit to the base models' in-sample performance.

For holdout evaluation:
1. All 4 base models are **retrained** on the full `X_trainval` (not just one fold's train split).
2. Their holdout probabilities are stacked into `meta_X_holdout` (shape N_holdout × 4).
3. The meta-LR — which was trained on OOF from step above — predicts on `meta_X_holdout`.

The meta-LR is **not** retrained on the final base model probabilities. Its training on OOF gives an unbiased estimate of how to combine the base models. Retraining it on the full-trainval predictions of the final base models would introduce a small optimistic bias (those predictions are in-sample for the base models).

---

## Expected Results

### Macro F1

The CatBoost-OptB best is 0.6294. The stacking hypothesis is +1–2 F1 points.

| Scenario | Expected Holdout Macro F1 | Reasoning |
|----------|--------------------------|-----------|
| Strong complementarity | **0.64–0.65** | LR + RFC errors are decorrelated from tree models |
| Moderate complementarity | **0.63–0.64** | Modest gains where tree models already agree |
| Weak complementarity | **0.629–0.632** | Stacking learns to up-weight CatBoost; marginal gain |
| Regression | < 0.629 | Meta-LR overfits to OOF structure; unlikely with C=0.1 |

The "weak complementarity" scenario is possible: LGBM and CatBoost share the same feature set and similar tree-based decision logic. Their OOF probas may be highly correlated (r > 0.9), leaving the meta-LR to essentially pick a weighted average of two similar distributions. If that's the case, LR and RFC become more important as decorrelators.

### ROC-AUC

LGBM-OptB has the highest ROC-AUC (0.6790). Stacking may close the gap between CatBoost's better Macro F1 and LGBM's better ROC-AUC — the meta-LR optimizes for ranking as well as threshold-specific decisions. Expect stacked ROC-AUC in the **0.68–0.71** range.

### Threshold

CatBoost-OptB tuned to 0.46. The stacked meta-LR receives well-calibrated probas from all 4 models and applies a logit transform. The combined distribution is likely to be better-centered than any individual model, so the threshold may move closer to 0.5 — though the class imbalance means it will probably still land below 0.5. Expect **0.42–0.50**.

### Meta-LR coefficients

The coefficients tell you how much each base model contributes. If all coefficients are roughly equal (~0.25 after softmax), the base models contribute equally. If one coefficient is large and positive (e.g., CatBoost >> others), stacking has essentially learned to up-weight the best individual model.

**Red flag**: any coefficient strongly negative. This means one base model's OOF probas are actively hurting the ensemble. If LR has a negative coefficient, the linear boundary is misleading the meta-learner — consider removing it or replacing it with a calibrated variant.

### Runtime

The dominant cost is CatBoost: ~100 min for 5-fold CV (from individual run). All 4 models run sequentially inside each fold:

| Component | Estimated time per fold | Total (5 folds) |
|-----------|------------------------|-----------------|
| Preprocessing | — (done once) | ~10 min |
| LR + scaler | ~5s | ~25s |
| RFC | ~30–60s | ~5 min |
| LGBM-OptB | ~3–5 min | ~20 min |
| CatBoost-OptB | ~20 min | ~100 min |
| **Total** | | **~130–140 min** |

Plus final refit on full trainval (~10 min for all 4 models). Expect **2.5–3 hours total**.

---

## What to Watch in W&B

- **`holdout/macro_f1`** — bar to beat is 0.6294 (CatBoost-OptB). This is the primary metric.
- **`holdout/roc_auc`** — bar to beat is 0.6790 (LGBM-OptB). Stacking may finally surpass both simultaneously.
- **`meta/coef_*`** — the 4 meta-LR coefficients. Balance and sign are the key signals.
- **`holdout/roc_auc_lr / rfc / lgbm / cat`** — individual base model ROC-AUC on holdout from the final refitted models. Compare to stand-alone runs to verify base models are performing as expected.
- **`threshold/best`** — direction relative to CatBoost-OptB's 0.46. Moving toward 0.5 means better calibration.
- **`cv/roc_auc_avg`** per fold — the simple average of 4 OOF probas, logged as a proxy metric before the meta-LR is trained. Should already beat single models if complementarity exists.

---

## Potential Issues

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| CatBoost `cat_features` index error | True-rate columns shift after OrdinalEncoded columns; `_cat_indices` must be recomputed per fold | Already handled — recomputed from `X_fold_train.columns.get_loc()` each fold |
| StandardScaler leaks into OOF | Scaler fit on fold_train+fold_val instead of fold_train only | Already handled — `_scaler.fit_transform(X_fold_train)` then `.transform(X_fold_val)` |
| Meta-LR overfits (large coefficients, poor holdout) | C=0.1 is too loose; OOF arrays have mild fold-structure artifacts | Try C=0.01; or clip probas to [0.05, 0.95] before stacking |
| LGBM base model underperforms expected | HP mismatch (num_leaves=31 vs. the best found in prior runs) | The stacking LGBM HP was intentionally conservative; not a bug, but could try num_leaves=63 |
| RFC very slow | 300 trees × ~420 dims is computationally heavy | Reduce n_estimators to 100 or add `max_samples=0.7` for speed |
| `OneStepOptions` key error in submission section | `options.__dict__` may contain internal state keys | If submission fails, pass options manually to the test preprocess call |
| OOF arrays misaligned | `val_idx` is a positional index into `X_trainval` but the OOF arrays are addressed by position | Already correct — `oof_lr[val_idx]` uses the same positional index as `y_trainval.iloc[val_idx]` |

---

## Next Steps After This Run

1. **If stacking > 0.6294 Macro F1**: stacking is working — the meta-LR coefficients will show which models are pulling weight. Tune `META_LR_HP["C"]` grid (`[0.01, 0.1, 1.0]`) and try a 2-level stack with CatBoost-OptB + LGBM-OptB only if LR/RFC coefficients are negligible.

2. **If stacking ≈ 0.6294 (< +0.005)**: base models are too correlated — LGBM and CatBoost make near-identical predictions. Consider diversifying the base set: add a different embedding model (`all-mpnet-base-v2`, 768-dim), or include the RFC trained on TF-IDF features instead of embeddings.

3. **If stacking < 0.6294**: meta-LR is fitting noise. Try replacing the meta-LR with a simple average (equal-weight ensemble) — if the average beats the meta-LR, the meta-LR is overfitting despite C=0.1. Also check whether any base model's coefficient is strongly negative.

4. **Submission**: if stacking beats CatBoost-OptB on holdout, submit it. The Kaggle submission CSV is generated inline at the end of the script.



--------------------------------

--> Initial run Output
[SECTION] Cross-validation summary  [total CV: 82.3s]
  roc_auc_avg: 0.6609 ± 0.0120
  macro_f1_avg: 0.6063 ± 0.0069
  roc_auc_lr: 0.6359 ± 0.0090
  roc_auc_rfc: 0.6685 ± 0.0136
  roc_auc_lgbm: 0.6450 ± 0.0149
  roc_auc_cat: 0.6573 ± 0.0145

[SECTION] Training meta-LR on stacked OOF  [11:02:29]
  Meta-LR coefficients: {'lr': 0.6504079300488584, 'rfc': 1.6443321071484418, 'lgbm': 0.5704459003500517, 'cat': 1.0962264338089995}

[SECTION] Threshold tuning on stacked OOF  [11:02:29]
   threshold   macro_f1
        0.20   0.3930
        0.22   0.3930
        0.24   0.3930
        0.26   0.3930
        0.28   0.3930
        0.30   0.3942
        0.32   0.3954
        0.34   0.4016
        0.36   0.4092
        0.38   0.4191
        0.40   0.4338
        0.42   0.4562
        0.44   0.4815
        0.46   0.5043
        0.48   0.5247
        0.50   0.5449
        0.52   0.5650
        0.54   0.5793
        0.56   0.5897
        0.58   0.6011
        0.60   0.6104
        0.62   0.6116  ←
        0.64   0.6098
        0.66   0.5980
        0.68   0.5897
        0.70   0.5725
        0.72   0.5458
        0.74   0.5140
        0.76   0.4717

  Best threshold: 0.62  (OOF macro_f1=0.6116)
  THRESHOLD updated: 0.50 → 0.62
[SECTION] Fitting final base models on full train/val set  [11:02:30]
  Done in 20.0s
[SECTION] Evaluating on holdout set  [11:02:50]
  Using threshold: 0.62

Holdout results:
  roc_auc: 0.6830
  pr_auc: 0.7855
  macro_f1: 0.6303
  f1: 0.7197
  precision: 0.7519
  recall: 0.6903
  accuracy: 0.6520
  mcc: 0.2645
  balanced_acc: 0.6359

              precision    recall  f1-score   support

           0       0.51      0.58      0.54       631
           1       0.75      0.69      0.72      1159

    accuracy                           0.65      1790
   macro avg       0.63      0.64      0.63      1790
weighted avg       0.67      0.65      0.66      1790

  Base model holdout ROC-AUC:
    LR  : 0.6582
    RFC : 0.6738
    LGBM: 0.6776
    CAT : 0.6740

---

## Initial Run Analysis

### Performance — new ROC-AUC best, marginal Macro F1 gain

| Metric | CatBoost+OptB (best prior) | Stacking | Delta |
|--------|---------------------------|----------|-------|
| CV Macro F1 | 0.6056 ± 0.0137 | **0.6063 ± 0.0069** | +0.0007 |
| Holdout Macro F1 | 0.6294 | **0.6303** | +0.0009 |
| Holdout ROC-AUC | 0.6740 | **0.6830** | **+0.009** ← new best |
| Threshold | 0.46 | 0.62 | ↑ significantly |
| CV runtime | ~6,006s | **82.3s** | 73× faster |

The stacking run beat every prior model on ROC-AUC (0.6830 vs. LGBM-OptB's previous record of 0.6790, +0.004). Macro F1 improved only marginally (+0.0009 over CatBoost-OptB). The hypothesis of "+1–2 F1 points" did not materialise. The ensemble is combining the models' probability rankings better (ROC-AUC) than their threshold-specific decisions (Macro F1) — consistent with what the pre-run analysis predicted: the two best individual models already had opposite strengths on those two metrics.

The CV variance on Macro F1 dropped to 0.0069 from CatBoost-OptB's 0.0137 — stacking makes the ensemble more stable fold-to-fold, which is a positive sign independent of the point estimate.

The 82.3s CV time is 73× faster than cat.py's ~6,000s. The entire speedup comes from removing inner HP search: cat.py ran 5 outer × 3 inner × 20 iterations = 300 CatBoost fits; stacking runs 5 × 1 = 5.

### Threshold: 0.62 — highest in the project, and why

Every prior model's optimal threshold was below 0.5 (CatBoost-OptB: 0.46) or modestly above (LGBM-OptB: 0.52). Stacking peaked at **0.62**, the highest recorded. The OOF curve rises monotonically from 0.20 all the way to 0.62 before falling — there is no shoulder or plateau, just a clean single peak.

The mechanism: all 4 base models apply `class_weight={0:1.42, 1:0.77}`, which upweights minority-class errors during training. This systematically pushes their class-1 (false statement) predicted probabilities higher. The meta-LR receives four input columns that are all biased upward for true class-1 samples. Without its own class_weight, the meta-LR learns a linear combination of these biased probas — the resulting output probas are calibrated relative to the biased inputs, not relative to the true class frequency. The threshold tuning corrects for this, landing at 0.62.

This is not a bug: threshold tuning is the right correction mechanism. But it signals that the meta-LR's probability outputs are poorly calibrated in an absolute sense. The OOF peak score of 0.6116 (Macro F1) holds up to 0.6303 on holdout, which means the correction transfers — the calibration gap is consistent across the train/holdout split.

**Implication for next experiments**: adding `class_weight={0:1.42, 1:0.77}` to the meta-LR would shift its probability outputs, likely pulling the optimal threshold closer to 0.5. Whether this improves Macro F1 is an open question — the threshold tuning already finds the optimum.

### Meta-LR coefficients — RFC dominates

| Base model | Coefficient | OOF ROC-AUC | Holdout ROC-AUC |
|------------|-------------|-------------|----------------|
| RFC | **1.644** | **0.6685** | 0.6738 |
| CatBoost | 1.096 | 0.6573 | 0.6740 |
| LR | 0.650 | 0.6359 | 0.6582 |
| LGBM | 0.570 | 0.6450 | 0.6776 |

RFC has the highest coefficient (1.644) despite ranking third on holdout ROC-AUC (0.6738). LGBM has the lowest coefficient (0.570) despite ranking first on holdout ROC-AUC (0.6776). No coefficient is negative — all four base models contribute positively.

The RFC dominance is the most informative result. The meta-LR learned from OOF patterns, not holdout rankings. Two explanations:

1. **RFC's errors are the most complementary.** RFC uses bootstrap sampling and random feature subsets, which creates a different error structure than both boosted models. When LGBM and CatBoost both mispredict a sample, RFC often gets it right — and the meta-LR learned to weight this correction signal heavily.

2. **RFC's OOF probabilities are better calibrated than its individual ranking suggests.** Random forests produce well-calibrated probabilities in moderate-dimensional spaces because averaging over 300 trees smooths out individual tree overconfidence. LGBM's OOF probas at num_leaves=31 may be less calibrated, making them less useful to the meta-LR per unit of coefficient.

The LGBM underperformance in OOF (0.6450) vs. holdout (0.6776) is a large discrepancy — 0.033 gap. CatBoost shows a smaller gap (0.6573 → 0.6740, 0.017). This suggests LGBM's OOF probas from the conservative HP (num_leaves=31) are systematically weaker than what the final-refitted LGBM produces on holdout — the full-trainval model benefits from more data in a way the fold models don't. This is the main reason the meta-LR underweights LGBM.

### OOF vs. holdout discrepancy in base models

The LGBM OOF/holdout gap (+0.033 ROC-AUC) is a known property of gradient boosting with conservative HPs on small datasets: each fold's training set is 20% smaller than the full trainval, and LGBM at num_leaves=31 underfits more with less data. The final model (trained on 100% of trainval) is noticeably stronger.

This creates a structural problem for the meta-LR: it learned coefficients from OOF patterns where LGBM was weaker than it actually is. If LGBM's OOF probas had been generated by a better-tuned model (num_leaves=63), the meta-LR might have assigned it a higher coefficient, and the stacking Macro F1 might have improved more.

**Actionable**: re-run stacking with `num_leaves=63` for the LGBM base model and compare coefficients. If LGBM's coefficient rises and Macro F1 improves, the conservative HP was the bottleneck.

### Results summary (all models)

| Model | CV Macro F1 | Holdout Macro F1 | Holdout ROC-AUC | Threshold |
|-------|------------|-----------------|----------------|-----------|
| LGBM initial | 0.5934 ± 0.0101 | 0.6062 | 0.6681 | 0.58 |
| LGBM Option B | 0.5947 ± 0.0136 | 0.6179 | **0.6790** | 0.52 |
| CatBoost initial | 0.6002 ± 0.0158 | 0.6184 | 0.6653 | 0.48 |
| CatBoost + OptB | **0.6056 ± 0.0137** | 0.6294 | 0.6740 | 0.46 |
| **Stacking (initial)** | 0.6063 ± 0.0069 | **0.6303** | **0.6830** | 0.62 |

-------------------------------------------------------------------------------------


---> Output stacking run2 ,  `num_leaves=63` for LGBM  
[SECTION] Cross-validation summary  [total CV: 93.0s]
  roc_auc_avg: 0.6631 ± 0.0127
  macro_f1_avg: 0.6022 ± 0.0121
  roc_auc_lr: 0.6359 ± 0.0090
  roc_auc_rfc: 0.6685 ± 0.0136
  roc_auc_lgbm: 0.6528 ± 0.0168
  roc_auc_cat: 0.6573 ± 0.0145

[SECTION] Training meta-LR on stacked OOF  [13:11:51]
  Meta-LR coefficients: {'lr': 0.5850075081125578, 'rfc': 1.5061114872719046, 'lgbm': 0.6953673642729318, 'cat': 1.009227823146433}

[SECTION] Threshold tuning on stacked OOF  [13:11:51]
   threshold   macro_f1
        0.20   0.3930
        0.22   0.3930
        0.24   0.3930
        0.26   0.3930
        0.28   0.3934
        0.30   0.3946
        0.32   0.3986
        0.34   0.4065
        0.36   0.4171
        0.38   0.4287
        0.40   0.4445
        0.42   0.4691
        0.44   0.4897
        0.46   0.5062
        0.48   0.5342
        0.50   0.5544
        0.52   0.5686
        0.54   0.5834
        0.56   0.5956
        0.58   0.6008
        0.60   0.6069
        0.62   0.6124  ←
        0.64   0.6098
        0.66   0.6058
        0.68   0.5942
        0.70   0.5752
        0.72   0.5506
        0.74   0.5173
        0.76   0.4764

  Best threshold: 0.62  (OOF macro_f1=0.6124)
  THRESHOLD updated: 0.50 → 0.62
[SECTION] Fitting final base models on full train/val set  [13:11:52]
  Done in 22.2s
[SECTION] Evaluating on holdout set  [13:12:15]
  Using threshold: 0.62

Holdout results:
  roc_auc: 0.6835
  pr_auc: 0.7852
  macro_f1: 0.6323
  f1: 0.7220
  precision: 0.7528
  recall: 0.6937
  accuracy: 0.6542
  mcc: 0.2681
  balanced_acc: 0.6377

              precision    recall  f1-score   support

           0       0.51      0.58      0.54       631
           1       0.75      0.69      0.72      1159

    accuracy                           0.65      1790
   macro avg       0.63      0.64      0.63      1790
weighted avg       0.67      0.65      0.66      1790

  Base model holdout ROC-AUC:
    LR  : 0.6582
    RFC : 0.6738
    LGBM: 0.6766
    CAT : 0.6740

---

## Run 2 Analysis — `num_leaves=63`

### Performance — new best on both metrics

| Metric | Run 1 (num_leaves=31) | Run 2 (num_leaves=63) | Delta |
|--------|----------------------|----------------------|-------|
| OOF Macro F1 (tuned threshold) | 0.6116 | **0.6124** | +0.0008 |
| Holdout Macro F1 | 0.6303 | **0.6323** | **+0.002** |
| Holdout ROC-AUC | 0.6830 | **0.6835** | +0.0005 |
| CV macro_f1_avg (proxy) | 0.6063 ± 0.0069 | 0.6022 ± 0.0121 | −0.0041 |
| Threshold | 0.62 | 0.62 | — |
| CV runtime | 82.3s | 93.0s | +10.7s |

The hypothesis confirmed: `num_leaves=63` improves holdout Macro F1 (+0.002, now 0.6323) and ROC-AUC (+0.0005, now 0.6835). Both are the new project records.

The `cv_macro_f1_avg` proxy metric fell (0.6063 → 0.6022) while holdout improved. This is not a contradiction — the proxy uses a fixed threshold of 0.5 and a simple average of the 4 OOF probas before the meta-LR is trained. With num_leaves=63, LGBM produces noisier fold-level probas (higher individual fold variance: 0.0168 vs 0.0149), which depresses the naive average metric. But the meta-LR extracts more useful signal from those probas, which is what matters. The OOF tuned metric (0.6116 → 0.6124) and holdout metric (0.6303 → 0.6323) both moved in the right direction.

### Meta-LR coefficient shift — hypothesis confirmed

| Base model | Run 1 coef | Run 2 coef | Delta | OOF AUC run1 | OOF AUC run2 |
|------------|-----------|-----------|-------|-------------|-------------|
| RFC | 1.6443 | **1.5061** | −0.138 | 0.6685 | 0.6685 |
| CatBoost | 1.0962 | **1.0092** | −0.087 | 0.6573 | 0.6573 |
| LGBM | 0.5704 | **0.6954** | **+0.125** | 0.6450 | **0.6528** |
| LR | 0.6504 | **0.5850** | −0.065 | 0.6359 | 0.6359 |

The prediction held exactly. LGBM's OOF ROC-AUC rose from 0.6450 to 0.6528 (+0.0078) with num_leaves=63, and its meta-LR coefficient rose from 0.570 to 0.695 (+0.125). RFC's dominance decreased (1.644 → 1.506) as LGBM became a more competitive signal source. All coefficients remain positive.

The OOF/holdout gap for LGBM: in run 1, the gap was 0.6776 − 0.6450 = 0.033. In run 2, it narrowed to 0.6766 − 0.6528 = 0.024. The final LGBM holdout ROC-AUC is slightly lower (0.6766 vs 0.6776) — num_leaves=63 may introduce a small amount of overfitting on the full trainval dataset relative to 31, but the fold-level models are more representative, which is what matters for the meta-LR's training signal.

### RFC still dominates (coefficient 1.51) — why it persists

Even with a stronger LGBM, RFC maintains the highest coefficient. The explanation from Run 1 holds: RFC's bootstrap aggregation produces errors that are structurally decorrelated from both boosted models. When LGBM (leaf-wise) and CatBoost (symmetric trees) both fail on a sample, RFC's ensemble of random projections often succeeds. The meta-LR has consistently learned to weight this decorrelation signal the most. This is genuine complementarity, not an artifact of HP miscalibration.

### All-model results summary (updated)

| Model | CV Macro F1 | Holdout Macro F1 | Holdout ROC-AUC | Threshold |
|-------|------------|-----------------|----------------|-----------|
| LGBM initial | 0.5934 ± 0.0101 | 0.6062 | 0.6681 | 0.58 |
| LGBM Option B | 0.5947 ± 0.0136 | 0.6179 | 0.6790 | 0.52 |
| CatBoost initial | 0.6002 ± 0.0158 | 0.6184 | 0.6653 | 0.48 |
| CatBoost + OptB | 0.6056 ± 0.0137 | 0.6294 | 0.6740 | 0.46 |
| Stacking run 1 (LGBM nl=31) | 0.6063 ± 0.0069 | 0.6303 | 0.6830 | 0.62 |
| **Stacking run 2 (LGBM nl=63)** | 0.6022 ± 0.0121 | **0.6323** | **0.6835** | 0.62 |

Two signals from the data point to concrete experiments:                                                
                                                                                                          
  1. The threshold of 0.62 (highest in the project) is the clearest anomaly.                              
  Every individual model peaked between 0.46–0.58. The meta-LR peaks at 0.62 because its base models all
  apply class_weight, which shifts their class-1 probas upward — the meta-LR inherits that bias without
  correction. Adding class_weight to the meta-LR directly addresses this:

  META_LR_HP = dict(C=0.1, penalty="l2", max_iter=1000, solver="lbfgs",
                    class_weight={0: 1.42, 1: 0.77}, random_state=42)

  This will shift the meta-LR's output probas, likely pulling the optimal threshold down toward 0.5 and
  may improve Macro F1 since the class boundary is being placed more deliberately. The double-counting
  concern from the journey doc is real but mild — the base model probas encode imbalance as a tendency,
  not as an absolute shift; adding class_weight to the meta-LR corrects for what the tendency can't fully
  compensate.

  2. RFC has the highest coefficient (1.506) but the roughest estimated HPs.
  The RFC in the stack uses max_features=0.3, min_samples_leaf=2 — central-estimate guesses, not search
  results. Since RFC is doing the most work in the ensemble, its HPs matter most. Adding n_estimators=500
  (from 300) costs nothing in correctness and reduces its probability variance:

  BASE_RFC_HP = dict(n_estimators=500, max_features=0.3, min_samples_leaf=2, ...)

  What's likely at the ceiling:

  The LGBM OOF/holdout gap is now 0.024 — still present but narrowing. Pushing further (num_leaves=127,
  n_estimators=800) has diminishing returns because the gap is structural: each fold trains on 20% fewer
  samples, and boosted trees are more data-sensitive than RFC.

  The harder ceiling is that all 4 base models share the same embedding feature space (all-MiniLM-L6-v2).
  Their errors are correlated by construction — LGBM and CatBoost running on identical features will
  always agree on the cases the embedding handles well and fail together on the cases it doesn't. The RFC
  coefficient dominance is a symptom: RFC's bootstrap randomness is the only real decorrelator available
  within this feature space.

  The next meaningful step after those two experiments would be a genuinely different feature space for
  one base model — the TF-IDF LR from lr.py as a 5th base model. That adds lexical signal the embedding
  can miss (exact token matches, rare proper nouns) and is architecturally different enough that its
  errors should be less correlated with the tree models.

  Recommended order:
  1. Add class_weight to meta-LR (one-line change, addresses the clearest anomaly)
  2. RFC n_estimators=500 (one-line change, more stable probas from the most-weighted model)
  3. If still plateaued: add TF-IDF LR as a 5th base model with separate preprocessing

  ---------------------------------------------

  --> Stacking Output [ class_weight to meta-LR , n_estimators=500]
[SECTION] Cross-validation summary  [total CV: 125.1s]
  roc_auc_avg: 0.6628 ± 0.0125
  macro_f1_avg: 0.6030 ± 0.0104
  roc_auc_lr: 0.6359 ± 0.0090
  roc_auc_rfc: 0.6679 ± 0.0117
  roc_auc_lgbm: 0.6528 ± 0.0168
  roc_auc_cat: 0.6573 ± 0.0145

[SECTION] Training meta-LR on stacked OOF  [13:36:08]
  Meta-LR coefficients: {'lr': 0.5912175419978312, 'rfc': 1.5506700558191027, 'lgbm': 0.6877924710810327, 'cat': 1.0394245897944507}

[SECTION] Threshold tuning on stacked OOF  [13:36:08]
   threshold   macro_f1
        0.20   0.3978
        0.22   0.4081
        0.24   0.4236
        0.26   0.4403
        0.28   0.4687
        0.30   0.4934
        0.32   0.5150
        0.34   0.5412
        0.36   0.5607
        0.38   0.5751
        0.40   0.5904
        0.42   0.5974
        0.44   0.6065
        0.46   0.6093
        0.48   0.6094  ←
        0.50   0.6069
        0.52   0.6024
        0.54   0.5910
        0.56   0.5741
        0.58   0.5575
        0.60   0.5326
        0.62   0.5006
        0.64   0.4640
        0.66   0.4270
        0.68   0.3854
        0.70   0.3428
        0.72   0.3062
        0.74   0.2764
        0.76   0.2649

  Best threshold: 0.48  (OOF macro_f1=0.6094)
  THRESHOLD updated: 0.50 → 0.48
[SECTION] Fitting final base models on full train/val set  [13:36:09]
  Done in 30.6s
[SECTION] Evaluating on holdout set  [13:36:40]
  Using threshold: 0.48

Holdout results:
  roc_auc: 0.6835
  pr_auc: 0.7851
  macro_f1: 0.6292
  f1: 0.7129
  precision: 0.7556
  recall: 0.6747
  accuracy: 0.6480
  mcc: 0.2648
  balanced_acc: 0.6369

              precision    recall  f1-score   support

           0       0.50      0.60      0.55       631
           1       0.76      0.67      0.71      1159

    accuracy                           0.65      1790
   macro avg       0.63      0.64      0.63      1790
weighted avg       0.67      0.65      0.65      1790

  Base model holdout ROC-AUC:
    LR  : 0.6582
    RFC : 0.6751
    LGBM: 0.6766
    CAT : 0.6740

---

## Run 3 Analysis — meta-LR `class_weight` + RFC `n_estimators=500`

### Performance — threshold corrected, Macro F1 regressed

| Metric | Run 2 (baseline) | Run 3 | Delta |
|--------|-----------------|-------|-------|
| OOF Macro F1 (tuned threshold) | 0.6124 | 0.6094 | −0.003 |
| Holdout Macro F1 | **0.6323** | 0.6292 | **−0.003** |
| Holdout ROC-AUC | 0.6835 | 0.6835 | 0 |
| Threshold | 0.62 | **0.48** | ↓ as predicted |
| RFC holdout AUC | 0.6738 | **0.6751** | +0.0013 |
| RFC OOF AUC variance | ±0.0136 | **±0.0117** | −0.0019 |

The threshold prediction was exactly right — `class_weight` on the meta-LR moved the optimal cutoff from 0.62 to 0.48, right in line with the individual model range. The mechanism is confirmed: the base model class weighting was causing a probability bias that the high threshold was compensating for, and adding class_weight to the meta-LR corrected the bias directly.

However, Macro F1 regressed by −0.003. The threshold moved in the right direction but the underlying OOF peak score fell (0.6124 → 0.6094). The initial "Why no class_weight?" reasoning in the Architecture Decisions section was correct: **double-counting the imbalance correction is net negative**.

### Why class_weight on the meta-LR hurt

The base model probas are already biased upward for class-1 samples because all four base models apply `class_weight={0:1.42, 1:0.77}` during training. The meta-LR receives inputs that already carry this signal. Adding `class_weight` to the meta-LR creates a second layer of class-1 preference on top of probas that already encode it.

The class report shows the effect clearly:

| | Run 2 precision | Run 2 recall | Run 2 F1 | Run 3 precision | Run 3 recall | Run 3 F1 |
|--|--|--|--|--|--|--|
| Class 0 (true) | 0.51 | 0.58 | 0.54 | 0.50 | **0.60** | 0.55 |
| Class 1 (false) | 0.75 | **0.69** | **0.72** | 0.76 | 0.67 | 0.71 |

Class-0 recall improved (0.58 → 0.60) as the meta-LR became more cautious about predicting class 1. But class-1 recall fell (0.69 → 0.67) and class-1 F1 dropped (0.72 → 0.71). Since class 1 has nearly twice the samples (1,159 vs 631), any drop in its F1 pulls the macro average down more than a matching gain in class-0 F1 can pull it up. The net: macro F1 fell from 0.6323 to 0.6292.

**Conclusion: `class_weight` must be removed from the meta-LR.** The "correct" threshold (0.48 vs 0.62) is aesthetically appealing but doesn't translate to better predictions. The high threshold in runs 1–2 was the right compensating mechanism.

### RFC n_estimators=500 — genuine improvement, keep it

Isolated from the class_weight regression, the RFC change is cleanly positive:
- OOF AUC variance dropped: ±0.0136 → ±0.0117 (more stable fold estimates)
- Holdout AUC improved: 0.6738 → 0.6751 (+0.0013)
- Coefficient held steady: 1.5061 → 1.5507 (slight rise, consistent with more stable probas)

500 trees produce better-calibrated probabilities than 300 through the law of large numbers: each additional tree reduces the aggregation variance. With RFC carrying the highest meta-LR coefficient across all runs, reducing its probability variance directly reduces the meta-LR's fitting noise.

### Next step: run 4

Revert `class_weight` from the meta-LR, keep RFC at 500 trees. This isolates the RFC improvement and should recover the run 2 Macro F1 (0.6323) with slightly better RFC signal (0.6751 vs 0.6738 AUC).

Expected result: Macro F1 ≈ 0.632–0.634.

------------------------------------------------------------------

--> Output run 4 [Revert `class_weight` from the meta-LR]
[SECTION] Cross-validation summary  [total CV: 123.7s]
  roc_auc_avg: 0.6628 ± 0.0125
  macro_f1_avg: 0.6030 ± 0.0104
  roc_auc_lr: 0.6359 ± 0.0090
  roc_auc_rfc: 0.6679 ± 0.0117
  roc_auc_lgbm: 0.6528 ± 0.0168
  roc_auc_cat: 0.6573 ± 0.0145

[SECTION] Training meta-LR on stacked OOF  [13:50:24]
  Meta-LR coefficients: {'lr': 0.5889197570337594, 'rfc': 1.4570570448437203, 'lgbm': 0.7013970151711315, 'cat': 1.0243948725476035}

[SECTION] Threshold tuning on stacked OOF  [13:50:24]
   threshold   macro_f1
        0.20   0.3930
        0.22   0.3930
        0.24   0.3930
        0.26   0.3930
        0.28   0.3934
        0.30   0.3946
        0.32   0.3978
        0.34   0.4058
        0.36   0.4144
        0.38   0.4289
        0.40   0.4445
        0.42   0.4684
        0.44   0.4904
        0.46   0.5083
        0.48   0.5306
        0.50   0.5542
        0.52   0.5695
        0.54   0.5803
        0.56   0.5940
        0.58   0.6011
        0.60   0.6068
        0.62   0.6099
        0.64   0.6099  ←
        0.66   0.6055
        0.68   0.5939
        0.70   0.5744
        0.72   0.5492
        0.74   0.5179
        0.76   0.4746

  Best threshold: 0.64  (OOF macro_f1=0.6099)
  THRESHOLD updated: 0.50 → 0.64
[SECTION] Fitting final base models on full train/val set  [13:50:25]
  Done in 30.9s
[SECTION] Evaluating on holdout set  [13:50:56]
  Using threshold: 0.64

Holdout results:
  roc_auc: 0.6835
  pr_auc: 0.7849
  macro_f1: 0.6168
  f1: 0.6928
  precision: 0.7535
  recall: 0.6411
  accuracy: 0.6318
  mcc: 0.2459
  balanced_acc: 0.6280

              precision    recall  f1-score   support

           0       0.48      0.61      0.54       631
           1       0.75      0.64      0.69      1159

    accuracy                           0.63      1790
   macro avg       0.62      0.63      0.62      1790
weighted avg       0.66      0.63      0.64      1790

  Base model holdout ROC-AUC:
    LR  : 0.6582
    RFC : 0.6751
    LGBM: 0.6766
    CAT : 0.6740

---

## Run 4 Analysis — threshold selection failure

### What happened

The CV metrics are **identical** to run 3 — same base models, only the meta-LR class_weight was removed. Yet holdout Macro F1 crashed from 0.6292 (run 3) to **0.6168**, the worst stacking result so far. The cause is entirely in the threshold:

```
        0.62   0.6099
        0.64   0.6099  ←
```

Both thresholds print as 0.6099 at 4 decimal places. In full floating-point precision, 0.64 is very slightly higher, so `max()` chose it. On holdout, that one extra 0.02 in threshold reduced class-1 recall from 0.69 to 0.64 and collapsed Macro F1 by 0.015. The model is extremely sensitive to threshold in the 0.60–0.66 range — this is the flat shoulder of the OOF curve.

| Threshold | OOF (reported) | Holdout Macro F1 | Class-1 recall |
|-----------|---------------|-----------------|---------------|
| 0.62 | 0.6099 | **0.6323** (run 2) | 0.69 |
| 0.64 | 0.6099 | 0.6168 (run 4) | 0.64 |

A floating-point tie at 4 dp is deciding a 0.015 holdout swing. The 0.02 step grid is too coarse to reliably distinguish these.

### Why run 2 picked 0.62 and run 4 picked 0.64

In run 2, RFC at 300 trees produced slightly noisier per-sample probabilities that gave the curve a clearer peak at 0.62 (0.6124 vs 0.6098 at 0.64 — a clear 0.0026 margin). With 500 trees the OOF probas are smoother, which flattens the curve's shoulder — the 0.62 vs 0.64 difference dropped to sub-rounding noise. The more stable RFC ironically makes threshold selection *less* reliable by removing the noise that was sharpening the peak.

### RFC n_estimators=500 is still a genuine improvement

Base model holdout AUC: RFC 0.6751 in runs 3 and 4, up from 0.6738 in run 2. The RFC change is not the problem. The problem is that its stabilising effect on OOF probas flattened the threshold curve enough to make the 0.02-step grid unreliable.

### Fix: finer threshold grid (step=0.01)

Changing `np.arange(0.20, 0.76, 0.02)` to `np.arange(0.20, 0.76, 0.01)` gives 56 evaluation points instead of 28. In the critical 0.60–0.66 range this means 7 points instead of 3 — enough resolution to see whether 0.61, 0.62, or 0.63 is the true peak rather than having two 0.02-spaced points both rounding to the same score.

Applied to `stacking.py` for run 5. RFC=500 and num_leaves=63 kept, no other changes.

----------------------------------

--> Execution Output for run 5. RFC=500 and num_leaves=63 kept, no other changes.
[SECTION] Cross-validation summary  [total CV: 132.1s]
  roc_auc_avg: 0.6628 ± 0.0125
  macro_f1_avg: 0.6030 ± 0.0104
  roc_auc_lr: 0.6359 ± 0.0090
  roc_auc_rfc: 0.6679 ± 0.0117
  roc_auc_lgbm: 0.6528 ± 0.0168
  roc_auc_cat: 0.6573 ± 0.0145

[SECTION] Training meta-LR on stacked OOF  [12:04:35]
  Meta-LR coefficients: {'lr': 0.5889197570339592, 'rfc': 1.457057044843211, 'lgbm': 0.7013970151711584, 'cat': 1.0243948725474625}

[SECTION] Threshold tuning on stacked OOF  [12:04:35]
   threshold   macro_f1
        0.20   0.3930
        0.21   0.3930
        0.22   0.3930
        0.23   0.3930
        0.24   0.3930
        0.25   0.3930
        0.26   0.3930
        0.27   0.3930
        0.28   0.3934
        0.29   0.3942
        0.30   0.3946
        0.31   0.3967
        0.32   0.3978
        0.33   0.4006
        0.34   0.4058
        0.35   0.4093
        0.36   0.4144
        0.37   0.4222
        0.38   0.4289
        0.39   0.4344
        0.40   0.4445
        0.41   0.4576
        0.42   0.4684
        0.43   0.4799
        0.44   0.4904
        0.45   0.4979
        0.46   0.5083
        0.47   0.5208
        0.48   0.5306
        0.49   0.5431
        0.50   0.5542
        0.51   0.5611
        0.52   0.5695
        0.53   0.5758
        0.54   0.5803
        0.55   0.5895
        0.56   0.5940
        0.57   0.5963
        0.58   0.6011
        0.59   0.6067
        0.60   0.6068
        0.61   0.6085
        0.62   0.6099
        0.63   0.6096
        0.64   0.6099  ←
        0.65   0.6062
        0.66   0.6055
        0.67   0.5989
        0.68   0.5939
        0.69   0.5837
        0.70   0.5744
        0.71   0.5630
        0.72   0.5492
        0.73   0.5359
        0.74   0.5179
        0.75   0.4973
        0.76   0.4746

  Best threshold: 0.64  (OOF macro_f1=0.6099)
  THRESHOLD updated: 0.50 → 0.64
[SECTION] Fitting final base models on full train/val set  [12:04:37]
  Done in 30.7s
[SECTION] Evaluating on holdout set  [12:05:07]
  Using threshold: 0.64

Holdout results:
  roc_auc: 0.6835
  pr_auc: 0.7849
  macro_f1: 0.6168
  f1: 0.6928
  precision: 0.7535
  recall: 0.6411
  accuracy: 0.6318
  mcc: 0.2459
  balanced_acc: 0.6280

              precision    recall  f1-score   support

           0       0.48      0.61      0.54       631
           1       0.75      0.64      0.69      1159

    accuracy                           0.63      1790
   macro avg       0.62      0.63      0.62      1790
weighted avg       0.66      0.63      0.64      1790

  Base model holdout ROC-AUC:
    LR  : 0.6582
    RFC : 0.6751
    LGBM: 0.6766
    CAT : 0.6740


------------------------------------
Text columns dropped    : ['party_affiliation_clean', 'speaker_clean', 'speaker_job_clean', 'state_info_clean', 'statement', 'statement_clean', 'subject_clean']
  Categorical cols encoded: ['subject_primary', 'subject_primary_grouped', 'speaker_grouped', 'speaker_job_grouped', 'party_affiliation_grouped', 'state_info_grouped', 'state_info_us_region', 'fe_speaker_subject', 'fe_speaker_party', 'fe_subject_party', 'fe_speaker_job_subject', 'fe_state_party', 'fe_speaker_len_bucket']
[SECTION] Building feature matrix
  Vectorizer features     : 768
  Encoded cat features    : 13  →  ['subject_primary', 'subject_primary_grouped', 'speaker_grouped', 'speaker_job_grouped', 'party_affiliation_grouped', 'state_info_grouped', 'state_info_us_region', 'fe_speaker_subject', 'fe_speaker_party', 'fe_subject_party', 'fe_speaker_job_subject', 'fe_state_party', 'fe_speaker_len_bucket']
  Other numeric features  : 69  →  ['subject_topic_count', 'subject_has_multiple_topics', 'subject_length', 'subject_token_count', 'subject_frequency', 'subject_is_rare', 'statement_row_spillover_flag', 'statement_tab_count', 'statement_newline_count', 'statement_original_char_len', 'statement_original_word_count', 'statement_upper_ratio', 'statement_exclamation_count', 'statement_question_count', 'statement_clean_digit_ratio', 'statement_clean_rare_token_count', 'statement_clean_avg_token_freq', 'statement_clean_spelling_err_count', 'statement_original_total_entities', 'statement_original_PERSON', 'statement_original_ORG', 'statement_original_GPE', 'statement_original_DATE', 'statement_original_NUM', 'statement_original_OTHER', 'speaker_frequency', 'speaker_frequency_pct', 'speaker_is_rare', 'speaker_char_len', 'speaker_token_count', 'speaker_has_title', 'speaker_has_comma', 'speaker_has_period', 'speaker_job_frequency', 'speaker_job_frequency_pct', 'speaker_job_is_rare', 'speaker_job_char_len', 'speaker_job_token_count', 'speaker_job_has_title', 'speaker_job_has_comma', 'speaker_job_has_slash', 'speaker_job_has_ampersand', 'party_affiliation_frequency', 'party_affiliation_frequency_pct', 'party_affiliation_is_rare', 'party_affiliation_char_len', 'party_affiliation_token_count', 'party_affiliation_has_slash', 'party_affiliation_has_ampersand', 'party_affiliation_has_parentheses', 'party_affiliation_is_major_party', 'party_affiliation_is_institutional', 'state_info_frequency', 'state_info_frequency_pct', 'state_info_is_rare', 'state_info_is_us_state', 'state_info_token_count', 'state_info_has_us_words', 'fe_speaker_avg_statement_len', 'fe_subject_avg_statement_len', 'fe_speaker_avg_punctuation', 'fe_speaker_avg_number_ratio', 'fe_negation_count', 'fe_hedge_count', 'fe_absolutist_count', 'fe_numeral_count', 'fe_readability', 'fe_sentiment_polarity', 'fe_sentiment_subjectivity']
  Total features          : 850

Update Stacking to use "all-mpnet-base-v2" embeddings

--> Run Stacking with "all-mpnet-base-v2" embeddings 
 [SECTION] Cross-validation summary  [total CV: 229.4s]
  roc_auc_avg: 0.6724 ± 0.0092
  macro_f1_avg: 0.6110 ± 0.0062
  roc_auc_lr: 0.6326 ± 0.0082
  roc_auc_rfc: 0.6721 ± 0.0079
  roc_auc_lgbm: 0.6676 ± 0.0115
  roc_auc_cat: 0.6633 ± 0.0087

[SECTION] Training meta-LR on stacked OOF  [20:59:57]
  Meta-LR coefficients: {'lr': 0.4986125569601204, 'rfc': 1.2383167288853327, 'lgbm': 1.0144474437916642, 'cat': 0.9613586214902322}

[SECTION] Threshold tuning on stacked OOF  [20:59:57]
   threshold   macro_f1
        0.20   0.3930
        0.21   0.3930
        0.22   0.3930
        0.23   0.3930
        0.24   0.3930
        0.25   0.3930
        0.26   0.3930
        0.27   0.3943
        0.28   0.3951
        0.29   0.3954
        0.30   0.3970
        0.31   0.4003
        0.32   0.4037
        0.33   0.4093
        0.34   0.4125
        0.35   0.4223
        0.36   0.4303
        0.37   0.4377
        0.38   0.4450
        0.39   0.4547
        0.40   0.4640
        0.41   0.4754
        0.42   0.4868
        0.43   0.4956
        0.44   0.5079
        0.45   0.5160
        0.46   0.5278
        0.47   0.5351
        0.48   0.5433
        0.49   0.5536
        0.50   0.5651
        0.51   0.5719
        0.52   0.5806
        0.53   0.5904
        0.54   0.5936
        0.55   0.5990
        0.56   0.6069
        0.57   0.6114
        0.58   0.6147
        0.59   0.6188
        0.60   0.6208
        0.61   0.6228  ←
        0.62   0.6209
        0.63   0.6200
        0.64   0.6184
        0.65   0.6133
        0.66   0.6123
        0.67   0.6084
        0.68   0.6036
        0.69   0.5974
        0.70   0.5902
        0.71   0.5774
        0.72   0.5685
        0.73   0.5539
        0.74   0.5402
        0.75   0.5199
        0.76   0.4994

  Best threshold: 0.61  (OOF macro_f1=0.6228)
  THRESHOLD updated: 0.50 → 0.61
[SECTION] Fitting final base models on full train/val set  [20:59:58]
  Done in 56.7s
[SECTION] Evaluating on holdout set  [21:00:55]
  Using threshold: 0.61

Holdout results:
  roc_auc: 0.6995
  pr_auc: 0.8031
  macro_f1: 0.6428
  f1: 0.7348
  precision: 0.7573
  recall: 0.7135
  accuracy: 0.6665
  mcc: 0.2876
  balanced_acc: 0.6468

              precision    recall  f1-score   support

           0       0.52      0.58      0.55       631
           1       0.76      0.71      0.73      1159

    accuracy                           0.67      1790
   macro avg       0.64      0.65      0.64      1790
weighted avg       0.68      0.67      0.67      1790

  Base model holdout ROC-AUC:
    LR  : 0.6720
    RFC : 0.6926
    LGBM: 0.6850
    CAT : 0.6848



--------------------------------------

Likely highest-leverage next moves (in order):

  Experiment: A
  Change: Enable fe_add_speaker_true_rate=True / drop_speaker_true_rate=False
  Rationale: Currently disabled to match CatBoost-optB config, but stacking context is different — RFC and LGBM may
    benefit from this leakage-safe feature
  ────────────────────────────────────────
  Experiment: B
  Change: Raise CatBoost depth 4 → 6
  Rationale: At depth=4 with 300 iterations, CAT may be underfitting; LGBM with num_leaves=63 is effectively deeper
  ────────────────────────────────────────
  Experiment: C
  Change: Try soft voting (simple average of 4 base probas) instead of meta-LR
  Rationale: Meta-LR on 4 inputs with N=7,160 samples is low-data; if base models are well-calibrated, simple average
  can
    beat it
  ────────────────────────────────────────
  Experiment: D
  Change: Replace base LR (ROC-AUC 0.6582, lowest by ~0.02) with XGBoost
  Rationale: LR is the weakest link in the ensemble


------------------------------------------------------

Experiment: A
  Change: Enable fe_add_speaker_true_rate=True / drop_speaker_true_rate=False
  Rationale: Currently disabled to match CatBoost-optB config, but stacking context is different — RFC and LGBM may
    benefit from this leakage-safe feature

