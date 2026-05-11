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
