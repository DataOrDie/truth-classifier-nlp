# CatBoost Journey

Experiment log for `src/training/cat.py`. Captures the reasoning behind initial decisions, what we expect to see, and what to investigate after the run.

---

## Why CatBoost after LR + RFC + LGBM

The LGBM experiments revealed a clear bottleneck: `fe_speaker_true_rate` monopolises model capacity (6.7–7.9× the gain of any other feature), and no HP lever — early stopping, regularization, or modified num_leaves — fixes it. Option B (dropping the feature) was the only intervention that improved holdout Macro F1 (+0.012), but it discards real signal.

CatBoost attacks the root cause directly through two mechanisms that LGBM lacks:

**Ordered boosting (target leakage prevention)**
For each training sample, CatBoost computes that sample's target statistics (the true-rate estimate for its categorical group) using only the samples that came before it in a random permutation — never using the sample itself. This means:
- The model can't overfit to exact training true-rates the way LGBM does
- `fe_speaker_true_rate`-style features lose their artificially high gain because their estimates are noisier per-sample during training
- Other features get a fairer share of gradient signal early in boosting

**Symmetric (oblivious) trees**
CatBoost grows trees where every node at a given depth uses the same split condition. This is a strong regularizer that naturally limits how deeply any single feature can be exploited within one tree. LGBM's leaf-wise growth had no such constraint and could build very deep paths through `fe_speaker_true_rate` splits.

**Native categorical handling**
CatBoost computes ordered target statistics internally for each categorical feature, using the same ordered permutation trick. This is better than LightGBM's requirement to pre-encode categoricals as integers — CatBoost's internal encoding is learned rather than arbitrary.

---

## What Carries Forward from LGBM

`cat.py` inherits the full preprocessing config from `lgbm.py`. All preprocessing decisions are identical:

| Module | Key choices |
|--------|------------|
| **Statement** | Embeddings (`all-MiniLM-L6-v2`, 384-dim). No stemming/stopword removal. NER features enabled. Spelling error count, rare token features, lexical + pollution features all on. |
| **Subject** | `most_frequent` strategy. Rare grouping at threshold 10. Frequency encoding, length features, topic count. |
| **Speaker** | Rare grouping at threshold 5. Frequency, title flag, comma/period flags. |
| **Speaker job** | Rare grouping, title flag, comma/slash/ampersand flags. |
| **Party affiliation** | Rare grouping, `is_major_party`, `is_institutional` flags. |
| **State** | US region, is_us_state flag, frequency. |
| **Feature Engineering** | All interaction keys (speaker×subject, speaker×party, subject×party, speaker_job×subject, state×party, speaker×len_bucket). All text-style features. Non-leaking aggregates. |
| **Scaling** | None — trees are invariant to monotone transforms. |
| **True-rate features** | **All four kept** (speaker, subject, party, speaker_job). No drop like Option B — CatBoost's ordered boosting should handle the dominance without needing to remove the feature. |

---

## Categorical Handling — Key Difference from LGBM

Both `lgbm.py` and `cat.py` use `OrdinalEncoder` to convert grouped string columns (e.g. `speaker_grouped`, `subject_primary_grouped`, interaction keys) into integers. This is necessary for `RandomizedSearchCV` to call `fit(X, y)` without a custom wrapper.

The difference is what happens next:

```python
# Identify which columns are categorical (after ordinal encoding)
_cat_feature_names = [c for c in _cat_encoded.columns if c != label_source_col]

# Inside CV loop — recomputed each fold because true-rate columns shift column positions
_cat_indices = [X_fold_train.columns.get_loc(c) for c in _cat_feature_names
                if c in X_fold_train.columns]

# Pass to CatBoostClassifier constructor — applies to all fit() calls via RandomizedSearchCV
_base_cat = CatBoostClassifier(cat_features=_cat_indices, ...)
```

CatBoost receives integer-encoded values for categorical columns but knows which columns to treat categorically via `cat_features`. It then applies its internal ordered target statistics on top — effectively rebuilding its own, leakage-free encoding during each boosting iteration. This is strictly better than treating ordinal integers as numeric, which is all LGBM does.

The indices are recomputed per fold (not cached globally) because true-rate columns are added to `X_fold_train` after the ordinal encoding step, shifting later column positions.

---

## Model Decisions

### Class imbalance

`auto_class_weights='Balanced'` — CatBoost computes `n_samples / (n_classes × n_samples_i)` per fold. With 8,950 samples (35.25% class 0, 64.75% class 1) this gives ≈ [1.42, 0.77], identical to the manual weights used in LGBM/RFC.

Using `auto_class_weights='Balanced'` instead of passing a float list `class_weights=[1.42, 0.77]` is required for sklearn compatibility: CatBoost normalises the float list internally, so `get_params()` returns a different value than the constructor received, which triggers `clone()`'s RuntimeError inside `RandomizedSearchCV`. The string `'Balanced'` round-trips correctly.

The final model (no `clone()` needed) uses `class_weights=CLASS_WEIGHTS` for explicit control.

No resampling — class weighting avoids information loss and is consistent with prior experiments.

### Hyperparameters

| Parameter | Search range | Reasoning |
|-----------|-------------|-----------|
| `iterations` | [300, 500, 800] | CatBoost equivalent to n_estimators. Same range as LGBM — prior experiments found 500 was consistently chosen. |
| `learning_rate` | [0.03, 0.05, 0.1, 0.2] | Same range as LGBM. In clean LGBM runs, 0.03–0.05 always won; CatBoost's symmetric trees may prefer a slightly different rate. |
| `depth` | [4, 6, 8, 10] | CatBoost's `depth` controls symmetric tree depth. Default is 6. Deeper trees capture more complex interactions but symmetric growth means depth 10 = 1024 possible leaf patterns. Lower values (4) are strongly regularized. |
| `l2_leaf_reg` | [1, 3, 5, 10] | L2 penalty on leaf values. Default is 3. CatBoost's built-in regularization — analogous to LGBM's `reg_lambda`. |
| `border_count` | [32, 64, 128] | Number of histogram bins for numeric features. Default is 254. Lower values reduce overfitting on dense embedding dims; 384 embedding dims with 128 bins each is still very expressive. |
| `bagging_temperature` | [0.0, 0.5, 1.0] | Controls the Bayesian bootstrap: 0.0 = no randomness (all samples used with full weight), 1.0 = standard Bayesian bootstrap (Exponential(1) weights). Adds diversity between trees. |

**What's intentionally omitted:**
- `min_data_in_leaf` is not searched — CatBoost's symmetric trees + ordered boosting already regularise heavily; adding another leaf-size constraint risks over-regularization (the LGBM Option C lesson: trading leaf-size regularization for weight regularization doesn't generalise well).
- `subsample` is not searched — controlled by `bagging_temperature` instead (different mechanism, same purpose).

### `verbose=0`

Suppresses CatBoost's per-iteration training/validation log during CV. Without this, CatBoost prints a line per tree × 5 folds × 3 inner folds × 20 iterations = thousands of lines.

### `thread_count=1` in inner CV, `thread_count=-1` in final model

CatBoost is multi-threaded internally via `thread_count` (not sklearn's `n_jobs`). Using `thread_count=1` on the base model in inner CV prevents nested parallelism when `RandomizedSearchCV` runs with `n_jobs=-1`. The final model uses all cores.

---

## Training Strategy

### Same nested CV as LGBM

- **Outer loop**: 5-fold stratified CV — OOF probabilities for threshold tuning, unbiased metric estimates.
- **Inner loop**: 3-fold `RandomizedSearchCV`, 20 iterations per outer fold — finds best HP per fold without leaking from the validation fold.
- **HP aggregation**: all CatBoost HPs are discrete lists, so mode is used for all of them (no median needed).

### Threshold tuning

Same as LGBM: searches `[0.20, 0.76]` in steps of 0.02 on OOF probabilities, optimising Macro F1. The threshold is expected to be in the 0.48–0.58 range — LGBM consistently produced thresholds above 0.5 due to majority-class bias. CatBoost's symmetric trees and ordered boosting may produce better-calibrated probabilities and shift the threshold closer to or below 0.5.

### Feature importance

`model.get_feature_importance()` — CatBoost's default importance type is `PredictionValuesChange`: the average change in the model's prediction magnitude when a feature is removed, evaluated over all training samples. This is analogous to gain-based importance in LGBM and more interpretable than split-count.

---

## Expected Results

Based on the LGBM experiments and CatBoost's known characteristics on tabular NLP data:

| Metric | LGBM initial | LGBM Option B (best) | CatBoost expectation |
|--------|-------------|---------------------|---------------------|
| CV Macro F1 | 0.5934 | 0.5947 | **0.60–0.63** |
| Holdout Macro F1 | 0.6062 | **0.6179** | **0.62–0.65** |
| Holdout ROC-AUC | 0.6681 | **0.6790** | **0.68–0.72** |
| Training time (CV) | ~17 min | ~17 min | ~20–35 min |

CatBoost is expected to match or beat Option B because:
1. Ordered boosting reduces `fe_speaker_true_rate` monopoly without dropping the feature
2. Native categorical handling extracts more signal from interaction keys
3. Symmetric trees provide stronger implicit regularization

### What the feature importance should look like

If CatBoost's ordered boosting is working as expected:
- `fe_speaker_true_rate` gain ratio vs next feature should be < 3× (vs LGBM's 6.7–7.9×)
- Embedding dims should collectively claim a larger share
- Interaction key features (speaker×subject, etc.) may rise in importance — CatBoost encodes these natively, so it can exploit them more fully than LGBM with ordinal integers

If `fe_speaker_true_rate` still dominates at 6–7×, CatBoost is not overcoming the problem and the next step would be to combine Option B (drop the feature) with CatBoost.

### Interpreting the threshold

- **Threshold < 0.5**: CatBoost's probabilities are better calibrated; the class weight is compensating correctly for imbalance.
- **Threshold > 0.5** (like all LGBM runs): CatBoost still inherits the majority-class probability bias; similar to LGBM.
- **Threshold > 0.60**: a strong sign that ordered boosting is not enough and `fe_speaker_true_rate` is still dominating in an unbalanced way.

### What to watch in W&B

- `cv_mean_macro_f1` — compare directly to LGBM initial (0.5934) and Option B (0.5947).
- `holdout/macro_f1` — the bar to beat is Option B's 0.6179.
- `threshold/best` — below 0.5 would be a positive signal for calibration quality.
- `feature_importance/table` — top-5 ratio between `fe_speaker_true_rate` and the next feature.
- Fold-to-fold variance (`cv_std_macro_f1`) — CatBoost's stronger regularization should reduce variance vs LGBM's 0.0101–0.0136 range.

---

## Potential Issues

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| `RuntimeError: Cannot clone object ... modifies parameter cat_features` (or `class_weights`, `auto_class_weights`) | CatBoost normalises several constructor params internally; `get_params()` returns different values than the constructor received, failing sklearn's post-clone equality check | **Fixed**: `_CatBoostCV` wrapper class stores raw `**kwargs` before `super().__init__()` and returns them verbatim from `get_params()`. Used for all CV estimators; the final model uses plain `CatBoostClassifier` (no clone needed). |
| CatBoost verbose output floods console despite `verbose=0` | Some versions use `silent=True` | Add `silent=True` to constructor alongside `verbose=0` |
| `cat_features` index mismatch error | True-rate columns shift indices if added after `_cat_feature_names` is built | Already handled — indices recomputed per fold from `X_fold_train.columns.get_loc()` |
| `RandomizedSearchCV` with CatBoost is slower than expected | CatBoost is multi-threaded but `thread_count=1` limits it in inner CV | Expected — inner CV is the bottleneck; outer folds run sequentially |
| `predict_proba` not available | Only `CatBoostClassifier` (not `CatBoostRegressor`) exposes it | Already using the classifier |
| CV much slower than LGBM | CatBoost's ordered statistics require a permutation pass per fold | Normal; expect 20–35 min total. If > 60 min, reduce `iterations` upper bound to 500. |
| HP aggregation ties | With 5 discrete choices and 5 folds, ties are common | Current logic takes `most_common()[0]` — ties are broken by Counter's insertion order, which is arbitrary. Accept this; it means no strong preference exists. |

---

## Next Steps After This Run

1. **If CatBoost > Option B**: CatBoost is the right direction — tune further (more iterations, finer `depth` grid).
2. **If CatBoost ≈ Option B**: try CatBoost + Option B together (drop `fe_speaker_true_rate` and let CatBoost work on the remaining features).
3. **If CatBoost ≤ LGBM initial**: the bottleneck is the feature space, not the model — investigate bigger embeddings (`all-mpnet-base-v2`, 768-dim) or fine-tuned transformers.
4. **Stacking**: combine OOF probas from LR + RFC + LGBM-optB + CatBoost as inputs to a meta-LR. Even modest individual models produce complementary errors — stacking often yields +1–2 F1 points at very low implementation cost.


---

## Initial Run Analysis

### Performance vs expectations

| Metric | Expected | Actual | vs LGBM Option B |
|--------|----------|--------|-----------------|
| CV Macro F1 | 0.60–0.63 | **0.6002** | +0.0055 vs 0.5947 |
| Holdout Macro F1 | 0.62–0.65 | **0.6184** | +0.0005 vs 0.6179 ← new best |
| Holdout ROC-AUC | 0.68–0.72 | **0.6653** | -0.0137 vs 0.6790 |
| Training time | 20–35 min | **98 min** | 3–5× slower than LGBM |

The result is a narrow win for Macro F1 (+0.0005 over Option B) but a clear loss on ROC-AUC (-0.0137). This is a mixed outcome: CatBoost produces better decisions at the tuned threshold but less well-separated probability scores overall.

### Threshold: first crossing below 0.5

All LGBM runs required a threshold above 0.5 (range: 0.52–0.58) to compensate for the model's tendency to assign majority-class probabilities. CatBoost tuned to **0.48** — the first time a model in this project has crossed below 0.5. The OOF threshold curve peaks sharply at 0.48 and falls steadily on both sides, indicating a well-defined optimum.

This confirms the expected calibration improvement from ordered boosting and symmetric trees. The class weighting (`auto_class_weights='Balanced'`) is interacting correctly with the boosting mechanism — the model is genuinely assigning higher probability of class 1 rather than relying on a post-hoc threshold shift to overcome the imbalance.

### HP convergence

| Parameter | Chosen | Votes (5 folds) | Interpretation |
|-----------|--------|-----------------|----------------|
| `iterations` | 500 | 3/5 | Consistent with LGBM — mid-range preferred |
| `learning_rate` | 0.03 | 4/5 | Near-unanimous: slow learning wins again |
| `depth` | 4 | 3/5 | Shallowest option selected — strong regularisation needed |
| `l2_leaf_reg` | 5 | 2/5 | Tied with 1 — no strong preference |
| `border_count` | 64 | 3/5 | Moderate histogram bins |
| `bagging_temperature` | 0.0 | 3/5 | No Bayesian bootstrap noise preferred |

`depth=4` is the most informative result. CatBoost's symmetric trees at depth 4 produce 2⁴ = 16 leaf patterns per tree. Combined with `bagging_temperature=0.0` (no subsample randomness), the model favours a conservative, low-variance configuration on this dataset. This is consistent with the 8,950-sample size — deeper symmetric trees generalise poorly when data is limited.

`learning_rate=0.03` winning 4/5 folds matches LGBM's pattern exactly. The optimal learning rate is a dataset property (noise level, feature scale, imbalance ratio) — it doesn't change with the model architecture.

### Feature importance — dominance ratio

`fe_speaker_true_rate` scores **21.9** (PredictionValuesChange); the next feature, `fe_subject_true_rate`, scores **2.53**. Raw ratio: **8.76×**.

This appears worse than LGBM's 6.7–7.9× gain ratio, but the comparison is not direct. PredictionValuesChange measures the average absolute change in predicted value when a feature is randomly shuffled — it is dominated by features whose signal is concentrated rather than distributed. LGBM's gain is cumulative across all splits in all trees; a feature used shallowly in many trees accumulates large gain even if each split contributes little. The two metrics compress differently, so the ratio cannot be compared to LGBM's numerically.

What matters: ordered boosting did not achieve the < 3× reduction in dominance ratio that was hoped for. `fe_speaker_true_rate` remains the single most important feature by a wide margin. The dominance problem is not solved.

### What ordered boosting did accomplish

Despite the unresolved dominance, two positive signals appear in the importance table:

**`speaker_grouped` at rank 3 (1.4629)** — this is a native categorical feature that LGBM could only treat as an ordinal integer. CatBoost's internal target statistics extracted enough signal to place it above `fe_party_true_rate` and `fe_speaker_job_true_rate`. This is direct evidence that native categorical handling is working.

**`fe_subject_true_rate` at rank 2 (2.53)** — substantially higher than any non-speaker feature seen in LGBM. CatBoost is distributing importance more broadly across the four true-rate features rather than concentrating everything in `fe_speaker_true_rate`.

**Embedding dimensions collectively** — approximately 20 embedding dims appear in the top 30 (all with scores 0.55–1.20), representing a significant aggregate share. The embeddings are contributing signal.

**`statement_original_PERSON` at rank 12 (0.89)** — NER PERSON count is the highest-ranked NER feature, confirming that named entity information adds value beyond what the embedding captures.

### Class recall balance

| | Precision | Recall | F1 |
|---|-----------|--------|----|
| Class 0 (true) | 0.49 | 0.57 | 0.53 |
| Class 1 (false) | 0.74 | 0.68 | 0.71 |

Class 0 recall improved to 0.57 (from 0.52 in LGBM Option B at threshold 0.52). The threshold shift to 0.48 is doing its job — more class 0 predictions at the cost of slightly lower class 1 precision. The macro F1 improvement comes from this recall boost.

### Known output display issue

The HP aggregation section shows `np.float64(0.03)` and `np.float64(0.0)` instead of plain `0.03` and `0.0`. This is cosmetic — `_rng.choice()` returns numpy scalars; the `int()` cast handles integer params but leaves floats as numpy types. The actual values used in training are correct. Fix: add `.item()` or `float()` cast alongside the existing `int()` branch.

### Verdict and next direction

CatBoost is the current best model by Macro F1 (0.6184), but the margin over LGBM Option B is within one standard deviation of both models' CV noise. ROC-AUC is lower, meaning the probability ordering is not better — only the threshold calibration improved.

The dominance problem persists under both architectures. The natural next experiment is **CatBoost + Option B**: keep CatBoost's ordered boosting and native categoricals but remove `fe_speaker_true_rate`. If Option B added +0.012 to LGBM and CatBoost's better calibration is complementary, the combination could push into the 0.63+ range. Alternatively, stacking OOF probabilities from LGBM-optB and CatBoost as meta-features captures their complementary error profiles at low implementation cost.

---

## Results Summary

| Model | CV Macro F1 | Holdout Macro F1 | Holdout ROC-AUC | Threshold |
|-------|------------|-----------------|----------------|-----------|
| LGBM initial | 0.5934 ± 0.0101 | 0.6062 | 0.6681 | 0.58 |
| LGBM Option B (drop speaker_true_rate) | 0.5947 ± 0.0136 | 0.6179 | 0.6790 | 0.52 |
| LGBM Option C (L1/L2 reg) | 0.5910 ± 0.0111 | 0.6069 | 0.6636 | 0.52 |
| **CatBoost initial** | **0.6002 ± 0.0158** | **0.6184** | 0.6653 | **0.48** |

---

-----------------------------------------

--> Catboost initial run output
[SECTION] Cross-validation summary  [total CV: 5891.6s]
  roc_auc: 0.6500 ± 0.0144
  pr_auc: 0.7603 ± 0.0079
  macro_f1: 0.6002 ± 0.0158
  f1: 0.6972 ± 0.0238
  precision: 0.7295 ± 0.0159
  recall: 0.6700 ± 0.0481
  accuracy: 0.6247 ± 0.0184
  mcc: 0.2071 ± 0.0326
  balanced_acc: 0.6058 ± 0.0166

[SECTION] Aggregating HP search results  [21:16:12]
  iterations               : [(500, 3), (300, 2)]  → chosen: 500
  learning_rate            : [(np.float64(0.03), 4), (np.float64(0.1), 1)]  → chosen: 0.03
  depth                    : [(4, 3), (6, 2)]  → chosen: 4
  l2_leaf_reg              : [(5, 2), (1, 2), (10, 1)]  → chosen: 5
  border_count             : [(64, 3), (32, 2)]  → chosen: 64
  bagging_temperature      : [(np.float64(0.0), 3), (np.float64(1.0), 2)]  → chosen: 0.0

  Final HP for fit: {'iterations': 500, 'learning_rate': np.float64(0.03), 'depth': 4, 'l2_leaf_reg': 5, 'border_count': 64, 'bagging_temperature': np.float64(0.0)}

[SECTION] Threshold tuning on OOF predictions  [21:16:12]
   threshold   macro_f1
        0.20   0.4314
        0.22   0.4435
        0.24   0.4539
        0.26   0.4656
        0.28   0.4779
        0.30   0.4933
        0.32   0.5128
        0.34   0.5276
        0.36   0.5445
        0.38   0.5583
        0.40   0.5712
        0.42   0.5803
        0.44   0.5925
        0.46   0.6013
        0.48   0.6062  ←
        0.50   0.6012
        0.52   0.6000
        0.54   0.5948
        0.56   0.5869
        0.58   0.5739
        0.60   0.5623
        0.62   0.5452
        0.64   0.5295
        0.66   0.5113
        0.68   0.4958
        0.70   0.4759
        0.72   0.4575
        0.74   0.4384
        0.76   0.4214

  Best threshold: 0.48  (OOF macro_f1=0.6062)
  THRESHOLD updated: 0.50 → 0.48
[SECTION] Fitting final model on full train/val set  [21:16:13]
  Done in 8.0s
[SECTION] Evaluating on holdout set  [21:16:21]
  Using threshold: 0.48

Holdout results:
  roc_auc: 0.6653
  pr_auc: 0.7644
  macro_f1: 0.6184
  f1: 0.7085
  precision: 0.7438
  recall: 0.6764
  accuracy: 0.6397
  mcc: 0.2413
  balanced_acc: 0.6243

              precision    recall  f1-score   support

           0       0.49      0.57      0.53       631
           1       0.74      0.68      0.71      1159

    accuracy                           0.64      1790
   macro avg       0.62      0.62      0.62      1790
weighted avg       0.65      0.64      0.64      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                21.9059
    fe_subject_true_rate                                2.5340
    speaker_grouped                                     1.4629
    fe_party_true_rate                                  1.4578
    fe_speaker_job_true_rate                            1.4446
    statement_original_vec_99                           1.2030
    statement_original_vec_35                           1.1494
    statement_original_vec_291                          1.0766
    statement_original_vec_164                          1.0701
    statement_original_vec_4                            1.0566
    statement_original_vec_158                          0.9691
    statement_original_PERSON                           0.8875
    statement_original_vec_1                            0.8731
    fe_speaker_party                                    0.8661
    statement_upper_ratio                               0.7827
    statement_original_vec_0                            0.7649
    statement_original_vec_249                          0.7378
    statement_original_vec_245                          0.7236
    statement_original_vec_97                           0.7178
    statement_original_vec_119                          0.7171
    statement_original_vec_171                          0.6998
    statement_original_vec_313                          0.6960
    statement_original_vec_188                          0.6860
    statement_original_vec_159                          0.6346
    statement_original_vec_30                           0.6224
    statement_original_vec_10                           0.6120
    statement_original_char_len                         0.5925
    statement_original_vec_329                          0.5793
    statement_original_vec_37                           0.5778
    statement_original_vec_14                           0.5537