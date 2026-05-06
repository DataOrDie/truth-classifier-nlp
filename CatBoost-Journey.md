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
