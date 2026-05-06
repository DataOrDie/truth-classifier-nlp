# LightGBM Journey

Experiment log for `src/training/lgbm.py`. Captures the reasoning behind initial decisions, what we expect to see, and what to investigate after the run.

---

## Why LightGBM after LR + RFC

We have two baselines: Logistic Regression (linear, good at high-dimensional sparse features) and Random Forest (tree ensemble, handles mixed feature types). LightGBM is the natural next step because:

- **Gradient boosting vs bagging**: RFC builds trees independently and averages them. LightGBM builds each tree to correct the residual errors of all previous trees — a much more efficient use of each split.
- **Leaf-wise growth**: LightGBM grows the leaf with the highest gain at each step, rather than level-wise like XGBoost. This makes it faster and often more accurate on tabular data.
- **Our feature matrix is a good fit**: 384-dim sentence embeddings + ~30 metadata/FE features. Gradient boosting handles mixed dense+structured feature tables extremely well.
- **Expected speed**: On 8,950 samples, one full CV run should complete in a few minutes on CPU — much faster than RFC with NER preprocessing.

---

## Feature Engineering — Same as RFC

`lgbm.py` inherits the full RFC preprocessing config. Key decisions carried forward:

| Module | Key choices |
|--------|------------|
| **Statement** | Embeddings (`all-MiniLM-L6-v2`, 384-dim). No stemming/stopword removal (embeddings need natural language). NER features enabled (`en_core_web_sm`). Spelling error count, rare token features, lexical + pollution features all on. |
| **Subject** | `most_frequent` strategy for primary topic. Rare grouping at threshold 10. Frequency encoding, length features, topic count. |
| **Speaker** | Rare grouping at threshold 5. Frequency, title flag, comma flag, period flag. |
| **Speaker job** | Rare grouping, title flag, comma/slash/ampersand flags. |
| **Party affiliation** | Rare grouping, `is_major_party`, `is_institutional` flags. |
| **State** | US region, is_us_state flag, frequency. |
| **Feature Engineering** | All interaction keys enabled (speaker×subject, speaker×party, subject×party, speaker_job×subject, state×party, speaker×statement_len_bucket). All text-style features (negation, hedge, absolutist, numeral, proper noun, readability, sentiment). Non-leaking aggregates (avg statement length, avg punctuation, avg number ratio per speaker/subject). |
| **Scaling** | None on all modules — trees are invariant to monotone transforms. |
| **True-rate features** | Computed fold-safe inside CV loop for speaker, subject, party, speaker_job. Fallback = 0.5 for unseen groups. These are the strongest individual signal on PolitiFact-style data. |

No scaling is applied to any module. `OrdinalEncoder` is used for categorical string columns (interaction keys, grouped categories, regions).

---

## Model Decisions

### Class imbalance
`CLASS_WEIGHT = {0: 1.42, 1: 0.77}` — same weights as RFC and LR. Class 0 (true statements, 35.25% of data) is the minority and gets upweighted. `LGBMClassifier` accepts the same dict format as sklearn, so no change needed.

No resampling — class weighting is sufficient and avoids information loss.

### Why these LightGBM hyperparameters

| Parameter | Value / Search range | Reasoning |
|-----------|---------------------|-----------|
| `n_estimators` | [300, 500, 800] | More trees = lower bias; early stopping not used so 800 is the upper cap to avoid overfitting. |
| `learning_rate` | [0.03, 0.05, 0.1, 0.2] | Low rates need more trees; high rates train faster but may overfit. 0.1 is the LightGBM default and a strong starting point. |
| `num_leaves` | [31, 63, 127] | The primary complexity control in LightGBM (vs `max_depth`). 31 is the default; 127 allows the model to capture complex interactions but risks overfitting on 8k samples. |
| `min_child_samples` | randint(10, 50) | Minimum samples required to form a leaf — key regularizer. Higher = more conservative. Sampled continuously because the optimal value is data-dependent. |
| `subsample` | [0.7, 0.8, 0.9, 1.0] | Row subsampling per tree — adds diversity and reduces overfitting. |
| `colsample_bytree` | [0.7, 0.8, 0.9, 1.0] | Feature subsampling per tree — especially useful with 400+ features (mostly embedding dims). |
| `verbose` | -1 | Silences LightGBM's per-iteration stdout so CV output stays readable. |

`max_depth` is intentionally left at its default (-1, unconstrained) and controlled indirectly through `num_leaves` and `min_child_samples`. This is the standard LightGBM approach.

---

## Training Strategy

### Nested cross-validation
- **Outer loop**: 5-fold stratified CV — produces OOF probabilities for threshold tuning and unbiased metric estimates.
- **Inner loop**: 3-fold `RandomizedSearchCV` with 20 iterations inside each outer fold — finds the best HP combination per fold without leaking from the validation fold.
- **HP aggregation**: list/categorical params (n_estimators, learning_rate, num_leaves, subsample, colsample_bytree) use the **mode** across folds; integer params (min_child_samples) use the **median**.

The aggregated params are used for the final fit on the full train/val set (80% of data).

### Threshold tuning
Searches `[0.20, 0.76]` in steps of 0.02 on OOF probabilities. Target metric: `macro_f1`. The dataset is imbalanced so the optimal threshold is almost always below 0.5 — lowering it increases recall on the minority class (true statements) at a controlled precision cost.

### Holdout set
20% stratified holdout, never touched until the final evaluation. Used only to report final metrics and generate W&B plots.

### Feature importance
Uses `booster_.feature_importance(importance_type="gain")` — average gain per split across all trees. Gain is more interpretable than split count because it weights by how much each split actually reduces the loss.

---

## Expected Results

Based on the dataset (8,950 samples, PolitiFact-style) and what is typical for gradient boosting on tabular NLP data:

| Metric | LR baseline (approx) | RFC baseline (approx) | LGBM expectation |
|--------|----------------------|-----------------------|------------------|
| CV Macro F1 | ~0.60–0.64 | ~0.64–0.68 | **0.67–0.72** |
| Holdout Macro F1 | similar | similar | similar to CV |
| ROC-AUC | ~0.75–0.80 | ~0.78–0.82 | **0.80–0.85** |
| Training time (CV) | ~3–5 min | ~10–20 min | ~3–8 min |

Key signals to look for:
- **If LGBM >> RFC**: gradient boosting is clearly the right family for this data — invest in further HP tuning (more iterations, finer grid).
- **If LGBM ≈ RFC**: the bottleneck is likely the features, not the model — focus on better embeddings (`all-mpnet-base-v2`) or fine-tuned transformers.
- **If LGBM < RFC**: unusual; check that the HP search isn't stuck in a bad region. Try disabling HP search and using fixed defaults to diagnose.

### What to watch in W&B
- `cv_mean_macro_f1` — primary metric. Compare directly with the RFC run.
- `threshold/best` — expect a value in the 0.38–0.48 range (below 0.5) due to class imbalance.
- Feature importance top-10 — true-rate features (fe_speaker_true_rate, fe_subject_true_rate) should dominate if they work correctly.
- Fold-to-fold variance (`cv_std_macro_f1`) — high variance (> 0.03) means the model is sensitive to the split; consider more folds or more regularization.

---

## Potential Issues

| Issue | Likely cause | Fix |
|-------|-------------|-----|
| LightGBM stdout floods the console | `verbose` not propagating through `RandomizedSearchCV` | Add `callbacks=[lgb.log_evaluation(period=0)]` to fit call |
| `booster_` attribute not found after HP search (refit=True path) | `best_estimator_` wraps the fitted model correctly; `booster_` is available | Should work; if not, fall back to `model.feature_importances_` |
| OOF Macro F1 much lower than RFC | HP search space too wide for 20 iterations | Narrow the grid or increase `N_ITER_SEARCH` to 40 |
| `subsample` warning | LightGBM requires `bagging_freq > 0` to activate row subsampling | Add `bagging_freq=1` to the model constructor if warning appears |

---

## Next Steps After This Run

1. **If results are good**: run CatBoost (`cat.py`) with native categorical support — no OrdinalEncoder needed.
2. **Stacking**: combine LR + RFC + LGBM OOF probas as inputs to a meta-LR. Low effort, often +1–2 F1 points.
3. **Bigger embeddings**: swap `all-MiniLM-L6-v2` for `all-mpnet-base-v2` (768-dim) — same training code, potentially better text signal.
4. **Fine-tuned transformer**: DistilBERT or DeBERTa-v3-small on Kaggle GPU — highest potential ceiling.


----
- Why LightGBM — leaf-wise growth, gradient boosting vs bagging, and why the feature matrix is a good fit
  - Preprocessing decisions — full table of what's enabled and why (same as RFC, with rationale for each module)
  - Model decisions — class weighting, every HP in param_dist with the reasoning behind the range chosen, why max_depth is left unconstrained
  - Training strategy — nested CV, HP aggregation logic, threshold tuning, why gain-based feature importance
  - Expected results — a comparison table vs LR and RFC baselines, with guidance on how to interpret each outcome
  - What to watch in W&B — which metrics matter and what signals to look for
  - Potential issues — known LightGBM gotchas (subsample/bagging_freq, verbose in CV, booster_ access)
  - Next steps — what to do depending on the result
  ----

-- > Initial Results

[SECTION] Cross-validation summary  [total CV: 1012.6s]
  roc_auc: 0.6520 ± 0.0092
  pr_auc: 0.7601 ± 0.0086
  macro_f1: 0.5934 ± 0.0101
  f1: 0.7337 ± 0.0143
  precision: 0.7076 ± 0.0064
  recall: 0.7625 ± 0.0304
  accuracy: 0.6422 ± 0.0128
  mcc: 0.1917 ± 0.0209
  balanced_acc: 0.5918 ± 0.0095

[SECTION] Aggregating HP search results  [11:11:13]
  n_estimators             : [(500, 5)]  → chosen: 500
  learning_rate            : [(0.03, 5)]  → chosen: 0.03
  num_leaves               : [(31, 4), (127, 1)]  → chosen: 31
  subsample                : [(1.0, 4), (0.9, 1)]  → chosen: 1.0
  colsample_bytree         : [(0.7, 3), (0.9, 2)]  → chosen: 0.7
  min_child_samples        : [33, 33, 48, 48, 48]  → median: 48

  Final HP for fit: {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 48, 'subsample': 1.0, 'colsample_bytree': 0.7}

[SECTION] Threshold tuning on OOF predictions  [11:11:14]
   threshold   macro_f1
        0.20   0.4692
        0.22   0.4785
        0.24   0.4879
        0.26   0.5024
        0.28   0.5099
        0.30   0.5240
        0.32   0.5357
        0.34   0.5440
        0.36   0.5499
        0.38   0.5598
        0.40   0.5666
        0.42   0.5728
        0.44   0.5798
        0.46   0.5833
        0.48   0.5900
        0.50   0.5938
        0.52   0.5989
        0.54   0.6022
        0.56   0.6040
        0.58   0.6045  ←
        0.60   0.6034
        0.62   0.6031
        0.64   0.5965
        0.66   0.5934
        0.68   0.5924
        0.70   0.5884
        0.72   0.5803
        0.74   0.5718
        0.76   0.5618

  Best threshold: 0.58  (OOF macro_f1=0.6045)
  THRESHOLD updated: 0.50 → 0.58
[SECTION] Fitting final model on full train/val set  [11:11:14]
  Done in 1.7s
[SECTION] Evaluating on holdout set  [11:11:15]
  Using threshold: 0.58

Holdout results:
  roc_auc: 0.6681
  pr_auc: 0.7761
  macro_f1: 0.6062
  f1: 0.6874
  precision: 0.7420
  recall: 0.6402
  accuracy: 0.6229
  mcc: 0.2226
  balanced_acc: 0.6157

              precision    recall  f1-score   support

           0       0.47      0.59      0.52       631
           1       0.74      0.64      0.69      1159

    accuracy                           0.62      1790
   macro avg       0.61      0.62      0.61      1790
weighted avg       0.65      0.62      0.63      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                10846.2658
    fe_speaker_job_true_rate                            1610.0826
    fe_subject_true_rate                                1588.6335
    statement_original_vec_99                           844.4821
    statement_original_vec_158                          820.6256
    statement_original_vec_164                          808.2625
    statement_original_vec_119                          781.1787
    fe_party_true_rate                                  769.3869
    statement_original_vec_0                            749.9398
    statement_original_vec_30                           607.8661
    statement_original_vec_219                          591.1063
    statement_original_vec_204                          590.1436
    statement_original_vec_250                          581.1888
    statement_original_vec_35                           570.0092
    statement_upper_ratio                               558.9656
    statement_original_vec_291                          555.0499
    statement_original_vec_349                          541.9768
    statement_original_vec_61                           523.8668
    statement_original_vec_1                            513.7898
    statement_original_vec_97                           512.0350
    statement_original_vec_56                           498.9182
    statement_original_vec_245                          495.8607
    statement_original_vec_4                            488.3421
    statement_original_vec_105                          482.2455
    statement_original_vec_313                          481.6894
    statement_original_vec_361                          476.5933
    statement_original_vec_159                          470.3703
    statement_original_vec_188                          461.9494
    statement_original_vec_171                          453.7662
    statement_original_vec_63                           453.6468

---

## Analysis of Initial Results

### Overall verdict: below expectation, at or below RFC level

The predicted Macro F1 range was 0.67–0.72. The actual result was **0.5934 CV / 0.6062 holdout** — this puts LGBM in the "LGBM ≈ LR or below RFC" scenario flagged in the Expected Results section. The model is not capturing enough complexity to beat the tree ensemble baseline.

### HP convergence: the model wants to be simple

Every single fold independently chose `learning_rate=0.03` and 4 out of 5 chose `num_leaves=31` (the minimum in the grid). `min_child_samples` converged to a median of 48 — very high regularization for an 8,950-sample dataset (each leaf needs ≥48 samples to split, so roughly ≤180 leaves maximum in any tree). `subsample=1.0` means no row subsampling was used.

This is a strong signal: **the inner 3-fold CV consistently found that more complexity hurt more than it helped.** Gradient boosting with high complexity is overfitting on this feature space. The likely cause is the extreme dominance of `fe_speaker_true_rate` — after the model learns that signal, the residuals are small and noisy, and adding more leaves just fits noise.

### The threshold went the wrong direction

The journey doc predicted the threshold would land at 0.38–0.48 (below 0.5, to recover recall on the minority class). The actual best threshold was **0.58** — above 0.5. This means LightGBM's raw probabilities are already biased toward class 1 (false statements), so we need to raise the threshold to reduce false positives on class 1. The class weight of `{0: 1.42, 1: 0.77}` is not fully compensating for the majority-class bias in LightGBM's leaf-wise optimization.

### Feature importance: one feature dominates by 6.7×

| Feature | Gain |
|---------|------|
| `fe_speaker_true_rate` | 10,846 |
| `fe_speaker_job_true_rate` | 1,610 |
| `fe_subject_true_rate` | 1,588 |
| `fe_party_true_rate` | 769 |
| *Best embedding dim* | ~844 |
| `statement_upper_ratio` | 559 |

`fe_speaker_true_rate` has **6.7× the gain of the next feature**. The 4 true-rate features together dominate; the 384 embedding dimensions together contribute roughly the same total gain as the true-rate group. Individual metadata/FE features (interaction keys, lexical, sentiment, NER) barely appear in the top 30.

This reveals that gradient boosting is allocating most of its capacity to refining `fe_speaker_true_rate` splits. After that signal is exploited, the residuals are noisy and the model struggles to extract more from the embedding dims or categorical features.

### Why RFC likely handles this better

Random Forest randomly subsamples both rows and columns per tree, giving every feature a fair chance to contribute. When one feature dominates, RFC's subsampling forces trees to find alternatives, building ensemble diversity. LightGBM greedily picks the highest-gain split at each step, which means almost every early split in every tree is on `fe_speaker_true_rate`. The result is low diversity and limited benefit from the sequential correction mechanism.

### CV runtime: 1012s (~17 min)

Slower than the "few minutes" expectation. The bottleneck is the preprocessing (NER with spaCy), not LightGBM itself — the final fit on the full trainval took only **1.7 seconds**. Future runs should cache `df_processed` to avoid re-running NER.

---

## Next Experiments for LGBM

Given the analysis above, there are two directions worth trying before moving to CatBoost or stacking.

### Option A — Early stopping instead of fixed n_estimators

Replace the fixed `n_estimators` HP grid with a large cap (e.g. 2000) and use LightGBM's native early stopping on the inner CV validation fold. This allows the model to use as many trees as helpful without overfitting, and removes the need to search `n_estimators` explicitly.

```python
# In the fold training:
fold_model.fit(
    X_fold_train, y_fold_train,
    eval_set=[(X_fold_val, y_fold_val)],
    callbacks=[lgb.early_stopping(50), lgb.log_evaluation(period=0)],
)
```

This may find that only 100–200 trees are needed at learning_rate=0.1, with better performance than 500 trees at 0.03.

### Option B — Reduce true-rate feature dominance

The `fe_speaker_true_rate` monopoly is the core problem. Options:
- **Clip or log-transform**: apply `np.log1p` to gain-importance features before feeding them in (does not affect tree splits directly, but clipping to a max value could reduce the range the model can exploit).
- **Set `max_bin` lower for true-rate features**: LightGBM's histogram binning — reducing bins on these features limits their split resolution.
- **Drop `fe_speaker_true_rate` and retrain**: diagnostic run to see how much the other features contribute on their own.

### Option C — Expand the regularization search

The current grid didn't include `reg_alpha` (L1) and `reg_lambda` (L2). Adding these to `param_dist` might find a better regularization regime than relying solely on `min_child_samples` and `num_leaves`.

```python
param_dist = {
    ...
    "reg_alpha":   [0.0, 0.1, 0.5, 1.0],
    "reg_lambda":  [0.0, 0.1, 0.5, 1.0, 5.0],
}
```

### Recommended priority

1. **Option A (early stopping)** — lowest code effort, most likely to find better n_estimators/learning_rate pairing.
2. **Move to CatBoost** — CatBoost handles high-cardinality categoricals natively and uses ordered boosting to reduce overfitting. It may handle the `fe_speaker_true_rate` dominance more gracefully.
3. **Stacking** — even at current LGBM performance, its OOF probas are complementary to LR/RFC OOF probas for a stacking ensemble.

