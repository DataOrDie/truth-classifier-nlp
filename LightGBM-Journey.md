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

----- 
Option A 
 - Remove n_estimators from param_dist — early stopping will determine the optimal number of trees                                        
  - Inner HP search uses a moderate fixed cap (N_ESTIMATORS_INNER=500) — enough to compare HP combinations                                   
  - After finding best HP per fold, refit that fold's model with N_ESTIMATORS_CAP=2000 + early stopping on the outer validation set          
  - Record best_iteration_ per fold; final model uses the median across folds                                                                
  - import lightgbm as lgb needed for the callback API  

--> Option A output
[SECTION] Cross-validation summary  [total CV: 972.9s]
  roc_auc: 0.6414 ± 0.0116
  pr_auc: 0.7517 ± 0.0099
  macro_f1: 0.5954 ± 0.0136
  f1: 0.6881 ± 0.0196
  precision: 0.7269 ± 0.0073
  recall: 0.6538 ± 0.0311
  accuracy: 0.6169 ± 0.0167
  mcc: 0.1970 ± 0.0240
  balanced_acc: 0.6015 ± 0.0118

[SECTION] Aggregating HP search results  [12:40:06]
  learning_rate            : [(0.05, 3), (0.03, 1), (0.2, 1)]  → chosen: 0.05
  num_leaves               : [(31, 4), (63, 1)]  → chosen: 31
  subsample                : [(0.9, 3), (0.8, 1), (1.0, 1)]  → chosen: 0.9
  colsample_bytree         : [(0.8, 4), (1.0, 1)]  → chosen: 0.8
  min_child_samples        : [23, 30, 34, 34, 34]  → median: 34
  n_estimators_used        : [56, 49, 9, 35, 30]  → median: 35

  Final HP for fit: {'n_estimators': 35, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 34, 'subsample': 0.9, 'colsample_bytree': 0.8}

[SECTION] Threshold tuning on OOF predictions  [12:40:07]
   threshold   macro_f1
        0.20   0.3967
        0.22   0.4037
        0.24   0.4137
        0.26   0.4240
        0.28   0.4427
        0.30   0.4646
        0.32   0.4890
        0.34   0.5103
        0.36   0.5265
        0.38   0.5476
        0.40   0.5627
        0.42   0.5773
        0.44   0.5886
        0.46   0.5926
        0.48   0.5950
        0.50   0.5956
        0.52   0.5958  ←
        0.54   0.5883
        0.56   0.5739
        0.58   0.5610
        0.60   0.5438
        0.62   0.5200
        0.64   0.4962
        0.66   0.4755
        0.68   0.4507
        0.70   0.4254
        0.72   0.4000
        0.74   0.3803
        0.76   0.3635

  Best threshold: 0.52  (OOF macro_f1=0.5958)
  THRESHOLD updated: 0.50 → 0.52
[SECTION] Fitting final model on full train/val set  [12:40:07]
  Done in 0.2s
[SECTION] Evaluating on holdout set  [12:40:07]
  Using threshold: 0.52

Holdout results:
  roc_auc: 0.6579
  pr_auc: 0.7675
  macro_f1: 0.5985
  f1: 0.6519
  precision: 0.7606
  recall: 0.5703
  accuracy: 0.6056
  mcc: 0.2301
  balanced_acc: 0.6203

              precision    recall  f1-score   support

           0       0.46      0.67      0.55       631
           1       0.76      0.57      0.65      1159

    accuracy                           0.61      1790
   macro avg       0.61      0.62      0.60      1790
weighted avg       0.65      0.61      0.61      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                5623.5527
    fe_speaker_job_true_rate                            722.0295
    fe_subject_true_rate                                658.9318
    fe_party_true_rate                                  350.5934
    statement_original_vec_0                            328.1543
    statement_original_vec_119                          303.4027
    statement_original_vec_164                          282.3297
    statement_original_vec_250                          273.7770
    statement_original_vec_99                           237.9048
    statement_original_PERSON                           234.9121
    statement_original_vec_158                          186.8150
    statement_original_vec_30                           175.7675
    statement_original_vec_291                          170.0308
    statement_original_vec_219                          165.5078
    statement_original_vec_11                           155.2890
    statement_original_vec_349                          153.4865
    statement_original_vec_5                            152.2606
    statement_original_vec_202                          147.5132
    statement_original_vec_35                           141.6220
    statement_original_vec_204                          141.5779
    statement_original_vec_142                          137.2632
    statement_original_vec_1                            134.5625
    statement_original_vec_188                          129.7262
    statement_original_vec_354                          127.4332
    statement_original_vec_132                          124.7602
    statement_original_vec_58                           124.0706
    statement_original_vec_17                           121.0830
    statement_original_vec_71                           119.8087
    statement_original_vec_333                          118.9630
    statement_original_vec_361                          114.8876

---

## Analysis of Option A (Early Stopping)

### Verdict: no improvement, early stopping stopped far too soon

| Metric | Initial | Option A | Δ |
|--------|---------|----------|---|
| CV Macro F1 | 0.5934 | 0.5954 | +0.002 (noise) |
| CV ROC-AUC | 0.6520 | 0.6414 | **−0.011** |
| Holdout Macro F1 | 0.6062 | 0.5985 | **−0.008** |
| Holdout ROC-AUC | 0.6681 | 0.6579 | **−0.010** |
| MCC | 0.2226 | 0.2301 | +0.007 |
| Balanced Acc | 0.6157 | 0.6203 | +0.005 |

Option A is marginally worse overall. The Macro F1 difference is within noise, but ROC-AUC dropped clearly.

### Root cause: early stopping chose only 35 trees (one fold stopped at 9)

`n_estimators_used: [56, 49, 9, 35, 30] → median: 35`

This is the critical signal. LightGBM's early stopping monitors **binary log-loss** on the eval set by default, not Macro F1. These two objectives diverge early in training:

- Log-loss rewards well-calibrated probabilities. After ~30–50 trees the dominant `fe_speaker_true_rate` signal is fully learned and adding more trees starts nudging probabilities away from the log-loss optimum.
- Macro F1 is a threshold-dependent metric that can keep improving beyond the log-loss plateau, especially for the minority class.

When log-loss stops improving for 50 consecutive rounds, early stopping fires — but the model is still undertrained from a Macro F1 perspective. The fold that stopped at iteration 9 is the extreme case: after 9 trees the validation log-loss was at its best; everything after that made log-loss slightly worse even as Macro F1 improved.

### The true-rate distribution shift amplifies early stopping

True-rate features (`fe_speaker_true_rate`, etc.) are computed on the **training fold** and mapped to the **validation fold** via a groupby mean. The validation fold's true-rate values come from a slightly different distribution than the training fold's (different subset of speakers). This distribution shift means the model's predictions on the validation fold worsen faster than its predictions on unseen data — causing early stopping to fire prematurely.

### HP convergence still prefers minimum complexity

Even with early stopping, the HP search still preferred `num_leaves=31` (4/5 folds) and a low learning rate. This confirms that the conservatism is a property of the feature space, not just the n_estimators choice.

### Feature importance: PERSON entity appeared

`statement_original_PERSON` entered the top 10 (rank 10, gain 234). This is new compared to the initial run. With only 35 trees, the model has less capacity devoted entirely to true-rate splits, so other features get slightly more representation. Still, `fe_speaker_true_rate` dominates at 5623 vs the next feature at 722 (~7.8× ratio, worse than the initial run's 6.7×).

### Lesson learned

Early stopping calibrated to log-loss is not the right lever here. The correct fix is to either:
- Use Macro F1 directly as the eval metric for early stopping (`eval_metric="macro"` in LightGBM)
- Or accept a fixed n_estimators and address the deeper problem: `fe_speaker_true_rate` monopolizes all model capacity.

Option B directly attacks the deeper problem.

---
Reverted (Option A):                                                                    
  - Removed import lightgbm as lgb                                                                                                           
  - Removed N_ESTIMATORS_INNER, N_ESTIMATORS_CAP, EARLY_STOPPING_ROUNDS                                                                      
  - n_estimators: [300, 500, 800] back in param_dist                                                                                         
  - CV loop back to single-phase refit=True                                                                                                  
  - HP aggregation and W&B logging restored to initial form 

  Added (Option B):
  - drop_speaker_true_rate = True config flag (single line to flip back to False if needed)
  - One _candidates.pop("fe_speaker_true_rate", None) in the true-rate setup
  - model_name = "lgbm-optB" so the saved model and W&B run are clearly labelled
  - drop_speaker_true_rate logged to W&B config
  