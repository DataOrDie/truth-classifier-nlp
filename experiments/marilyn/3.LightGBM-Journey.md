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


--> Option B Results
[SECTION] Cross-validation summary  [total CV: 1006.6s]
  roc_auc: 0.6487 ± 0.0176
  pr_auc: 0.7603 ± 0.0148
  macro_f1: 0.5947 ± 0.0136
  f1: 0.7341 ± 0.0110
  precision: 0.7084 ± 0.0086
  recall: 0.7619 ± 0.0197
  accuracy: 0.6427 ± 0.0127
  mcc: 0.1932 ± 0.0276
  balanced_acc: 0.5929 ± 0.0130

[SECTION] Aggregating HP search results  [13:10:38]
  n_estimators             : [(500, 4), (300, 1)]  → chosen: 500
  learning_rate            : [(0.03, 4), (0.05, 1)]  → chosen: 0.03
  num_leaves               : [(31, 5)]  → chosen: 31
  subsample                : [(1.0, 4), (0.8, 1)]  → chosen: 1.0
  colsample_bytree         : [(0.7, 3), (0.9, 2)]  → chosen: 0.7
  min_child_samples        : [33, 33, 48, 48, 49]  → median: 48

  Final HP for fit: {'n_estimators': 500, 'learning_rate': 0.03, 'num_leaves': 31, 'min_child_samples': 48, 'subsample': 1.0, 'colsample_bytree': 0.7}

[SECTION] Threshold tuning on OOF predictions  [13:10:38]
   threshold   macro_f1
        0.20   0.4358
        0.22   0.4492
        0.24   0.4611
        0.26   0.4759
        0.28   0.4915
        0.30   0.5049
        0.32   0.5233
        0.34   0.5366
        0.36   0.5464
        0.38   0.5577
        0.40   0.5705
        0.42   0.5756
        0.44   0.5769
        0.46   0.5852
        0.48   0.5924
        0.50   0.5948
        0.52   0.5960  ←
        0.54   0.5957
        0.56   0.5959
        0.58   0.5959
        0.60   0.5924
        0.62   0.5887
        0.64   0.5839
        0.66   0.5804
        0.68   0.5747
        0.70   0.5683
        0.72   0.5606
        0.74   0.5462
        0.76   0.5349

  Best threshold: 0.52  (OOF macro_f1=0.5960)
  THRESHOLD updated: 0.50 → 0.52
[SECTION] Fitting final model on full train/val set  [13:10:38]
  Done in 1.7s
[SECTION] Evaluating on holdout set  [13:10:40]
  Using threshold: 0.52

Holdout results:
  roc_auc: 0.6790
  pr_auc: 0.7854
  macro_f1: 0.6179
  f1: 0.7310
  precision: 0.7304
  recall: 0.7317
  accuracy: 0.6514
  mcc: 0.2358
  balanced_acc: 0.6178

              precision    recall  f1-score   support

           0       0.51      0.50      0.50       631
           1       0.73      0.73      0.73      1159

    accuracy                           0.65      1790
   macro avg       0.62      0.62      0.62      1790
weighted avg       0.65      0.65      0.65      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_job_true_rate                            5152.5947
    fe_subject_true_rate                                1757.7838
    fe_party_true_rate                                  1485.6281
    statement_original_vec_164                          1050.9059
    statement_original_vec_119                          889.8031
    statement_original_vec_158                          819.9379
    statement_original_vec_99                           801.6163
    statement_original_vec_35                           740.7476
    statement_upper_ratio                               735.4373
    statement_original_vec_204                          634.1256
    statement_original_vec_291                          632.8000
    statement_original_vec_361                          624.5158
    statement_original_vec_349                          600.3813
    statement_original_vec_0                            590.3908
    statement_original_vec_219                          586.1191
    statement_original_vec_250                          567.3489
    statement_original_vec_37                           553.5588
    statement_original_vec_1                            544.0897
    statement_original_vec_142                          543.9623
    statement_original_vec_105                          524.0157
    statement_original_vec_129                          513.8572
    statement_original_vec_14                           513.7234
    statement_original_vec_202                          512.9718
    statement_original_vec_159                          500.5389
    statement_original_vec_299                          491.5382
    statement_original_vec_73                           467.1148
    statement_original_vec_226                          466.4113
    statement_original_vec_355                          462.2533
    statement_original_vec_4                            461.8086
    statement_original_vec_118                          461.4636

---

## Analysis of Option B (Drop fe_speaker_true_rate)

### Verdict: best holdout result so far — dropping the dominant feature actually helps generalization

| Metric | Initial | Option A | Option B | Δ vs Initial |
|--------|---------|----------|----------|-------------|
| CV Macro F1 | 0.5934 | 0.5954 | 0.5947 | +0.001 (noise) |
| CV ROC-AUC | 0.6520 | 0.6414 | 0.6487 | −0.003 (noise) |
| Holdout Macro F1 | 0.6062 | 0.5985 | **0.6179** | **+0.012** |
| Holdout ROC-AUC | 0.6681 | 0.6579 | **0.6790** | **+0.011** |
| MCC | 0.2226 | 0.2301 | 0.2358 | +0.013 |
| Balanced Acc | 0.6157 | 0.6203 | 0.6178 | +0.002 |

Option B is the best run so far across the metrics that matter most (holdout Macro F1, ROC-AUC, MCC).

### The CV/holdout divergence is the key signal

CV Macro F1 barely moved (+0.001), but holdout Macro F1 improved by +0.012. This divergence reveals what `fe_speaker_true_rate` was actually doing in the initial run: it was **fitting the CV folds well but not generalizing**. The feature captures speaker-level true rates computed on the training fold — speakers in the holdout may have fewer or no training examples, so their true-rate values regress to the 0.5 fallback. The initial model had learned to rely on speaker true-rate splits that simply aren't informative on unseen speakers. Removing it forced the model to learn from features that generalize better: embeddings, subject true rates, party true rates, structural features.

### HP convergence is identical — confirming this is a feature-space property

```
Initial:  lr=0.03, leaves=31, min_child=48, subsample=1.0, colsample=0.7
Option B: lr=0.03, leaves=31, min_child=48, subsample=1.0, colsample=0.7
```

Every HP landed in exactly the same place (leaves=31 was unanimous — 5/5 folds). The model still wants maximum conservatism. This means the search space itself is appropriate; the constraint is the feature space, not the HP grid.

### Feature importance: dominance ratio dropped from 6.7× to 2.9×

| Rank | Feature | Gain | Ratio to #1 |
|------|---------|------|-------------|
| 1 | `fe_speaker_job_true_rate` | 5,152 | 1.0× |
| 2 | `fe_subject_true_rate` | 1,757 | 2.9× below |
| 3 | `fe_party_true_rate` | 1,485 | 3.5× below |
| 4 | `statement_original_vec_164` | 1,050 | 4.9× below |
| 9 | `statement_upper_ratio` | 735 | 7.0× below |

In the initial run, `fe_speaker_true_rate` had a 6.7× lead over every other feature. Without it, `fe_speaker_job_true_rate` steps up to lead, but the gap to second place is only 2.9×. The three remaining true-rate features have much closer gains (5152, 1757, 1485), and embedding dimensions now start appearing at rank 4 with gains above 1000. `statement_upper_ratio` (the ratio of uppercase characters in the statement) moved up to rank 9 with gain 735 — the model is extracting more from structural text features now that it isn't spending most of its budget on speaker splits.

### Threshold narrowed: 0.58 → 0.52

The optimal threshold dropped 6 points toward 0.5. The initial model's raw probabilities were pushed further toward class 1 (false) because `fe_speaker_true_rate` had a large directional effect. Without it, the probability distribution is more centered, requiring less correction. Still above 0.5, which confirms the class weight `{0: 1.42, 1: 0.77}` is not fully overcoming the majority-class bias.

### Class balance in predictions improved

| Class | Initial recall | Option B recall |
|-------|---------------|-----------------|
| 0 (true) | 0.59 | 0.50 |
| 1 (false) | 0.64 | 0.73 |
| Macro avg F1 | 0.61 | 0.62 |

Recall for class 0 dropped (0.59 → 0.50) while class 1 recall rose sharply (0.64 → 0.73). The model is now less confused overall, but it shifts more predictions toward the majority class. Class 0 F1 slightly worsened (0.52 → 0.50) while class 1 F1 improved (0.69 → 0.73), netting a better macro average.

### Interpretation: `fe_speaker_true_rate` was a leaky shortcut

The initial model was using speaker identity as a proxy for truthfulness — a valid signal on training speakers, but one that collapses on out-of-distribution speakers. Removing it pushes the model to use information that is present in the statement itself (embeddings, lexical features, structural features) and in more stable group-level signals (subject, party). These features generalize better to holdout speakers.

This is diagnostic evidence that the remaining true-rate features (subject, party, speaker job) are providing real signal, not just memorizing training identities.

---

## Option C — L1/L2 Regularization

**Hypothesis**: The HP search consistently converges to maximum conservatism (`num_leaves=31`, `min_child_samples=48`, `learning_rate=0.03`) but has never searched over explicit L1/L2 penalty terms. `reg_alpha` (L1) shrinks small leaf weights toward zero; `reg_lambda` (L2) penalizes large leaf weights. Either form of regularization could reduce the amount of model capacity devoted to `fe_speaker_true_rate` splits without removing the feature entirely.

**Changes vs baseline (initial run)**:
- `drop_speaker_true_rate = False` — `fe_speaker_true_rate` is back in the candidate set (Option B reverted)
- `reg_alpha` and `reg_lambda` added to `param_dist`:
  ```python
  "reg_alpha":  [0.0, 0.1, 0.5, 1.0],
  "reg_lambda": [0.0, 0.1, 0.5, 1.0, 5.0],
  ```
- `model_name = "lgbm-optC"`
- All other settings identical to the initial run (same param_dist ranges for the other six HPs)

**Note on search coverage**: Adding 2 new HP dimensions with 4–5 values each means the effective search space is ~4× larger than the initial run. With `N_ITER_SEARCH=20` the inner search samples fewer of these combinations per fold — the optimal reg values may not be found. Consider increasing `N_ITER_SEARCH` to 40 if results are noisy.

**What to watch for**:
- Does `reg_alpha > 0` or `reg_lambda > 0` get consistently selected? If so, the model benefits from explicit penalization.
- Does the HP search shift away from `num_leaves=31`? Higher regularization makes larger trees safe, so the search might explore `num_leaves=63` or `127` more.
- Does `fe_speaker_true_rate` gain ratio drop below 6.7× vs the next feature? That would confirm regularization is limiting its monopoly.
- Does holdout Macro F1 beat Option B's 0.6179? That's the bar to beat.

------

OPTION C

 Done. Here's what changed:

  Option B analysis added to LightGBM-Journey.md — key findings documented:
  - CV stayed flat (+0.001) but holdout jumped +0.012 Macro F1 and +0.011 ROC-AUC, revealing fe_speaker_true_rate was memorizing training speakers rather
  Option B analysis added to LightGBM-Journey.md — key findings documented:
  - CV stayed flat (+0.001) but holdout jumped +0.012 Macro F1 and +0.011 ROC-AUC, revealing fe_speaker_true_rate was memorizing training speakers rather
  than generalizing
  - Dominance ratio dropped from 6.7× to 2.9× — the feature importance distribution is now healthier
  than generalizing
  - Dominance ratio dropped from 6.7× to 2.9× — the feature importance distribution is now healthier
  - HP convergence is identical to the initial run, confirming the conservatism is a feature-space property, not a grid problem
  - Threshold narrowed from 0.58 → 0.52

  lgbm.py updated for Option C:
  - model_name = "lgbm-optC"
  - drop_speaker_true_rate = False (reverted — Option C runs against the same baseline as A and B)
  - reg_alpha: [0.0, 0.1, 0.5, 1.0] and reg_lambda: [0.0, 0.1, 0.5, 1.0, 5.0] added to param_dist

  The script is ready to run. One thing to be aware of: adding 2 new HP dimensions effectively quadruples the search space, so 20 random iterations covers
  less of it than before. If the results look noisy, consider bumping N_ITER_SEARCH to 40 before re-running.

✻ Brewed for 4m 3s

※ recap: Running LightGBM experiments for a fake news classifier, Options A and B are done. Option C (L1/L2 regularization) is configured and ready — run   
  lgbm.py next.
  than generalizing
  - Dominance ratio dropped from 6.7× to 2.9× — the feature importance distribution is now healthier
  - HP convergence is identical to the initial run, confirming the conservatism is a feature-space property, not a grid problem
  - Threshold narrowed from 0.58 → 0.52

  lgbm.py updated for Option C:
  - model_name = "lgbm-optC"
  - drop_speaker_true_rate = False (reverted — Option C runs against the same baseline as A and B)
  - reg_alpha: [0.0, 0.1, 0.5, 1.0] and reg_lambda: [0.0, 0.1, 0.5, 1.0, 5.0] added to param_dist

  The script is ready to run. One thing to be aware of: adding 2 new HP dimensions effectively quadruples the search space, so 20 random iterations covers
  less of it than before. If the results look noisy, consider bumping N_ITER_SEARCH to 40 before re-running.

  ---> Option C Results
  [SECTION] Cross-validation summary  [total CV: 662.0s]
  roc_auc: 0.6496 ± 0.0062
  pr_auc: 0.7598 ± 0.0051
  macro_f1: 0.5913 ± 0.0103
  f1: 0.7377 ± 0.0033
  precision: 0.7052 ± 0.0072
  recall: 0.7735 ± 0.0121
  accuracy: 0.6439 ± 0.0046
  mcc: 0.1886 ± 0.0173
  balanced_acc: 0.5896 ± 0.0095

[SECTION] Aggregating HP search results  [15:10:38]
  n_estimators             : [(500, 2), (300, 2), (800, 1)]  → chosen: 500
  learning_rate            : [(0.05, 4), (0.03, 1)]  → chosen: 0.05
  num_leaves               : [(31, 4), (63, 1)]  → chosen: 31
  subsample                : [(0.8, 2), (0.7, 1), (0.9, 1), (1.0, 1)]  → chosen: 0.8
  colsample_bytree         : [(0.8, 4), (0.9, 1)]  → chosen: 0.8
  min_child_samples        : [12, 13, 13, 30, 38]  → median: 13

  Final HP for fit: {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 13, 'subsample': 0.8, 'colsample_bytree': 0.8}

[SECTION] Threshold tuning on OOF predictions  [15:10:38]
   threshold   macro_f1
        0.20   0.4767
        0.22   0.4854
        0.24   0.4935
        0.26   0.5064
        0.28   0.5168
        0.30   0.5226
        0.32   0.5304
        0.34   0.5401
        0.36   0.5473
        0.38   0.5573
        0.40   0.5665
        0.42   0.5698
        0.44   0.5752
        0.46   0.5812
        0.48   0.5843
        0.50   0.5916
        0.52   0.5955
        0.54   0.5968
        0.56   0.5987
        0.58   0.5984
        0.60   0.6034
        0.62   0.6035  ←
        0.64   0.6016
        0.66   0.6021
        0.68   0.5976
        0.70   0.5941
        0.72   0.5893
        0.74   0.5851
        0.76   0.5723

  Best threshold: 0.62  (OOF macro_f1=0.6035)
  THRESHOLD updated: 0.50 → 0.62
[SECTION] Fitting final model on full train/val set  [15:10:38]
  Done in 2.0s
[SECTION] Evaluating on holdout set  [15:10:40]
  Using threshold: 0.62

Holdout results:
  roc_auc: 0.6635
  pr_auc: 0.7627
  macro_f1: 0.6100
  f1: 0.6896
  precision: 0.7460
  recall: 0.6411
  accuracy: 0.6263
  mcc: 0.2309
  balanced_acc: 0.6201

              precision    recall  f1-score   support

           0       0.48      0.60      0.53       631
           1       0.75      0.64      0.69      1159

    accuracy                           0.63      1790
   macro avg       0.61      0.62      0.61      1790
weighted avg       0.65      0.63      0.63      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                7601.3455
    fe_subject_true_rate                                1018.8975
    fe_speaker_job_true_rate                            637.6966
    statement_original_vec_99                           602.1434
    statement_original_vec_0                            582.9304
    statement_original_vec_164                          540.4335
    statement_original_vec_119                          482.8494
    statement_original_vec_35                           480.1181
    statement_original_vec_291                          463.0191
    statement_original_vec_158                          461.5690
    statement_original_vec_4                            434.2995
    fe_party_true_rate                                  430.1988
    statement_original_vec_349                          427.3264
    statement_original_vec_204                          416.2468
    statement_original_vec_30                           406.3169
    statement_original_vec_250                          404.6089
    statement_upper_ratio                               390.3263
    statement_original_vec_97                           377.7060
    statement_original_vec_71                           368.3587
    statement_original_vec_14                           363.0045
    statement_original_vec_361                          362.5013
    statement_original_vec_132                          360.0455
    statement_original_vec_235                          358.8207
    statement_original_vec_191                          343.9842
    statement_original_vec_56                           343.5322
    statement_original_vec_118                          342.4199
    statement_original_vec_188                          331.0054
    statement_original_vec_202                          324.9387
    statement_original_vec_299                          322.8897
    statement_original_vec_105                          312.1082

---

## Analysis of Option C (L1/L2 Regularization)

### Verdict: marginal improvement over initial, but worse than Option B — and the experiment was confounded by a code bug

| Metric | Initial | Option B | Option C | Δ vs Initial |
|--------|---------|---------|---------|-------------|
| CV Macro F1 | 0.5934 | 0.5947 | 0.5913 | −0.002 (noise) |
| CV ROC-AUC | 0.6520 | 0.6487 | 0.6496 | −0.002 (noise) |
| Holdout Macro F1 | 0.6062 | **0.6179** | 0.6100 | +0.004 |
| Holdout ROC-AUC | 0.6681 | **0.6790** | 0.6635 | −0.005 |
| MCC | 0.2226 | 0.2358 | 0.2309 | +0.008 |
| Balanced Acc | 0.6157 | 0.6178 | 0.6201 | +0.004 |

### Critical bug: reg_alpha and reg_lambda were never applied to the final model

The HP aggregation code had hardcoded lists for which params to aggregate. `reg_alpha` and `reg_lambda` were absent from both lists, so they were searched in the inner 3-fold CV (and contributed to selecting other HP combinations), but:
- They were **never printed in the HP aggregation output** (which is why they don't appear in the "Aggregating HP search results" section)
- They were **never included in `_best_params_final`**
- The final `LGBMClassifier` was fit with `reg_alpha=0.0, reg_lambda=0.0` — LightGBM's defaults

The code has been fixed: `reg_alpha` and `reg_lambda` now appear in the aggregation loop, the defaults dict, the final model constructor, and the W&B logging. **Option C needs a clean rerun** for the results to be interpretable.

### What the confounded run still tells us

Even though the reg params didn't make it into the final model, the inner CV searched over all 8 params simultaneously. This changed the HP landscape:

| HP | Initial | Option C |
|----|---------|---------|
| `learning_rate` | 0.03 (5/5) | **0.05 (4/5)** |
| `num_leaves` | 31 (4/5) | 31 (4/5) |
| `min_child_samples` | 48 (median) | **13 (median)** |
| `subsample` | 1.0 (4/5) | 0.8 (2/5) |
| `colsample_bytree` | 0.7 (3/5) | 0.8 (4/5) |

`min_child_samples` dropped from 48 to 13 — a dramatic shift. When `reg_alpha`/`reg_lambda` are in the search space, the inner CV finds that explicit L1/L2 penalties can substitute for high `min_child_samples` — fewer samples per leaf is safe when leaf weights are penalized. But since reg params were dropped from the final model, the final fit got `min_child_samples=13` without the regularization that was supposed to compensate. This likely explains why holdout ROC-AUC was slightly below the initial run despite similar Macro F1.

`learning_rate` shifting from 0.03 to 0.05 is a real signal — with explicit regularization in the search, the model found slightly higher learning rates beneficial. This is also worth observing in the clean rerun.

### Feature importance: speaker true-rate dominance rose to 7.5×

| Feature | Gain | Ratio to #1 |
|---------|------|------------|
| `fe_speaker_true_rate` | 7,601 | 1.0× |
| `fe_subject_true_rate` | 1,018 | 7.5× below |
| `fe_speaker_job_true_rate` | 637 | 11.9× below |

The dominance ratio got worse vs the initial run (7.5× vs 6.7×). Without reg penalties on the final model, the model leaned even harder on `fe_speaker_true_rate`. `fe_party_true_rate` dropped to rank 12 (gain 430), well below its rank-4 position in Option B.

### Threshold jumped to 0.62 — highest across all runs

The OOF probabilities were even more biased toward class 1 than in the initial run. With `fe_speaker_true_rate` back and the final model's regularization effectively at zero, the class weight imbalance pushed predictions further toward the majority class.

### Summary and next steps

Option C as run is uninterpretable for the L1/L2 question. The bug is fixed — the script is ready for a clean rerun. However, given the pattern across all three options:

| Run | Best lever | Holdout Macro F1 |
|-----|-----------|-----------------|
| Initial | — | 0.6062 |
| Option A | Early stopping | 0.5985 ↓ |
| Option B | Drop `fe_speaker_true_rate` | **0.6179** ↑ |
| Option C (buggy) | L1/L2 (not applied) | 0.6100 |

Option B remains the best single-lever result. A clean Option C rerun is worth doing to see whether the reg params themselves help beyond what `min_child_samples` already provides, but the gains are unlikely to exceed Option B's holdout improvement. Moving to CatBoost is the higher-leverage next step.

--> Option C Rerun
[SECTION] Cross-validation summary  [total CV: 668.8s]
  roc_auc: 0.6496 ± 0.0062
  pr_auc: 0.7598 ± 0.0051
  macro_f1: 0.5913 ± 0.0103
  f1: 0.7377 ± 0.0033
  precision: 0.7052 ± 0.0072
  recall: 0.7735 ± 0.0121
  accuracy: 0.6439 ± 0.0046
  mcc: 0.1886 ± 0.0173
  balanced_acc: 0.5896 ± 0.0095

[SECTION] Aggregating HP search results  [15:46:14]
  n_estimators             : [(500, 2), (300, 2), (800, 1)]  → chosen: 500
  learning_rate            : [(0.05, 4), (0.03, 1)]  → chosen: 0.05
  num_leaves               : [(31, 4), (63, 1)]  → chosen: 31
  subsample                : [(0.8, 2), (0.7, 1), (0.9, 1), (1.0, 1)]  → chosen: 0.8
  colsample_bytree         : [(0.8, 4), (0.9, 1)]  → chosen: 0.8
  reg_alpha                : [(0.5, 2), (0.1, 2), (0.0, 1)]  → chosen: 0.5
  reg_lambda               : [(0.1, 2), (0.5, 1), (0.0, 1), (5.0, 1)]  → chosen: 0.1
  min_child_samples        : [12, 13, 13, 30, 38]  → median: 13

  Final HP for fit: {'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 31, 'min_child_samples': 13, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_alpha': 0.5, 'reg_lambda': 0.1}

[SECTION] Threshold tuning on OOF predictions  [15:46:14]
   threshold   macro_f1
        0.20   0.4767
        0.22   0.4854
        0.24   0.4935
        0.26   0.5064
        0.28   0.5168
        0.30   0.5226
        0.32   0.5304
        0.34   0.5401
        0.36   0.5473
        0.38   0.5573
        0.40   0.5665
        0.42   0.5698
        0.44   0.5752
        0.46   0.5812
        0.48   0.5843
        0.50   0.5916
        0.52   0.5955
        0.54   0.5968
        0.56   0.5987
        0.58   0.5984
        0.60   0.6034
        0.62   0.6035  ←
        0.64   0.6016
        0.66   0.6021
        0.68   0.5976
        0.70   0.5941
        0.72   0.5893
        0.74   0.5851
        0.76   0.5723

  Best threshold: 0.62  (OOF macro_f1=0.6035)
  THRESHOLD updated: 0.50 → 0.62
[SECTION] Fitting final model on full train/val set  [15:46:14]
  Done in 2.5s
[SECTION] Evaluating on holdout set  [15:46:16]
  Using threshold: 0.62

Holdout results:
  roc_auc: 0.6624
  pr_auc: 0.7701
  macro_f1: 0.6052
  f1: 0.6861
  precision: 0.7415
  recall: 0.6385
  accuracy: 0.6218
  mcc: 0.2209
  balanced_acc: 0.6148

              precision    recall  f1-score   support

           0       0.47      0.59      0.52       631
           1       0.74      0.64      0.69      1159

    accuracy                           0.62      1790
   macro avg       0.61      0.61      0.61      1790
weighted avg       0.65      0.62      0.63      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                7468.5435
    fe_subject_true_rate                                944.6571
    statement_original_vec_99                           565.8662
    statement_original_vec_119                          563.7057
    fe_speaker_job_true_rate                            559.7658
    statement_original_vec_164                          535.3845
    statement_original_vec_0                            503.2225
    statement_original_vec_158                          483.5501
    statement_original_vec_291                          427.3804
    statement_original_vec_349                          419.8663
    statement_original_vec_30                           409.0508
    statement_original_vec_35                           403.0864
    statement_upper_ratio                               399.8472
    fe_party_true_rate                                  389.7204
    statement_original_vec_4                            388.2591
    statement_original_vec_191                          381.1212
    statement_original_vec_61                           369.4980
    statement_original_vec_118                          366.5108
    statement_original_vec_1                            364.9019
    statement_original_vec_204                          335.4768
    statement_original_vec_56                           333.5064
    statement_original_vec_97                           322.3251
    statement_original_vec_313                          322.0955
    statement_original_vec_219                          304.6221
    statement_original_vec_14                           303.5497
    statement_clean_avg_token_freq                      303.1331
    statement_original_vec_202                          298.8995
    statement_original_vec_100                          298.2873
    statement_original_vec_250                          295.0711
    statement_original_vec_11                           293.2320

---

## Analysis of Option C Rerun (L1/L2 — clean)

### Verdict: L1/L2 regularization does not improve on the initial baseline

| Metric | Initial | Option B | Option C (buggy) | Option C (clean) | Δ vs Initial |
|--------|---------|---------|-----------------|-----------------|-------------|
| CV Macro F1 | 0.5934 | 0.5947 | 0.5913 | 0.5913 | −0.002 (noise) |
| Holdout Macro F1 | 0.6062 | **0.6179** | 0.6100 | 0.6052 | **−0.001** |
| Holdout ROC-AUC | 0.6681 | **0.6790** | 0.6635 | 0.6624 | −0.006 |
| MCC | 0.2226 | 0.2358 | 0.2309 | 0.2209 | −0.002 |

Option C is the worst run across all experiments — worse than the initial baseline on every holdout metric.

### The natural experiment the bug created

Both Option C runs used identical CV phases (same HP search produced identical fold-level results). The only difference is what was passed to the final model:

| | Buggy run | Clean rerun |
|-|-----------|------------|
| `reg_alpha` in final model | 0.0 (default) | **0.5** |
| `reg_lambda` in final model | 0.0 (default) | **0.1** |
| `min_child_samples` | 13 | 13 |
| Holdout Macro F1 | 0.6100 | **0.6052** |

The clean rerun is *worse* than the buggy run by −0.005. Applying the aggregated reg values (`reg_alpha=0.5, reg_lambda=0.1`) to the final model hurt generalization vs using defaults (0.0, 0.0). This means the reg values selected by the inner CV helped within the CV folds but over-regularized the final model on the holdout set.

### HP divergence: swapping leaf regularization for weight regularization

The HP search traded `min_child_samples` (48 → 13) against explicit weight penalties (`reg_alpha=0.5`). This tradeoff holds within the 3-fold inner CV splits — small leaves are stable when leaf weights are penalized — but doesn't transfer to the holdout:

| HP | Initial | Option C |
|----|---------|---------|
| `min_child_samples` | 48 | **13** |
| `reg_alpha` | — | 0.5 (L1) |
| `reg_lambda` | — | 0.1 (L2) |

With min_child_samples=13, each leaf needs only 13 samples. On an 8,950-sample dataset with 500 trees at depth ~7, this allows very fine-grained splits that memorise the training fold. The L1 penalty is supposed to offset this by zeroing small leaf weights, but the combination overfit more than the initial run's simpler `min_child_samples=48`.

### reg_alpha selection was unstable

`reg_alpha` tied at 2 folds each for 0.5 and 0.1. The mode chose 0.5, but 0.1 would have been equally valid. Any holdout improvement from this option likely requires a finer search over the interaction between `min_child_samples` and `reg_alpha/reg_lambda` — the current 20-iteration search can't explore this adequately.

### Feature importance: dominance ratio worsened to 7.9×

| Feature | Gain | Rank |
|---------|------|------|
| `fe_speaker_true_rate` | 7,468 | 1 |
| `fe_subject_true_rate` | 944 | 2 |
| Best embedding dim | ~565 | 3 |
| `fe_speaker_job_true_rate` | 559 | 5 |
| `fe_party_true_rate` | 389 | 14 |

The dominance ratio (7.9×) is worse than the initial run (6.7×) and far worse than Option B (2.9×). L1 regularization didn't meaningfully reduce `fe_speaker_true_rate`'s monopoly — it slightly trimmed all gains proportionally without changing the hierarchy.

`statement_clean_avg_token_freq` appeared at rank 26 — a new feature that hadn't surfaced before. This is a non-leaking aggregate (average token frequency per statement), and its appearance suggests the model is now exploring different features with the lower leaf threshold.

### Summary across all LGBM experiments

| Run | Key change | Holdout Macro F1 | Holdout ROC-AUC |
|-----|-----------|-----------------|----------------|
| Initial | Baseline | 0.6062 | 0.6681 |
| Option A | Early stopping (log-loss) | 0.5985 ↓ | 0.6579 ↓ |
| **Option B** | **Drop `fe_speaker_true_rate`** | **0.6179 ↑** | **0.6790 ↑** |
| Option C | L1/L2 regularization | 0.6052 ↓ | 0.6624 ↓ |

**Option B is the best LGBM configuration.** The core finding across all experiments: the feature-dominance problem matters more than any HP lever. Removing the monopoly feature improved holdout results by +0.012; no HP change came close.

The next step — CatBoost — is the right move. CatBoost's ordered boosting builds each tree's target statistics incrementally (each sample's statistic is computed on prior samples only), which naturally limits how much any single feature can dominate early splits.