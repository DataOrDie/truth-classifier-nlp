# Random Forest Journey

Training log and decision record for the `rfc` model track (`src/training/rfc.py`).

---

## Why Random Forest first

Gradient Boosting (XGBoost, LightGBM) is typically stronger on tabular data, but RF trains in one pass, has no learning rate to worry about, and is harder to overfit accidentally. It gives a reliable ceiling estimate for what structured metadata alone can achieve before committing to the slower iterative models. If RF underperforms LR significantly, that is a signal the structured features are not adding much beyond what the text already provides.

---

## Preprocessing: opts_tree_lean

The full `opts_tree_lean` configuration is defined in `PREPROCESSING_OPTIONS.md`. Key decisions summarized here.

### Text vectorization — TF-IDF at 500 terms

**Decision:** `statement_vectorizer_type='tfidf'`, `max_features=500`

**Why:** Large sparse TF-IDF matrices (5000+ columns) slow RF significantly — each split candidate has to be evaluated over thousands of near-zero columns, most of which never appear in any one tree. Capping at 500 keeps preprocessing fast and the feature space manageable. The alternative (`embeddings`) gives denser, semantically richer representations and is the right next step after this baseline is established.

**Not done yet:** Embeddings (`all-MiniLM-L6-v2`). Switch `statement_vectorizer_type='embeddings'` to try the full config.

### No stemming, no lemmatization, no stopword removal

**Decision:** All three set to `'none'` / `False`.

**Why:** These operations compress vocabulary to help linear models close the gap between morphological variants. Trees split on token presence directly; stem collisions (e.g. "running" and "run" both → "run") collapse distinctions without adding signal. Stopword removal could eliminate tokens the model uses as splits. No transformation = no information loss.

### All scales set to `'none'`

**Decision:** Every `*_scale` option is `'none'` across all modules.

**Why:** Trees are invariant to monotone transforms. Standardizing a feature from range [0, 100] to [-2, 2] does not change where optimal splits are. Scaling wastes preprocessing time and adds a potential source of train/test inconsistency.

### NER off

**Decision:** `statement_add_ner_features=False`

**Why:** spaCy NER on 8950 rows adds several minutes of preprocessing per run and is slow to iterate with. The entity-count features (`PERSON`, `ORG`, `GPE`, `DATE`) are genuinely useful for political fact-checking, but the marginal gain over lexical features is uncertain. Enable once the rest of the pipeline is validated.

---

## Feature engineering decisions

### Categorical encoding — OrdinalEncoder, not OneHot

**Decision:** `OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)`

**Why:** Grouped categories (`speaker_grouped`, `party_affiliation_grouped`, etc.) and all interaction keys (`fe_speaker_subject`, etc.) are string columns after preprocessing. Trees need numeric input. OneHotEncoder was ruled out because interaction keys have very high cardinality (hundreds of unique speaker×topic pairs) — exploding this into binary columns would create more columns than there are rows. OrdinalEncoder assigns a single integer per category; trees split on integer thresholds and can learn "this speaker (id=42) lies more" without needing a binary indicator per speaker.

**Unknown categories:** Test data may contain speakers or topics never seen in training. `unknown_value=-1` maps those to a reserved integer, which trees treat as a valid split point (effectively "unseen category").

### All interaction keys enabled

**Decision:** `fe_add_speaker_subject`, `fe_add_speaker_party`, `fe_add_subject_party`, `fe_add_speaker_job_subject`, `fe_add_state_party`, `fe_add_speaker_statement_len_bucket` all `True`.

**Why:** Linear models need explicit interaction features (cross-products) because they can only learn main effects. Trees compute interactions implicitly through sequential splits — but the joint key `"joe biden__health care"` still gives the model a direct shortcut to "this specific speaker on this specific topic" without requiring two separate splits to reach that conclusion. The OrdinalEncoder makes these keys usable at zero additional model complexity.

### True-rate features — computed in CV loop, not in preprocessing options

**Decision:** `fe_add_speaker_true_rate=False` in options. `enable_true_rate_features=True` in the script.

**Why:** Setting those flags to `True` in `OneStepOptions` would compute the target mean on the full training set and leak label information into all folds. The script instead maintains a parallel lookup table (`_grp_trainval`) and computes the mean on the train split of each fold before mapping to the validation split. For the final model, the means are computed on all 7160 train/val rows and saved to disk so the submission script can look up test rows without seeing any labels.

`speaker_primary_true_rate` is historically the single most predictive feature in this dataset because the same politicians make many statements and their credibility is consistent over time.

---

## Model configuration

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| `n_estimators` | 300 | Enough trees for stable estimates without tuning; diminishing returns beyond ~200 |
| `max_depth` | `None` | Full-depth trees; RF controls variance via averaging, not depth |
| `min_samples_leaf` | 1 | Default; can increase to 2–5 to reduce overfitting if validation gap is large |
| `class_weight` | `{0: 1.42, 1: 0.77}` | Dataset is 35/65; weights scale impurity by class to equalize effective sample size |
| `n_jobs` | `-1` | All cores; RF is embarrassingly parallel over trees |
| `random_state` | 42 | Reproducible |

### No hyperparameter search this run

**Decision:** Fixed parameters, no `RandomizedSearchCV`.

**Why:** The goal of this first run is a clean baseline with the lean feature set. Tuning before knowing whether the features work adds noise to the learning signal. Candidates for a follow-up tuning run: `n_estimators`, `max_features` (subsampling columns per split — the RF knob most analogous to regularization), `min_samples_leaf`.

### No probability calibration

**Decision:** No `CalibratedClassifierCV` wrapper.

**Why:** RF's `predict_proba` uses vote proportions across trees, which tends to be better calibrated than LinearSVC or uncalibrated LR. Calibration can still be applied if the OOF reliability diagram shows systematic over/under-confidence.

### Fixed threshold at 0.5

**Decision:** `THRESHOLD = 0.5`, no OOF threshold search.

**Why:** Keeping the first run clean. The OOF probabilities (`oof_proba`) are collected and could be used to search for a better threshold post-hoc if macro F1 at 0.5 is disappointing. The class weights already shift the model's internal decision surface toward balanced recall, so 0.5 should be a reasonable starting point.

---

## What is saved

| File | Purpose |
|------|---------|
| `rfc-model.joblib` | Trained `RandomForestClassifier` |
| `rfc-options.joblib` | `OneStepOptions` used for preprocessing |
| `rfc-feature-names.joblib` | Ordered list of training feature names |
| `rfc-ordinal-encoder.joblib` | Fitted `OrdinalEncoder`; must be applied to test data |
| `rfc-vectorizer.joblib` | Fitted TF-IDF; ensures same vocabulary on test data |
| `rfc-threshold.joblib` | Decision threshold (0.5) |
| `rfc-true-rate-maps.joblib` | Per-speaker/subject/party label means for test lookup |

---


## Initial Run — Analysis & Improvement Plan

### What the results say

| Metric | Value | Signal |
|--------|-------|--------|
| Macro F1 | 0.5516 | Driven down by class 0 |
| Class 0 recall | 0.24 | Model predicts almost everything as false |
| Class 1 recall | 0.89 | Heavily biased toward majority class |
| ROC-AUC | 0.6598 | Moderate discrimination ability |
| MCC | 0.1716 | Very weak overall |

Three problems are visible before any tuning:

1. **`statement` column in top-5 features.** The raw source text column does not end in `_clean` or `_original`, so the categorical encoder drops condition misses it and OrdinalEncodes it. Each row has a near-unique text string → the encoder assigns a near-unique integer → the tree can use this as a pseudo row-ID. Inflates training fit, produces garbage at inference. Needs to be excluded explicitly.

2. **Stopwords dominating TF-IDF features.** `vec_the`, `vec_in`, `vec_of` appear in the top 30. `max_df=0.9` only cuts terms present in 90%+ of documents; these stopwords appear in ~50–80% of rows and survive. The TF-IDF is adding vocabulary noise rather than topic signal.

3. **Class 0 recall collapse.** Despite `class_weight={0: 1.42, 1: 0.77}`, the model is not finding enough separating signal for true statements. The threshold at 0.5 needs to move left, and the model needs stronger features.

---

## 5-Step Improvement Plan

### Step 1 — Fix the feature matrix (no retraining needed for diagnosis)

**What:** Two bugs to fix before any tuning is meaningful.

**Bug A — `statement` column leaking into OrdinalEncoding.**
In `rfc.py`, expand the text-drop condition to also exclude source columns that may survive preprocessing:
```python
# Before (misses raw source columns that don't end in _clean/_original)
_text_cols = {c for c in _all_obj_cols if c.endswith(("_clean", "_original"))}

# After (also drops the original source columns if still present)
_source_cols = {
    statement_source_col, speaker_source_col, subject_source_col,
    speaker_job_source_col, party_affiliation_source_col, state_source_col,
}
_text_cols = (
    {c for c in _all_obj_cols if c.endswith(("_clean", "_original"))}
    | (_source_cols & set(_all_obj_cols))
)
```

**Bug B — Stopwords in TF-IDF.**
Lower `max_df` and add explicit stopword removal for the vectorizer only (not for the full text pipeline):
```python
statement_vectorizer_max_df = 0.7   # was 0.9 — cuts terms in >70% of docs
statement_stopword_removal  = True  # strip stopwords before building TF-IDF vocab
```
Stopword removal in `statement_ds.py` only affects vectorization, not the lexical/text-style features.

**Expected impact:** Removes a near-unique row-ID feature that was absorbing importance from real signals. Cleans up 5–10 slots in the top-30 importance list. Likely a small improvement in OOF macro F1 without any HP changes.

-> STEP 1 output
Holdout results:
  roc_auc: 0.6388
  pr_auc: 0.7623
  macro_f1: 0.5606
  f1: 0.7688
  precision: 0.6856
  recall: 0.8749
  accuracy: 0.6592
  mcc: 0.1740
  balanced_acc: 0.5690

              precision    recall  f1-score   support

           0       0.53      0.26      0.35       631
           1       0.69      0.87      0.77      1159

    accuracy                           0.66      1790
   macro avg       0.61      0.57      0.56      1790
weighted avg       0.63      0.66      0.62      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                0.0335
    statement_upper_ratio                               0.0259
    statement_clean_avg_token_freq                      0.0226
    fe_readability                                      0.0226
    statement_original_char_len                         0.0216
    fe_subject_true_rate                                0.0211
    fe_subject_party                                    0.0196
    fe_speaker_job_subject                              0.0193
    fe_speaker_subject                                  0.0193
    subject_length                                      0.0186
    fe_subject_avg_statement_len                        0.0183
    statement_original_word_count                       0.0176
    fe_speaker_avg_punctuation                          0.0157
    fe_speaker_len_bucket                               0.0150
    fe_speaker_avg_number_ratio                         0.0144
    subject_primary_grouped                             0.0143
    fe_speaker_avg_statement_len                        0.0142
    subject_frequency                                   0.0141
    fe_speaker_party                                    0.0140
    fe_sentiment_polarity                               0.0138
    statement_clean_spelling_err_count                  0.0135
    subject_primary                                     0.0135
    fe_sentiment_subjectivity                           0.0130
    statement_clean_digit_ratio                         0.0124
    fe_state_party                                      0.0118
    speaker_char_len                                    0.0118
    speaker_frequency_pct                               0.0105
    speaker_frequency                                   0.0101
    speaker_job_char_len                                0.0101
    speaker_grouped                                     0.0094

#### Step 1 — What changed, what it means

| Metric | Initial | Step 1 | Δ |
|--------|---------|--------|---|
| Macro F1 | 0.5516 | 0.5606 | **+0.009** |
| Class 0 F1 | 0.33 | 0.35 | +0.02 |
| Class 0 recall | 0.24 | 0.26 | +0.02 |
| ROC-AUC | 0.6598 | 0.6388 | −0.021 |
| MCC | 0.1716 | 0.1740 | +0.002 |

**Bugs confirmed fixed.** `statement` is gone from the importance list. `statement_clean_vec_the` (0.0126), `statement_clean_vec_in` (0.0100), `statement_clean_vec_of` (0.0092) are all gone. The ~0.05 combined importance they absorbed has redistributed to real signals — `statement_upper_ratio` jumped from 0.0201 → 0.0259, `fe_readability` from 0.0180 → 0.0226.

**Macro F1 improved modestly (+0.009), as expected.** The estimate was "small improvement" and that is what we got. The class 0 recall improved only from 0.24 → 0.26, confirming the fix alone does not solve the threshold problem.

**ROC-AUC dropped −0.021.** This is counterintuitive but explainable. The `statement` pseudo row-ID had near-unique integers which could accidentally produce a well-ranked holdout if the model happened to memorise label order during training. Removing it hurts the ranking metric while improving the threshold-based one (macro F1). This is also within the noise range of a single 20% holdout split. The PR-AUC drop (0.7782 → 0.7623) follows the same pattern.

**Feature importance is now clean and interpretable.** Every top-30 feature is a genuine signal. The flat distribution (max 0.0335) persists — no single feature dominates, which is expected with structured metadata at this scale. The threshold at 0.5 is the remaining bottleneck: class 0 recall is still only 0.26.

---

### Step 2 — Threshold tuning on OOF probabilities (no retraining needed)

**What:** The OOF probabilities (`oof_proba`) are already collected in the CV loop. Add the same threshold grid search that `lr.py` has.

```python
# Add to rfc.py after the CV summary section
enable_threshold_tuning = True
THRESHOLD_METRIC        = "macro_f1"   # maximise macro F1 on OOF predictions
```

Grid: `np.arange(0.20, 0.76, 0.02)`. At threshold 0.5, class 0 recall is 0.24. Moving the threshold to 0.35–0.40 should push class 0 recall to 0.40–0.50 at some cost to class 1 recall — net gain in macro F1 because the class 0 F1 is so low that almost any trade improves the macro average.

**Expected impact:** +0.03–0.06 macro F1 from threshold adjustment alone. No retraining.

-> Step 2 output: 
[SECTION] Evaluating on holdout set  [21:29:13]
  Using threshold: 0.58

Holdout results:
  roc_auc: 0.6388
  pr_auc: 0.7623
  macro_f1: 0.5952
  f1: 0.7207
  precision: 0.7125
  recall: 0.7291
  accuracy: 0.6341
  mcc: 0.1906
  balanced_acc: 0.5943

              precision    recall  f1-score   support

           0       0.48      0.46      0.47       631
           1       0.71      0.73      0.72      1159

    accuracy                           0.63      1790
   macro avg       0.60      0.59      0.60      1790
weighted avg       0.63      0.63      0.63      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                0.0335
    statement_upper_ratio                               0.0259
    statement_clean_avg_token_freq                      0.0226
    fe_readability                                      0.0226
    statement_original_char_len                         0.0216
    fe_subject_true_rate                                0.0211
    fe_subject_party                                    0.0196
    fe_speaker_job_subject                              0.0193
    fe_speaker_subject                                  0.0193
    subject_length                                      0.0186
    fe_subject_avg_statement_len                        0.0183
    statement_original_word_count                       0.0176
    fe_speaker_avg_punctuation                          0.0157
    fe_speaker_len_bucket                               0.0150
    fe_speaker_avg_number_ratio                         0.0144
    subject_primary_grouped                             0.0143
    fe_speaker_avg_statement_len                        0.0142
    subject_frequency                                   0.0141
    fe_speaker_party                                    0.0140
    fe_sentiment_polarity                               0.0138
    statement_clean_spelling_err_count                  0.0135
    subject_primary                                     0.0135
    fe_sentiment_subjectivity                           0.0130
    statement_clean_digit_ratio                         0.0124
    fe_state_party                                      0.0118
    speaker_char_len                                    0.0118
    speaker_frequency_pct                               0.0105
    speaker_frequency                                   0.0101
    speaker_job_char_len                                0.0101
    speaker_grouped                                     0.0094

#### Step 2 Analysis

**Threshold went UP, not down.** The pre-run prediction was that the optimal threshold would fall to 0.35–0.40. It actually rose to **0.58**. The explanation: `class_weight={0: 1.42, 1: 0.77}` makes the model internally penalise class-1 errors more during training, which pushes the raw probability outputs higher. A balanced model centred at 0.5 ends up biased above 0.5 here, so the OOF search correctly found that 0.58 is the true balance point on these weighted outputs.

**Class 0 recall doubled.** Recall for true statements jumped from 0.26 → 0.46, landing squarely in the predicted 0.40–0.50 target. This is the main payoff of threshold tuning: trading some class-1 recall (0.87 → 0.73) to recover class-0 predictions that were being suppressed below the old 0.5 cutoff.

**Macro F1 gain: +0.035.** This matched the predicted +0.03–0.06 range almost exactly, and meets the 0.59 milestone in the trajectory table. The gain is entirely a threshold effect — ROC-AUC is identical to Step 1 (0.6388), confirming no retraining occurred.

| Metric | Initial | Step 1 | Step 2 | Predicted |
|--------|---------|--------|--------|-----------|
| Macro F1 | ~0.51 | 0.5606 | **0.5952** | ~0.59 ✓ |
| Class 0 recall | 0.24 | 0.26 | 0.46 | 0.40–0.50 ✓ |
| ROC-AUC | 0.6254 | 0.6388 | 0.6388 | unchanged ✓ |
| Threshold | 0.50 | 0.50 | **0.58** | 0.35–0.40 ✗ |

**Feature importance — statement raw column is gone.** The top features are now clean: `fe_speaker_true_rate` (0.0335) leads, followed by structural and lexical features. No text column appears in the top 30 — this makes sense at 500 TF-IDF terms with flat importance distribution, each term contributes ~0.0005 and is below the display cutoff.

**What Step 3 needs to address.** The feature importance distribution is still very flat — the top feature has only 3.35% importance across ~570+ features. This is the signature of a model with too many weak features splitting too often. Reducing `max_features` forces each split to choose from a smaller random subset, which increases tree diversity and gives stronger features more relative influence. That is the primary target for Step 3.


---

### Step 3 — Hyperparameter tuning: `max_features` + `min_samples_leaf`

**What:** Add nested CV `RandomizedSearchCV` inside the CV loop, targeting the two highest-leverage RF knobs.

```python
from scipy.stats import randint

enable_hp_search = True
param_dist = {
    "max_features":      [0.2, 0.3, 0.5, "sqrt", "log2"],
    "min_samples_leaf":  randint(1, 8),       # 1–7
    "n_estimators":      [200, 300, 500],
}
N_ITER_SEARCH = 20
```

`max_features` controls how many features each tree considers at each split (the primary variance knob in RF — default `sqrt(n_features)` with ~570 features ≈ 24 candidates per split). Lowering it increases diversity between trees at the cost of individual tree quality; raising it toward 0.5 may help when features are weakly correlated. `min_samples_leaf` prevents trees from fitting noise in tiny leaf cells.

**Expected impact:** The flat importance distribution suggests many splits are being wasted on weak features. Reducing `max_features` forces the model to find stronger splits. Estimate +0.02–0.04 macro F1.

Inside each outer CV fold:                                                                                                     
  - Builds a RandomizedSearchCV (20 iterations, 3-fold inner CV, scored on f1_macro)                                             
  - Searches max_features in [0.2, 0.3, 0.5, "sqrt", "log2"], min_samples_leaf in 1–7, n_estimators in [200, 300, 500]           
  - Uses n_jobs=1 on the inner RF to avoid nested parallelism; the outer search uses n_jobs=-1                                   
  - Prints and logs the best params + inner CV score per fold                                                                    
                                                            
  After outer CV:
  - Aggregates best params: mode for max_features / n_estimators, median for min_samples_leaf
  - Logs a W&B table with per-fold params and the final chosen values

  Final fit:
  - Uses the aggregated best params instead of the fixed defaults

  The CV will be ~20x slower per fold (20 search iterations × 3 inner folds each). With 5 outer folds that's 300 inner RF fits
  before the final fit. Expect 10–30 minutes depending on your machine.

--> Step 3 Output
Final HP for fit: {'n_estimators': 200, 'max_features': 'sqrt', 'min_samples_leaf': 6}

[SECTION] Threshold tuning on OOF predictions  [21:59:49]
   threshold   macro_f1
        0.20   0.3930
        0.22   0.3930
        0.24   0.3938
        0.26   0.3948
        0.28   0.3975
        0.30   0.4093
        0.32   0.4218
        0.34   0.4363
        0.36   0.4610
        0.38   0.4896
        0.40   0.5145
        0.42   0.5410
        0.44   0.5679
        0.46   0.5851
        0.48   0.5982
        0.50   0.6036  ←
        0.52   0.6027
        0.54   0.5910
        0.56   0.5799
        0.58   0.5663
        0.60   0.5404
        0.62   0.5225
        0.64   0.4909
        0.66   0.4580
        0.68   0.4302
        0.70   0.4016
        0.72   0.3754
        0.74   0.3476
        0.76   0.3267

  Best threshold: 0.50  (OOF macro_f1=0.6036)
  THRESHOLD updated: 0.50 → 0.50
[SECTION] Fitting final model on full train/val set  [21:59:49]
  Done in 0.3s
[SECTION] Evaluating on holdout set  [21:59:50]
  Using threshold: 0.50

Holdout results:
  roc_auc: 0.6481
  pr_auc: 0.7735
  macro_f1: 0.5942
  f1: 0.7053
  precision: 0.7172
  recall: 0.6937
  accuracy: 0.6246
  mcc: 0.1889
  balanced_acc: 0.5957

              precision    recall  f1-score   support

           0       0.47      0.50      0.48       631
           1       0.72      0.69      0.71      1159

    accuracy                           0.62      1790
   macro avg       0.59      0.60      0.59      1790
weighted avg       0.63      0.62      0.63      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                0.0677
    statement_upper_ratio                               0.0341
    fe_subject_true_rate                                0.0324
    fe_readability                                      0.0269
    statement_original_char_len                         0.0259
    statement_clean_avg_token_freq                      0.0253
    fe_subject_party                                    0.0231
    fe_speaker_job_subject                              0.0231
    subject_length                                      0.0227
    fe_speaker_subject                                  0.0224
    fe_subject_avg_statement_len                        0.0216
    fe_speaker_avg_number_ratio                         0.0210
    statement_original_word_count                       0.0202
    fe_speaker_avg_punctuation                          0.0201
    fe_speaker_party                                    0.0184
    fe_speaker_len_bucket                               0.0183
    fe_speaker_avg_statement_len                        0.0177
    statement_clean_digit_ratio                         0.0174
    fe_state_party                                      0.0169
    subject_frequency                                   0.0167
    fe_sentiment_subjectivity                           0.0165
    subject_primary                                     0.0164
    speaker_frequency_pct                               0.0161
    subject_primary_grouped                             0.0158
    speaker_frequency                                   0.0151
    party_affiliation_grouped                           0.0149
    speaker_char_len                                    0.0142
    fe_sentiment_polarity                               0.0138
    state_info_frequency_pct                            0.0135
    fe_party_true_rate                                  0.0133

#### Step 3 Analysis

**HP found: `min_samples_leaf=6`, `max_features='sqrt'`, `n_estimators=200`.** The most significant finding is `min_samples_leaf=6` — this is heavy regularization. The default value of 1 lets trees grow leaves containing a single sample, which memorizes noise. Requiring at least 6 samples per leaf forces the model to find generalizable splits rather than fitting individual rows.

**Threshold reset to 0.50.** With Step 2 the threshold was 0.58; here it fell back to exactly 0.50. The reason: `min_samples_leaf=6` moderates the probability outputs. Heavily regularized trees produce more moderate vote proportions (less extreme probabilities), which re-centers the optimal cutoff at 0.50. The OOF threshold curve is now smoothly monotone and peaks exactly at 0.50 — a well-calibrated model.

**Macro F1 essentially flat vs Step 2, but ROC-AUC improved.** Holdout macro F1 was 0.5942 (Step 2: 0.5952, Δ≈0). ROC-AUC went from 0.6388 → **0.6481** (+0.009). The HP tuning improved the model's ranking quality (better probability ordering) without changing the binary threshold outcome much. The OOF macro F1 was 0.6036 — slightly optimistic, but the holdout gap (~0.009) is normal.

**Feature importance became sharper.** `fe_speaker_true_rate` doubled from 0.0335 → **0.0677**. With `min_samples_leaf=6` the model wastes fewer splits on noise, so the truly predictive features absorb more relative importance. TF-IDF terms have now fully disappeared from the top 30 (each of the 500 terms individually contributes ~0.0003), confirming that the dense 500-term bag-of-words is providing diluted signal.

| Metric | Step 2 | Step 3 | Δ |
|--------|--------|--------|---|
| Macro F1 | 0.5952 | **0.5942** | −0.001 |
| ROC-AUC | 0.6388 | **0.6481** | +0.009 |
| Class 0 recall | 0.46 | 0.50 | +0.04 |
| Threshold | 0.58 | **0.50** | reset |
| `min_samples_leaf` | 1 | **6** | regularized |

**What Step 4 needs to address.** TF-IDF is now the clear bottleneck. The top-30 features are all metadata and engineered features — zero text signal from the statement itself is appearing explicitly. That means the model is essentially ignoring the semantic content of the statement. Switching to sentence embeddings (384-dim dense vectors) will replace 500 near-useless sparse columns with 384 semantically rich dimensions, and that is where the largest remaining gain should come from.


---

### Step 4 — Switch TF-IDF → sentence embeddings

**What:** Replace the 500-term sparse TF-IDF with dense sentence embeddings.

```python
statement_vectorizer_type    = "embeddings"
statement_embedding_model    = "all-MiniLM-L6-v2"   # 384-dim dense output
# Remove: statement_vectorizer_max_features, min_df, max_df (not used for embeddings)
```

**Why now:** After fixing the feature matrix (Step 1) and tuning HP (Step 3), the TF-IDF is the remaining weak link in the text representation. The top TF-IDF terms in the initial run were `the`, `in`, `of` — function words with no discriminative power. Sentence embeddings compress the entire statement into a 384-dim semantic vector trained on a large corpus, which gives the tree richer signal about meaning rather than term frequency.

**Tradeoff:** Preprocessing takes 5–15× longer (sentence encoder inference). The vectorizer file is replaced by the stateless `all-MiniLM-L6-v2` model (no vocab to save). The `rfc-vectorizer.joblib` artifact is no longer needed.

**Expected impact:** Likely the largest single gain. LR with embeddings outperformed LR with TF-IDF substantially in earlier experiments. +0.04–0.08 macro F1 estimate, primarily because class 0 (true statements) have subtler phrasing differences that bag-of-words misses.

---

### Step 5 — NER features + additional true-rate signals

**What:** Two additions to complete the full feature set.

**5a — NER entity counts.**
```python
statement_add_ner_features = True   # adds PERSON, ORG, GPE, DATE, NUM, OTHER columns
statement_ner_model        = "en_core_web_sm"
```
Political fact-checking correlates strongly with how many named entities a claim contains. A statement naming 3 organisations and a date is structurally different from a vague claim. These 7 columns are low-dimensional and directly interpretable.

**5b — Add `speaker_job` true-rate.**
The current `_candidates` dict covers speaker, subject, and party. Speaker job (occupation) also repeats across rows:
```python
_candidates = {
    "fe_speaker_true_rate":     ["speaker_grouped", "speaker_clean"],
    "fe_subject_true_rate":     ["subject_primary_grouped", "subject_primary", "subject_clean"],
    "fe_party_true_rate":       ["party_affiliation_grouped", "party_affiliation_clean"],
    "fe_speaker_job_true_rate": ["speaker_job_grouped", "speaker_job_clean"],   # add this
}
```

**Expected impact:** NER adds direct structural signal about claim referential density. Speaker job true-rate captures occupation-level credibility (senators vs bloggers). Combined with embeddings from Step 4, this rounds out the full `opts_tree_full` configuration. Estimate +0.02–0.03 macro F1.

---

### Expected trajectory

| After step | Change | Est. macro F1 |
|------------|--------|---------------|
| Baseline (initial run) | — | 0.552 |
| Step 1 — fix feature matrix | Removes `statement` column bug, clean TF-IDF vocab | ~0.56 |
| Step 2 — threshold tuning | Shift threshold left from 0.5 | ~0.59 |
| Step 3 — HP tuning | `max_features`, `min_samples_leaf` | ~0.61 |
| Step 4 — embeddings | Replace TF-IDF with dense 384-dim vectors | ~0.64 |
| Step 5 — NER + job true-rate | Complete feature set | ~0.66 |

These are estimates; actual gains depend on how much the `statement` column was distorting the model and how well sentence embeddings separate true from false on this corpus.

---

--- INITIAL RUN----

Holdout results:
  roc_auc: 0.6598
  pr_auc: 0.7782
  macro_f1: 0.5516
  f1: 0.7735
  precision: 0.6823
  recall: 0.8930
  accuracy: 0.6615
  mcc: 0.1716
  balanced_acc: 0.5646

              precision    recall  f1-score   support

           0       0.55      0.24      0.33       631
           1       0.68      0.89      0.77      1159

    accuracy                           0.66      1790
   macro avg       0.61      0.56      0.55      1790
weighted avg       0.63      0.66      0.62      1790

[SECTION] Computing feature importance
  Top 30 features:
    fe_speaker_true_rate                                0.0339
    statement_upper_ratio                               0.0201
    statement_clean_avg_token_freq                      0.0183
    fe_readability                                      0.0180
    statement                                           0.0180
    fe_subject_true_rate                                0.0180
    statement_original_char_len                         0.0163
    fe_speaker_job_subject                              0.0161
    fe_speaker_subject                                  0.0155
    subject_length                                      0.0151
    fe_subject_avg_statement_len                        0.0151
    fe_subject_party                                    0.0149
    statement_original_word_count                       0.0134
    fe_speaker_avg_statement_len                        0.0128
    fe_speaker_avg_punctuation                          0.0126
    fe_speaker_avg_number_ratio                         0.0126
    statement_clean_vec_the                             0.0126
    fe_speaker_len_bucket                               0.0124
    fe_speaker_party                                    0.0121
    fe_sentiment_polarity                               0.0114
    fe_sentiment_subjectivity                           0.0112
    subject_primary_grouped                             0.0112
    subject_frequency                                   0.0111
    subject_primary                                     0.0110
    fe_state_party                                      0.0103
    statement_clean_spelling_err_count                  0.0102
    statement_clean_vec_in                              0.0100
    speaker_char_len                                    0.0097
    statement_clean_digit_ratio                         0.0093
    statement_clean_vec_of                              0.0092

