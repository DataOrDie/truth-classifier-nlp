# Logistic Regression

---

### [2026-05-03] Logistic Regression — `src/training/lr.py`

**Goal:** Establish a strong LR baseline using the best preprocessing config found through individual sweeps.

---

#### Preprocessing decisions (from sweep results)

Each module was swept independently and the best config was locked in `bestflags.txt`.

| Module | Key decisions |
|---|---|
| Statement | TF-IDF (5,000 features, min_df=2, max_df=0.9), Porter stemmer, lexical features on, standardize |
| Subject | Primary strategy: most_frequent, grouped (rare→other), topic_count + is_rare, standardize |
| Speaker | Frequency + is_rare + grouped + title/comma flags, standardize |
| Speaker job | Length features + title/comma flags, no scaling |
| Party | All flags on (frequency, grouped, is_rare, flags, is_major, is_institutional), standardize |
| State | is_us_state + grouped + token_count + us_region, no scaling |

---

#### Feature engineering decisions

The FE module was swept in `lr_fe.py`. Sweep narrowed to **Interaction** and **Text-style** features only (Aggregate features excluded — low value for linear models).

| Feature | Decision | Reason |
|---|---|---|
| `fe_add_negation_count` | **Fixed True** | Established deception-detection signal |
| `fe_add_hedge_count` | **Fixed True** | Established deception-detection signal |
| `fe_add_absolutist_count` | **Fixed True** | Absolutist language correlates with false claims |
| `fe_add_numeral_count` | **Fixed True** | Specific numbers are a strong Politifact signal |
| `fe_add_speaker_job_subject` | **Fixed False** | 3-way combo, too high cardinality for LR |
| `fe_add_speaker_statement_len_bucket` | **Fixed False** | High cardinality, redundant with base speaker features |
| `fe_add_readability` | **True** (best config) | Swept — won |
| `fe_add_sentiment` | **True** (best config) | Swept — won |
| All interaction features | **False** (best config) | Swept — none helped |
| `fe_scale` | **standardize** | Swept — standardize won |

---

#### Model configuration

```
Model:         LogisticRegression (liblinear solver)
C:             1.0
Class weights: {0: 1.42,  1: 0.77}   ← from dataset imbalance (35% true, 65% false)
Max iter:      1000
CV:            5-fold stratified
Holdout:       20% stratified split
```

Total numeric features fed to the model: **~4,750** (dominated by ~4,700 TF-IDF word columns + ~50 from all other modules).

---

#### Results (holdout set, 1,790 samples)

| Metric | Value |
|---|---|
| ROC-AUC | 0.6539 |
| Macro F1 | 0.6042 |
| MCC | 0.2191 |
| Balanced accuracy | 0.6139 |
| Accuracy | 0.6207 |

**Per-class breakdown:**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| 0 — True statements | 0.47 | 0.59 | 0.52 |
| 1 — False statements | 0.74 | 0.64 | 0.68 |

**Main weakness:** Class 0 (true statements) is poorly predicted. Precision 0.47 means nearly half of all "true" predictions are wrong. The model is biased toward predicting false.

---

#### Next moves to try (in order)

1. **Threshold tuning** — lower decision threshold (e.g. 0.40) to improve class 0 recall. Zero code change, immediate test.
2. **Tune C + try L1 penalty** — L1 performs feature selection on TF-IDF. Try `C ∈ [0.01, 0.05, 0.1, 0.5]`.
3. **More aggressive class weights** — e.g. `{0: 2.0, 1: 0.7}` or `class_weight='balanced'`.
4. **True-rate features inside CV folds** — `fe_add_speaker_true_rate`, `fe_add_subject_true_rate`, `fe_add_party_true_rate`. These are the strongest Politifact-specific signals (a speaker's historical lie rate). Excluded so far due to leakage risk; requires computing them per fold.
5. **Bigrams** — change `statement_vectorizer_type = 'bigram'` to capture phrase patterns.

---

Looking at your results, the clearest signal is class 0 (true statements) is 
  weak — precision 0.47 means half your "true" predictions are wrong. Here are
  your options ranked by likely impact:                                         
   
  ---                                                                           
  1. True-rate features inside CV folds — highest impact for Politifact
  fe_add_speaker_true_rate, fe_add_subject_true_rate, fe_add_party_true_rate are
   currently excluded. A speaker's historical lie rate is the strongest single
  feature in this dataset. They're excluded from the global preprocessing
  because of leakage risk, but they're safe inside CV folds. This requires
  restructuring the training loop to compute them per-fold.

  2. Tune C + try L1 penalty — easy win
  L1 regularization does automatic feature selection on TF-IDF (zeroes out
  irrelevant word weights). Try sweeping:
  C_VALUE  = 0.1   # more regularization — often better for sparse TF-IDF
  # penalty = 'l1'  # add to LogisticRegression(...)
  Good values to try: C ∈ [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

  3. Adjust class weights — class 0 is undersupported
  Your current weights {0: 1.42, 1: 0.77} come from the prior distribution but
  may not be aggressive enough. Try class_weight='balanced' (sklearn computes it
   automatically) or bump class 0 manually: {0: 2.0, 1: 0.7}.

  4. Bigrams — capture phrase patterns
  Change statement_vectorizer_type = 'bigram'. "never raised taxes" as a phrase
  Your current weights {0: 1.42, 1: 0.77} come from the prior distribution but may not be aggressive enough. Try
  class_weight='balanced' (sklearn computes it automatically) or bump class 0 manually: {0: 2.0, 1: 0.7}.

  4. Bigrams — capture phrase patterns
  Change statement_vectorizer_type = 'bigram'. "never raised taxes" as a phrase is more signal than "never" + "raised" + "taxes"
  separately.

  5. Try a tree-based model — you already have trees.py
  LR is linear and can't learn interactions between features (e.g. "republican speaker + economic topic → likely false").
  XGBoost/LightGBM handle that natively and often outperform LR on tabular+text hybrid datasets like this one.

  6. Threshold tuning — quick post-hoc fix
  LR outputs probabilities. The default threshold is 0.5, but with imbalanced classes, lowering it (e.g. 0.35–0.40) shifts the
  model to predict class 0 more often, improving its recall. Add this after y_proba:
  THRESHOLD = 0.40
  y_pred = (y_proba >= THRESHOLD).astype(int)

  ---
  Recommended first steps in order:
  1. Threshold tuning — zero code change, immediate feedback
  2. Try C=0.1 with penalty='l1'
  3. Adjust class weights toward {0: 2.0, 1: 0.7}
  4. True-rate features inside CV (biggest potential, most work)

  ---

  DID -  Threshold tunning
  The flow now is:                                     
                                                                                          
  1. CV loop — collects OOF probabilities in oof_proba / oof_true alongside the normal fold metrics
  2. Threshold tuning — scans [0.20 … 0.75] in steps of 0.02, evaluates THRESHOLD_METRIC on OOF predictions (no holdout leakage),  
  prints the full grid table with the winner marked ←
  3. Holdout eval — uses the tuned THRESHOLD instead of the default 0.5                                                            
                                                                                                                                 
  Three knobs at the top of the config block:
  enable_threshold_tuning = True   # set False to skip entirely
  overwrite_threshold     = True   # set False to see results but keep THRESHOLD=0.5
  THRESHOLD_METRIC        = "macro_f1"  # or "mcc" | "balanced_acc"

  [SECTION] Evaluating on holdout set  [18:54:42]
  Using threshold: 0.46

Holdout results:
  roc_auc: 0.6533
  pr_auc: 0.7668
  macro_f1: 0.5958
  f1: 0.7107
  precision: 0.7167
  recall: 0.7049
  accuracy: 0.6285
  mcc: 0.1918
  balanced_acc: 0.5965

              precision    recall  f1-score   support

           0       0.47      0.49      0.48       631
           1       0.72      0.70      0.71      1159

    accuracy                           0.63      1790
   macro avg       0.60      0.60      0.60      1790
weighted avg       0.63      0.63      0.63      1790

-----------------------------------------------------------------
 DID - added gridsearch

 GridSearchCV runs 70 fits (7 × 2 × 5 folds) in parallel (n_jobs=-1), picks the winner, then the detailed CV loop evaluates only
  the best config. To disable the search and use fixed values, set enable_hp_search = False.

C and L1 penalty explained        

  What is regularization?                                                                                                          
   
  When a model trains, it can "memorize" the training data instead of learning general patterns — this is called overfitting.      
  Regularization is a penalty added during training that forces the model to stay simple and not rely too heavily on any single
  feature.

  C controls how strong that penalty is — but it's inverted:
  - Small C (e.g. 0.01) → strong penalty → simpler model → less overfitting
  - Large C (e.g. 10.0) → weak penalty → model can use features freely → more risk of overfitting

  With ~4,700 TF-IDF features and only 8,950 training samples, the model has a lot of words to work with and relatively few
  examples, so a stronger penalty (smaller C) usually helps.

  ---
  L2 vs L1 — what's the difference?

  Your model currently uses L2 (the default). Both penalize large weights, but differently:

  ┌────────────────────┬──────────────────────────────────┬─────────────────────────────────────────────────────┐
  │                    │           L2 (default)           │                         L1                          │
  ├────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ What it does       │ Shrinks all weights toward zero  │ Pushes many weights to exactly zero                 │
  ├────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Effect on features │ Keeps all features, just smaller │ Automatic feature selection — kills irrelevant ones │
  ├────────────────────┼──────────────────────────────────┼─────────────────────────────────────────────────────┤
  │ Good for           │ Dense features                   │ Sparse features like TF-IDF                         │
  └────────────────────┴──────────────────────────────────┴─────────────────────────────────────────────────────┘

  With TF-IDF you have thousands of word features — most are probably noise ("the", "and", rare political jargon). L1 zeroes those
  out entirely, leaving only the words that actually help. L2 just makes all of them smaller but keeps them all.

  ---
  In practice for your script

  # Current (L2, default)
  model = LogisticRegression(solver="liblinear", C=1.0, ...)

  # L1 with more regularization — good first test
  model = LogisticRegression(solver="liblinear", C=0.1, penalty="l1", ...)

  ▎ liblinear is the only solver that supports L1, which is why you're already using it — it's ready to go.

  ---
  What to actually try

  Run the script 3–4 times changing only C_VALUE and penalty:

  C_VALUE = 0.1  ;  penalty = "l1"   # most likely to help — try this first
  C_VALUE = 0.05 ;  penalty = "l1"
  C_VALUE = 0.5  ;  penalty = "l1"
  C_VALUE = 0.1  ;  penalty = "l2"   # compare L1 vs L2 at same C

  Watch macro F1 and class 0 precision in the holdout results — those are the numbers most likely to move.


----------------------------------------------------------------------------------------------

# kaggle

### [2026-05-03] Kaggle submission pipeline — `src/submit/kaggle-modulo.py`

**Problem:** The old `kaggle-modulo.py` was written for a completely different project (SBA loans, `Accept` column, `is_tree_model` logic) and would crash immediately on this dataset.

**What was built:**

1. **`lr.py` — save section extended** — after `save_model()`, two extra artifacts are now written to `models/lr/`:
   - `lr-vectorizer.joblib` — the TF-IDF vectorizer re-fit on `df_processed["statement_clean"]` with the exact same params (`max_features=5000`, `min_df=2`, `max_df=0.9`). Re-fitting on the same already-cleaned text produces an identical vocabulary to the one used during training.
   - `lr-threshold.joblib` — the final `THRESHOLD` value (post-tuning if `enable_threshold_tuning=True`).

2. **`kaggle-modulo.py` — full rewrite** — correct flow for this project:
   - Loads model + options + feature_names via `load_model`.
   - Loads `lr-threshold.joblib` (falls back to 0.5 if file missing).
   - Loads `lr-vectorizer.joblib` and injects it into `options.statement_fitted_vectorizer` — this forces `statement_ds.py` to call `.transform()` instead of `.fit_transform()` on test text, so the exact training vocabulary is used (no vocabulary mismatch).
   - Sets `options.label_option = "skip"` (test CSV has no label column; `"drop"` would crash with KeyError).
   - Runs `preprocess_one_step`, selects numeric columns, aligns to `feature_names` with `fill_value=0`.
   - Applies `model.predict_proba(X_test)[:, 1] >= threshold` for predictions.
   - Outputs `id, label` CSV (not `id, Accept`).

-----------------------------------------------------------------------------------------

# BIGRAMS

 4. Bigrams — capture phrase patterns
  Change statement_vectorizer_type = 'bigram'. "never raised taxes" as a phrase
  Your current weights {0: 1.42, 1: 0.77} come from the prior distribution but may not be aggressive enough. Try
  class_weight='balanced' (sklearn computes it automatically) or bump class 0 manually: {0: 2.0, 1: 0.7}.

  4. Bigrams — capture phrase patterns
  Change statement_vectorizer_type = 'bigram'. "never raised taxes" as a phrase is more signal than "never" + "raised" + "taxes"
  separately.

Unigrams (current — 'tfidf')                                                                                                     
                                                            
  Each word becomes its own feature. The sentence "he never raised taxes" produces:

  "he" → 1,  "never" → 1,  "raised" → 1,  "taxes" → 1

  The model sees four independent signals. It can learn that "never" correlates with false claims, but it has no idea "never
  raised" appeared together.

  ---
  Bigrams ('bigram')

  Features are individual words plus all adjacent two-word pairs. Same sentence:

  "he" → 1,  "never" → 1,  "raised" → 1,  "taxes" → 1,
  "he never" → 1,  "never raised" → 1,  "raised taxes" → 1

  The phrase "never raised taxes" is now a single feature. If politicians who make false claims disproportionately use that exact
  phrase, the model can learn it directly — rather than having to infer it from three separate word weights.

  ---
  Why it might help here

  Political misinformation has characteristic phrases: "lowest unemployment ever", "no new taxes", "100 percent voted for". These
  are more meaningful as units than their individual words. TF-IDF bigrams capture that.

  The tradeoff: feature count explodes. With max_features=5000 unigrams you have 5,000 columns. With bigrams and the same limit
  you're still capped at 5,000, but now those 5,000 slots are competed for by both single words and pairs — you may lose some
  important single words. You can raise statement_vectorizer_max_features to 10,000–15,000 to compensate.

  To try it, just change one line in lr.py:

  statement_vectorizer_type = 'bigram'
  statement_vectorizer_max_features = 10000  # optional: give it more room

  Holdout results:
  roc_auc: 0.6534
  pr_auc: 0.7643
  macro_f1: 0.6067
  f1: 0.7073
  precision: 0.7300
  recall: 0.6859
  accuracy: 0.6324
  mcc: 0.2153
  balanced_acc: 0.6100

              precision    recall  f1-score   support

           0       0.48      0.53      0.51       631
           1       0.73      0.69      0.71      1159

    accuracy                           0.63      1790
   macro avg       0.61      0.61      0.61      1790
weighted avg       0.64      0.63      0.64      1790


------------------------------------------------

DID: True-rate features inside CV folds — highest impact for Politifact                                      
    fe_add_speaker_true_rate, fe_add_subject_true_rate, fe_add_party_true_rate are                                                 
     currently excluded. A speaker's historical lie rate is the strongest single                                                   
    feature in this dataset. They're excluded from the global preprocessing                                                        
    because of leakage risk, but they're safe inside CV folds. This requires                                                       
    restructuring the training loop to compute them per-fold. for Logistic Regression 

    How the true-rate features work (leakage-free)                                                                                   
   
  The problem: if you compute "what fraction of Barack Obama's statements are false?" on all 8,950 rows, and then use that as a    
  feature, the model has seen the validation rows' labels — that's leakage.                                                      

  The fix: compute the rates only from the training fold, then apply them to the validation fold.

  ---
  Changes made

  lr.py — 6 locations:

  1. Config — enable_true_rate_features = True, true_rate_fallback = 0.5
  2. True-rate setup block (new section, runs after model config) — detects which grouped columns exist (speaker_grouped,
  subject_grouped, party_affiliation_grouped), adds placeholder columns (= 0.5) to X_trainval and X_holdout, and builds
  _grp_trainval/_grp_holdout (positional metadata frames for groupby).
  3. CV fold loop — each fold now: (a) computes speaker_grouped → mean(label) on the training fold only, (b) maps those rates to
  both the training fold and validation fold rows, (c) passes the enriched X_fold_train_raw into rebalance_training_data so
  oversampled rows carry the correct rates through.
  4. Final fit — computes rates from all of X_trainval, updates both X_trainval_final and X_holdout with the real values before
  fitting and evaluation.
  5. Save — writes lr-true-rate-maps.joblib containing the rate dicts + source column names + fallback value.
  6. W&B config — logs true_rate_features and true_rate_cols.

  kaggle-modulo.py:
  - Loads lr-true-rate-maps.joblib if it exists.
  - After preprocessing test data, maps each test row's speaker/subject/party to its training-set false-claim rate. Unseen groups
  (not in training data) get fallback = 0.5.