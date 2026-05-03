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