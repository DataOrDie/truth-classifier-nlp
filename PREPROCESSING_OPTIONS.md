# Preprocessing Options Reference

This document describes every configurable option in `OneStepOptions` (declared in `src/preprocessing/one_step.py`). Options are grouped by the module that owns them. Each entry notes what column(s) the flag produces, what changes in behavior, and which model families benefit.

**Model family legend used throughout this document:**

| Tag | Means |
|-----|-------|
| `[linear]` | Linear models — LogisticRegression, LinearSVC, Ridge, Lasso |
| `[tree]` | Tree-based models — RandomForest, GradientBoosting, XGBoost, LightGBM |
| `[nlp-classical]` | Bag-of-words, TF-IDF, n-gram pipelines |
| `[dl]` | Dense neural networks, LSTM, CNN-text |
| `[transformer]` | Fine-tuned BERT/RoBERTa-style models, sentence transformers |
| `[all]` | Beneficial across all families |

---

## Pipeline Execution Order

`preprocess_one_step` runs modules in this fixed sequence. Each module receives the output of the previous one.

```
label → id → subject → statement_ds → speaker → speaker_job → party_affiliation → state → feature_engineering
```

Downstream modules (e.g. `feature_engineering`) depend on `*_clean` columns produced by all earlier modules. Do not skip an upstream module if a downstream flag references its output.

---

## Shared Patterns

Several patterns repeat across modules; they are documented once here.

### `keep_original: bool` (default varies)

When `True`, the raw source column is preserved under its original name alongside the `*_clean` output. When `False`, the raw column is dropped after cleaning.

- Keep it `True` during exploration (easy comparison), `False` for production feature matrices to avoid feeding raw strings to models.
- `statement_ds.py` always writes `statement_original` regardless of this flag (needed for NER and `fe_proper_noun_count`).

### `rare_threshold: int` and `group_rare: bool`

Rows where a categorical value appears fewer than `rare_threshold` times in the training data are flagged as rare (`add_*_is_rare`) and optionally replaced with a single `rare_label` string (`add_grouped_*`). `group_rare=True` requires the corresponding `add_grouped_*` flag to also be `True`.

Collapsing rare categories prevents high-cardinality label-encoding from creating spurious features. Threshold tuning: start at 5, increase to 10–20 for very high cardinality columns like `speaker`.

### `scale: str` — `'none'` | `'standardize'` | `'normalize'`

Scales **numeric** output columns only. Binary flags (0/1) and string columns are never scaled.

- `'standardize'` (z-score): preferred for `[linear]` models — centers and unit-variances the feature.
- `'normalize'` (min-max to [0,1]): useful when you need a bounded range, e.g. feeding a neural network alongside embeddings.
- `'none'` (default): correct choice for `[tree]` models, which are invariant to monotone transforms.

### `*_primary_true_rate` flags — leakage risk

Any flag ending in `_true_rate` computes the empirical mean of the label column grouped by a categorical column. **This is target encoding and will leak label information if computed on the full training set.** Always compute these features inside cross-validation folds: fit on the training fold, map to the validation fold.

---

## `label.py`

Controls what happens to the `label` column.

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `label_option` | `'skip'` \| `'drop'` | `'skip'` | `'skip'` passes the column through. `'drop'` removes it — use for inference on unlabeled test data. |
| `label_source_col` | str | `'label'` | Source column name. |

> **Note:** Class distribution is 35.25% true (0) / 64.75% false (1). The pipeline does not add class weights — set `class_weight={0: 1.42, 1: 0.77}` in your classifier.

---

## `id.py`

Controls what happens to the `id` column.

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `id_option` | `'drop'` \| `'hash_bucket'` | `'drop'` | `'drop'` removes the id entirely. `'hash_bucket'` hashes the id into an integer bucket in `[0, n_buckets)` and adds it as `id_bucket`. |

`'hash_bucket'` is useful as a proxy for dataset provenance or temporal ordering if the ids were assigned sequentially. Produces a stable, deterministic split signal.

- `[tree]`: `id_bucket` can help a tree discover dataset structure not captured elsewhere.
- `[linear]` / `[transformer]`: ignore — adds no signal.

---

## `statement_ds.py`

The most complex module. Owns text cleaning, token-level normalization, lexical/pollution features, optional NER, and vectorization. All options are prefixed `statement_` in `OneStepOptions`.

**Always produces:**
- `statement_original` — repair-only, original casing (always written; used by NER and `fe_proper_noun_count`)
- `statement_clean` — the fully processed text string

### Text Cleaning

| Option | Default | Effect |
|--------|---------|--------|
| `statement_lower` | `True` | Lowercases the text. Turn off only when passing to case-sensitive models or NER. `[transformer]` models handle case natively and may perform better with mixed case. |
| `statement_remove_html` | `True` | Strips HTML tags. Safe to leave on. |
| `statement_remove_urls` | `True` | Replaces `http://...` and `www....` patterns with a space. |
| `statement_repair_polluted_statements` | `True` | Splits on tabs/newlines and keeps only the first segment. Strips leading/trailing quotes. Mitigates CSV column-spillover artifacts in this dataset. Leave on. |
| `statement_remove_punctuation` | `False` | Strips all punctuation from tokens. Useful for pure bag-of-words `[nlp-classical]` pipelines. Harmful for `[transformer]` models that rely on punctuation for syntax. |
| `statement_replace_numbers` | `False` | Replaces digit sequences with a `<NUM>` token (configurable via `statement_number_token`). Reduces vocabulary size; trades specificity for generalization. Good for `[nlp-classical]` and `[linear]` with TF-IDF. Bad for `[transformer]` — let the tokenizer handle numbers. |

### Stopword and Morphological Normalization

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `statement_stopword_removal` | bool | `False` | Removes NLTK English stopwords before vectorization. Reduces noise for `[nlp-classical]` / `[linear]` TF-IDF. Hurts `[transformer]` and `[dl]` pipelines — those models learn stopword semantics. Requires NLTK. |
| `statement_keep_negations` | bool | `True` | When stopword removal is on, preserves "no", "not", "never", "nor", "n't". Almost always keep `True` — removing negations destroys sentence polarity. |
| `statement_stemmer` | `'none'` \| `'porter'` \| `'snowball'` | `'none'` | Applies morphological stemming. Reduces vocabulary; helps `[nlp-classical]` on small datasets. Porter is aggressive; Snowball is slightly softer. Both hurt `[transformer]` inputs — do not use with embeddings. Requires NLTK. |
| `statement_lemmatizer` | `'none'` \| `'wordnet'` | `'none'` | Applies WordNet lemmatization. Softer than stemming — maps "running" → "run" but preserves more information. Prefer over stemming when using `[linear]` + TF-IDF. Requires NLTK. |

> **Practical rule:** For `[linear]` / `[nlp-classical]`, enabling stopword removal + lemmatizer + TF-IDF bigram is a strong baseline. For `[transformer]`, keep all three at their defaults (`False`, `True`, `'none'`).

### Lexical and Pollution Features

| Option | Default | Produces | Effect |
|--------|---------|----------|--------|
| `statement_add_lexical_features` | `False` | `statement_original_char_len`, `statement_original_word_count`, `statement_upper_ratio`, `statement_exclamation_count`, `statement_question_count`, `statement_clean_digit_ratio` | Surface-level text statistics. Cheap to compute. Upper ratio and punctuation counts are loose proxies for claim style. `[all]` — scalable columns if `statement_scale != 'none'`. |
| `statement_add_pollution_features` | `False` | `statement_row_spillover_flag` (bool), `statement_tab_count`, `statement_newline_count` | Measures data quality artifacts from CSV parsing. Useful as a noise indicator; low signal after repair is applied. Consider for `[tree]` where the flag becomes a split condition. |
| `statement_add_spelling_errors` | `False` | `statement_clean_spelling_err_count` | Counts tokens not in NLTK's English word list. Proxy for informal writing or garbled text. `[linear]`, `[tree]`. Requires NLTK words corpus. |
| `statement_add_rare_token_features` | `False` | `statement_clean_rare_token_count`, `statement_clean_avg_token_freq` | Counts tokens appearing ≤ `rare_token_threshold` times in the training corpus. Low average frequency may indicate unusual or fabricated claims. `[linear]`, `[tree]`. Requires a pass over the full corpus to build `token_freqs`; computed automatically if `statement_token_freqs=None`. |
| `statement_rare_token_threshold` | `1` | — | Frequency cutoff for the rare-token counter. Value of 1 means hapax legomena only; raise to 3–5 to capture very-low-frequency tokens. |

### NER Features

| Option | Default | Produces | Effect |
|--------|---------|----------|--------|
| `statement_add_ner_features` | `False` | `statement_original_PERSON`, `_ORG`, `_GPE`, `_DATE`, `_NUM`, `_OTHER`, `_total_entities` | Named entity counts from spaCy. Political statements frequently reference persons, organizations, and places — these counts capture referential density. `[linear]`, `[tree]`. Scales with `statement_scale`. Requires spaCy and the model specified by `statement_ner_model` (default: `en_core_web_sm`). Slow on large datasets. |
| `statement_ner_model` | `'en_core_web_sm'` | — | spaCy model name. `en_core_web_sm` is fast; `en_core_web_trf` is slower but more accurate (transformer-based). |

### Vectorization

Vectorization converts `statement_clean` (or `statement_original` for embeddings) into numeric columns prefixed `statement_clean_vec_*` or `statement_original_vec_*`. Only one vectorizer type is active at a time.

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `statement_vectorizer_type` | `'none'` \| `'tfidf'` \| `'bigram'` \| `'binary'` \| `'embeddings'` | `'none'` | The core vectorization strategy. See detail below. |
| `statement_vectorizer_max_features` | int \| None | `None` | Cap the vocabulary size. `None` = keep all terms. Setting 5000–20000 is standard for TF-IDF with `[linear]` models. Large vocabularies increase memory and training time. |
| `statement_vectorizer_min_df` | int | `1` | Ignore terms appearing in fewer than this many documents. `min_df=2` is a cheap way to cut hapax legomena; `min_df=5` reduces noise further on larger corpora. |
| `statement_vectorizer_max_df` | float | `1.0` | Ignore terms in more than this fraction of documents. Values like `0.9` cut near-universal stopwords that escaped the stopword list. |
| `statement_embedding_model` | str | `'all-MiniLM-L6-v2'` | Sentence transformer model used when `vectorizer_type='embeddings'`. |
| `statement_fitted_vectorizer` | object \| None | `None` | Pass a pre-fitted vectorizer (e.g. from training) to apply it to test data without refitting. Required during inference. |

**Vectorizer type detail:**

| Type | Output | Best for | Notes |
|------|--------|----------|-------|
| `'none'` | No vec columns | `[transformer]` pipelines that take raw text | Minimal feature matrix |
| `'tfidf'` | Sparse float columns, one per vocabulary term | `[linear]`, `[nlp-classical]` | TF-IDF down-weights common words. Strong baseline for LinearSVC and LogisticRegression. |
| `'bigram'` | Sparse int count columns, unigrams + bigrams | `[linear]`, `[nlp-classical]` | Larger vocabulary than TF-IDF unigrams. Captures two-word phrases ("not true", "has never"). Memory intensive without `max_features`. |
| `'binary'` | Sparse binary (0/1) columns | `[linear]`, `[nlp-classical]` | Presence/absence only. Slightly lower signal than TF-IDF but simpler; good for Bernoulli Naive Bayes. |
| `'embeddings'` | Dense float columns (384 for `all-MiniLM-L6-v2`) | `[dl]`, `[linear]` (with PCA), `[tree]` | Semantic dense representations. Use `all-MiniLM-L6-v2` for speed; larger models improve quality. Applied to `statement_original` (original casing). Requires `sentence-transformers`. |

> For `[tree]` models, avoid sparse TF-IDF matrices (very high dimensionality). Use `'embeddings'` or select a small `max_features` vocabulary.

---

## `subject.py`

Processes the `subject` column, which can contain multiple comma/pipe-separated topic tags. All options are prefixed `subject_` in `OneStepOptions`.

**Always produces:** `subject_clean`

### Presets

| Name | Key settings |
|------|-------------|
| `minimal` | No length features, no topic list, no grouped primary |
| `expanded` | All features on (default strategy: `'first'`) |
| `multi_topic` | `primary_strategy='multi'`, caps at 3 topics |
| `rare_safe` | `primary_strategy='most_frequent'`, `group_rare=True`, safe for all CV splits |

### Cleaning and Topic Splitting

| Option | Default | Effect |
|--------|---------|--------|
| `subject_clean_text` | `True` | Lowercases, strips HTML/URLs, normalizes separators. Leave on. |
| `subject_normalize_separators` | `True` | Converts `|`, `/`, `;` to `,` before splitting. Ensures consistent topic parsing. |
| `subject_split_topics` | `True` | Parses comma-separated topic strings into a list. Required for all topic-count and multi-topic features. |

### Primary Topic Selection

| Option | Values | Default | Effect |
|--------|--------|---------|--------|
| `subject_primary_strategy` | `'first'` \| `'most_frequent'` \| `'longest'` \| `'multi'` | `'first'` | Determines which topic from a multi-topic row is used as `subject_primary`. `'first'` is fastest and deterministic. `'most_frequent'` picks the topic with the most rows across the dataset — more stable encoding for `[linear]` and `[tree]` with low-cardinality outcomes. `'multi'` always writes `'multi-topic'` and is useful only for flagging multi-topic rows. |
| `subject_max_topics_for_primary` | int \| None | `None` | If set, rows with more than this many topics receive `multi_topic_label` instead of a computed primary. Useful with `'first'` or `'most_frequent'` strategy to avoid picking a primary for very noisy rows. |
| `subject_multi_topic_label` | str | `'multi-topic'` | String label for rows exceeding `max_topics_for_primary`. |

### Feature Flags

| Option | Default | Produces | Best for |
|--------|---------|----------|----------|
| `subject_add_primary` | `False` | `subject_primary` (string) | Required prerequisite for frequency, grouped, and true-rate features. `[tree]` with label encoding; `[linear]` with one-hot/frequency encoding. |
| `subject_add_topic_count` | `False` | `subject_topic_count` (int) | Count of comma-separated topics. More topics = vaguer claim. `[all]`. Scalable. |
| `subject_add_multiple_topics_flag` | `False` | `subject_has_multiple_topics` (0/1) | Binary flag for multi-topic rows. `[all]`. |
| `subject_add_topic_list` | `False` | `subject_topics` (Python list) | Raw list of topics per row. Not a numeric feature — useful for downstream multi-label encoding or inspection. Not directly usable in sklearn pipelines without further processing. |
| `subject_add_length_features` | `False` | `subject_length` (char count), `subject_token_count` | Proxy for how specific or complex the subject label is. `[all]`. Scalable. |
| `subject_add_subject_frequency` | `False` | `subject_frequency` (int count) | How often this primary subject appears in the dataset. Frequency encoding is directly usable by `[linear]` and `[tree]` models. Requires `add_primary=True`. |
| `subject_add_subject_is_rare` | `False` | `subject_is_rare` (0/1) | Binary flag: `1` if frequency < `rare_threshold`. Requires `add_primary=True`. `[all]`. |
| `subject_add_grouped_primary` | `False` | `subject_primary_grouped` (string) | Rare subjects collapsed to `rare_label`. Lower-cardinality version of `subject_primary` — safer for encoding. Requires `add_primary=True` and `group_rare=True`. `[tree]`, `[linear]` (after label encoding). |
| `subject_add_subject_primary_true_rate` | `False` | `subject_primary_true_rate` (float) | **Leakage risk.** Empirical label mean per subject. Very predictive — compute only inside CV folds. `[linear]`, `[tree]`. |
| `subject_group_rare` | `False` | Activates rare-grouping logic | Prerequisite flag for `add_grouped_primary`. |
| `subject_rare_threshold` | `10` | — | Frequency cutoff for rare detection. |

---

## `speaker.py`

Processes the `speaker` column (politician names). High cardinality — the dataset has hundreds of unique speakers. All options are prefixed `speaker_` in `OneStepOptions`.

**Always produces:** `speaker_clean`

### Presets

| Name | Key settings |
|------|-------------|
| `minimal` | No grouping, no title/comma/period flags |
| `expanded` | All features on |
| `rare_safe` | Grouping on, no true-rate |

### Options

| Option | Default | Produces | Best for |
|--------|---------|----------|----------|
| `speaker_clean_text` | `True` | Normalized lowercase name | Baseline normalization. |
| `speaker_normalize_separators` | `True` | Collapses `|`, `/`, `;`, `,`, `-` to space | Standardizes "Smith, John" → "smith john". |
| `speaker_add_frequency` | `False` | `speaker_frequency`, `speaker_frequency_pct` | How often this speaker appears. High-frequency speakers tend to be established politicians with a known credibility record. `[linear]`, `[tree]`. Scalable. |
| `speaker_add_is_rare` | `False` | `speaker_is_rare` (0/1) | `1` if frequency < `rare_threshold`. Useful for downstream grouping. `[all]`. |
| `speaker_add_grouped_speaker` | `False` | `speaker_grouped` (string) | Rare speakers replaced by `'other'`. Drastically reduces cardinality. Requires `group_rare=True`. `[tree]`, `[linear]` after encoding. |
| `speaker_add_length_features` | `False` | `speaker_char_len`, `speaker_token_count` | Name length loosely correlates with how formal or institutional the speaker name was entered. `[all]`. Scalable. |
| `speaker_add_title_flag` | `False` | `speaker_has_title` (0/1) | `1` if raw name contains "mr", "mrs", "gov", "sen", "rep", etc. Distinguishes elected officials from private citizens. `[all]`. |
| `speaker_add_comma_flag` | `False` | `speaker_has_comma` (0/1) | Names entered as "Last, First" indicate a different data-entry convention. Could be a proxy for data source. `[tree]`. |
| `speaker_add_period_flag` | `False` | `speaker_has_period` (0/1) | Presence of "." in the name — initials, abbreviations. `[tree]`. |
| `speaker_add_token_count` | `False` | `speaker_token_count` (int) | Number of name tokens. `[all]`. |
| `speaker_add_speaker_primary_true_rate` | `False` | `speaker_primary_true_rate` (float) | **Leakage risk.** Empirical label mean per speaker. The single most predictive feature in this dataset — speakers repeat over time. Must be computed inside CV folds. `[linear]`, `[tree]`. |
| `speaker_rare_threshold` | `5` | — | Frequency cutoff. `5` is reasonable; lower if you want to keep more speakers distinct. |

---

## `speaker_job.py`

Processes the `speaker_job` column (job title / occupation). Similar cardinality concerns to `speaker`. All options prefixed `speaker_job_` in `OneStepOptions`.

**Always produces:** `speaker_job_clean`

### Presets

Same three names as `speaker`: `minimal`, `expanded`, `rare_safe`.

### Options

The option set mirrors `speaker.py` with the following additions:

| Option | Default | Produces | Notes |
|--------|---------|----------|-------|
| `speaker_job_add_slash_flag` | `False` | `speaker_job_has_slash` (0/1) | Job titles like "CEO/Chairman" contain slashes. Signals multi-role positions. `[tree]`. |
| `speaker_job_add_ampersand_flag` | `False` | `speaker_job_has_ampersand` (0/1) | "Law & Policy Director" style entries. `[tree]`. |
| `speaker_job_add_comma_flag` | `False` | `speaker_job_has_comma` (0/1) | Commas in job titles often indicate "Title, Organization". `[tree]`. |
| `speaker_job_add_title_flag` | `False` | `speaker_job_has_title` (0/1) | Detects senior titles: CEO, CFO, professor, senator, governor, judge, etc. Strong signal for institutional speakers. `[all]`. |
| `speaker_job_add_grouped_job` | `False` | `speaker_job_grouped` (string) | Rare jobs → `'other'`. Requires `group_rare=True`. `[tree]`, `[linear]`. |
| `speaker_job_add_job_primary_true_rate` | `False` | `speaker_job_primary_true_rate` (float) | **Leakage risk.** CV folds only. `[linear]`, `[tree]`. |

All options from `speaker.py` are mirrored: `add_frequency`, `add_is_rare`, `add_length_features`, `add_token_count`, `group_rare`, `rare_threshold`.

---

## `party_affiliation.py`

Processes the `party_affiliation` column. Relatively low cardinality (democrat, republican, and ~30 smaller parties/roles). All options prefixed `party_affiliation_` in `OneStepOptions`.

**Always produces:** `party_affiliation_clean`

### Presets

| Name | Key settings |
|------|-------------|
| `minimal` | No grouping, no flags, no major/institutional markers |
| `expanded` | All features on |
| `rare_safe` | Grouping on, no true-rate |

### Options

Standard options mirror `speaker_job`: `add_frequency`, `add_is_rare`, `add_grouped_party`, `add_length_features`, `add_token_count`, `group_rare`, `rare_threshold`.

**Party-specific additions:**

| Option | Default | Produces | Notes |
|--------|---------|----------|-------|
| `party_affiliation_add_is_major_party` | `False` | `party_affiliation_is_major_party` (0/1) | `1` for `'democrat'` or `'republican'`. Binary proxy for two-party affiliation — removes the need for full one-hot encoding in simple pipelines. `[all]`. |
| `party_affiliation_add_is_institutional` | `False` | `party_affiliation_is_institutional` (0/1) | `1` for non-partisan institutional roles: `state-official`, `business-leader`, `journalist`, `newsmaker`, `columnist`, `county-commissioner`, `education-official`, `labor-leader`, `activist`. These speakers are not classic party members and may have distinct credibility patterns. `[all]`. |
| `party_affiliation_add_party_primary_true_rate` | `False` | `party_affiliation_primary_true_rate` (float) | **Leakage risk.** CV folds only. `[linear]`, `[tree]`. |
| `party_affiliation_add_slash_flag` | `False` | `party_affiliation_has_slash` (0/1) | Slash in party string ("democrat/progressive"). `[tree]`. |
| `party_affiliation_add_ampersand_flag` | `False` | `party_affiliation_has_ampersand` (0/1) | Ampersand in party string. `[tree]`. |
| `party_affiliation_add_comma_flag` | `False` | `party_affiliation_has_comma` (0/1) | Comma in party string. `[tree]`. |
| `party_affiliation_add_parentheses_flag` | `False` | `party_affiliation_has_parentheses` (0/1) | Parentheses in raw value. `[tree]`. |

---

## `state.py`

Processes the `state_info` column (U.S. state or geographic label). Low cardinality — 50 states plus a tail of non-state values. All options prefixed `state_` in `OneStepOptions`.

**Always produces:** `state_info_clean` (after normalization to full state name)

### Presets

| Name | Key settings |
|------|-------------|
| `drop` | Removes the column entirely |
| `minimal` | is_us_state + frequency only |
| `expanded` | All features including US region |
| `rare_safe` | Grouping on, no region |

### Options

| Option | Default | Produces | Notes |
|--------|---------|----------|-------|
| `state_drop` | `False` | Column removed | Use when geographic signal is not relevant to your hypothesis. |
| `state_normalize_state` | `True` | Standardizes 2-letter codes (e.g. `'ca'`) to full names (`'california'`). Essential for consistent encoding. Leave on. | |
| `state_add_is_us_state` | `False` | `state_info_is_us_state` (0/1) | `1` if the value maps to a known U.S. state. Separates in-state political claims from international or non-geographic entries. `[all]`. |
| `state_add_frequency` | `False` | `state_info_frequency`, `state_info_frequency_pct` | How often this state appears. High-frequency states (Texas, Florida, California) drive more claims. `[linear]`, `[tree]`. Scalable. |
| `state_add_is_rare` | `False` | `state_info_is_rare` (0/1) | `1` if frequency < `rare_threshold`. `[all]`. |
| `state_add_grouped_state` | `False` | `state_info_grouped` (string) | Rare states → `'other'`. Requires `group_rare=True`. `[tree]`, `[linear]`. |
| `state_add_has_us_words` | `False` | `state_info_has_us_words` (0/1) | `1` if value contains "us", "united states", or "usa". Catches non-state rows that still refer to the U.S. `[tree]`. |
| `state_add_us_region` | `False` | `state_info_us_region` (string: `northeast`/`south`/`midwest`/`west`/`unknown`) | Census region of the state. Lower-cardinality version of state; helps `[linear]` and `[tree]` capture regional political patterns without per-state encoding. `[tree]`, `[linear]` after encoding. |
| `state_add_length_features` | `False` | `state_info_char_len` (int) | Character length of the normalized state name. `[tree]`. Scalable. |
| `state_add_token_count` | `False` | `state_info_token_count` (int) | Word count in state name. `[tree]`. |

---

## `feature_engineering.py`

Cross-column features computed after all other modules. Output columns are always prefixed `fe_`. Three families: interaction keys, aggregate statistics, and text-style signals. All options prefixed `fe_` in `OneStepOptions`.

Depends on `*_clean` columns from all upstream modules, so it must run last.

### Source Column Overrides

The `fe_statement_col`, `fe_speaker_col`, etc. options let you point the module at a renamed upstream column. Only change these if you renamed a module's output. Default values match the standard `*_clean` column names.

`fe_label_col` must be set (e.g. `fe_label_col='label'`) to activate any aggregate/target-stat feature.

### Interaction Features

Concatenate two cleaned categorical columns into a joint string key: `"speaker_clean__subject_clean"`. After this, encode the joint column (label encode or frequency encode) before feeding to a model.

These features are best for `[tree]` models, which split on joint keys natively, and for `[linear]` models when the joint string is frequency-encoded (not one-hot — cardinality is too high).

| Option | Default | Produces | Captures |
|--------|---------|----------|----------|
| `fe_add_speaker_subject` | `False` | `fe_speaker_subject` | Whether specific speakers are more deceptive on specific topics. |
| `fe_add_speaker_party` | `False` | `fe_speaker_party` | Per-speaker party combination key. |
| `fe_add_subject_party` | `False` | `fe_subject_party` | Party-topic credibility patterns. |
| `fe_add_speaker_job_subject` | `False` | `fe_speaker_job_subject` | Whether a speaker's occupation affects truthfulness by topic. |
| `fe_add_state_party` | `False` | `fe_state_party` | Regional political credibility. |
| `fe_add_speaker_statement_len_bucket` | `False` | `fe_speaker_len_bucket` | Whether certain speakers tend to make short/medium/long claims. |
| `fe_statement_len_bins` | `(50, 150)` | — | Word count thresholds for short/medium/long buckets. Adjust based on claim length distribution. |

**Preset `interactions`** enables all six of the above at once.

### Aggregate Features

Per-group statistics computed by grouping on a categorical column and taking the mean of another column. Label-based aggregates are target encoding — CV-fold discipline required.

| Option | Default | Produces | Leakage? | Notes |
|--------|---------|----------|----------|-------|
| `fe_add_speaker_true_rate` | `False` | `fe_speaker_true_rate` (float) | **Yes** | Mean label per speaker. Requires `fe_label_col`. CV folds only. Highly predictive for repeat speakers. |
| `fe_add_subject_true_rate` | `False` | `fe_subject_true_rate` (float) | **Yes** | Mean label per subject topic. Requires `fe_label_col`. CV folds only. |
| `fe_add_party_true_rate` | `False` | `fe_party_true_rate` (float) | **Yes** | Mean label per party. Requires `fe_label_col`. CV folds only. |
| `fe_add_speaker_avg_statement_len` | `False` | `fe_speaker_avg_statement_len` (float) | No | Average word count per speaker across the dataset. No leakage. `[linear]`, `[tree]`. Scalable. |
| `fe_add_subject_avg_statement_len` | `False` | `fe_subject_avg_statement_len` (float) | No | Average word count per subject. No leakage. `[linear]`, `[tree]`. Scalable. |
| `fe_add_speaker_avg_punctuation` | `False` | `fe_speaker_avg_punctuation` (float) | No | Mean punctuation density per speaker. High punctuation may indicate informal or emotional style. `[linear]`, `[tree]`. Scalable. |
| `fe_add_speaker_avg_number_ratio` | `False` | `fe_speaker_avg_number_ratio` (float) | No | Mean digit ratio per speaker. Speakers who cite specific numbers may make more verifiable (and sometimes more false) claims. `[linear]`, `[tree]`. Scalable. |

**Preset `expanded`** enables all non-leakage aggregates plus all interaction and text-style features.

### Text-Style Features

Derived directly from `statement_clean`. No leakage risk. Best for `[linear]` and `[tree]` models when combined with modest TF-IDF features. `[transformer]` models learn these patterns implicitly and gain little from them.

| Option | Default | Produces | Signal |
|--------|---------|----------|--------|
| `fe_add_negation_count` | `False` | `fe_negation_count` (int) | Count of negation words (not, never, nor, etc.). False claims sometimes use negation to create ambiguity. Scalable. |
| `fe_add_hedge_count` | `False` | `fe_hedge_count` (int) | Count of uncertainty words (maybe, possibly, seems, approximately, etc.). Vague or hedged claims are harder to verify — can correlate with false claims or with inherently uncertain topics. Scalable. |
| `fe_add_absolutist_count` | `False` | `fe_absolutist_count` (int) | Count of absolutist/extreme words (always, everyone, nothing, proven, fact, lie, hoax, etc.). Exaggerated language is a known misinformation signal. Scalable. |
| `fe_add_numeral_count` | `False` | `fe_numeral_count` (int) | Count of digit sequences. Specific numerical claims are more verifiable; the presence of many numbers can indicate either careful sourcing or manufactured specificity. Scalable. |
| `fe_add_proper_noun_count` | `False` | `fe_proper_noun_count` (int) | Heuristic count of mid-sentence capitalized words in `statement_original`. Approximates proper noun density. High proper noun count suggests name-dropping or geo/org references. Scalable. |
| `fe_add_readability` | `False` | `fe_readability` (float) | Flesch Reading Ease score (no external library). Higher = easier to read. Very simple or very complex language may be indicative. Scalable. |
| `fe_add_sentiment` | `False` | `fe_sentiment_polarity`, `fe_sentiment_subjectivity` (floats) | TextBlob polarity (−1 to +1) and subjectivity (0 to 1). Subjective, emotionally charged statements may correlate with misleading claims. Requires `pip install textblob`. Scalable. |

**Preset `text`** enables negation, hedge, absolutist, numeral, proper noun, and readability counts.

---

## Leakage Risk Summary

All features with leakage risk in one place. Use only inside CV training folds.

| Feature | Option | Module |
|---------|--------|--------|
| `subject_primary_true_rate` | `subject_add_subject_primary_true_rate=True` | `subject.py` |
| `speaker_primary_true_rate` | `speaker_add_speaker_primary_true_rate=True` | `speaker.py` |
| `speaker_job_primary_true_rate` | `speaker_job_add_job_primary_true_rate=True` | `speaker_job.py` |
| `party_affiliation_primary_true_rate` | `party_affiliation_add_party_primary_true_rate=True` | `party_affiliation.py` |
| `fe_speaker_true_rate` | `fe_add_speaker_true_rate=True` | `feature_engineering.py` |
| `fe_subject_true_rate` | `fe_add_subject_true_rate=True` | `feature_engineering.py` |
| `fe_party_true_rate` | `fe_add_party_true_rate=True` | `feature_engineering.py` |

---

## Recommended Configurations by Model Type

### LinearSVC / LogisticRegression

```python
opts = OneStepOptions(
    # Text
    statement_vectorizer_type="tfidf",
    statement_vectorizer_max_features=10000,
    statement_vectorizer_min_df=2,
    statement_vectorizer_max_df=0.9,
    statement_stopword_removal=True,
    statement_lemmatizer="wordnet",
    statement_keep_negations=True,
    statement_add_lexical_features=True,
    statement_scale="standardize",
    # Subject
    subject_add_primary=True,
    subject_add_subject_frequency=True,
    subject_add_topic_count=True,
    subject_scale="standardize",
    # Speaker
    speaker_add_frequency=True,
    speaker_add_grouped_speaker=True,
    speaker_group_rare=True,
    speaker_scale="standardize",
    # Party
    party_affiliation_add_is_major_party=True,
    party_affiliation_add_frequency=True,
    # Feature engineering
    fe_add_negation_count=True,
    fe_add_hedge_count=True,
    fe_add_absolutist_count=True,
    fe_add_readability=True,
    fe_scale="standardize",
)
```

### GradientBoosting / XGBoost / LightGBM

```python
opts = OneStepOptions(
    # Text: embeddings over TF-IDF for tree models
    statement_vectorizer_type="embeddings",
    statement_add_lexical_features=True,
    statement_add_ner_features=True,
    # Subject
    subject_add_primary=True,
    subject_add_subject_frequency=True,
    subject_add_topic_count=True,
    subject_add_multiple_topics_flag=True,
    subject_add_grouped_primary=True,
    subject_group_rare=True,
    # Speaker
    speaker_add_frequency=True,
    speaker_add_grouped_speaker=True,
    speaker_group_rare=True,
    speaker_add_title_flag=True,
    # Party
    party_affiliation_add_is_major_party=True,
    party_affiliation_add_is_institutional=True,
    party_affiliation_add_grouped_party=True,
    party_affiliation_group_rare=True,
    # State
    state_add_is_us_state=True,
    state_add_us_region=True,
    state_add_grouped_state=True,
    state_group_rare=True,
    # Feature engineering
    fe_add_speaker_subject=True,
    fe_add_subject_party=True,
    fe_add_speaker_avg_statement_len=True,
    fe_add_absolutist_count=True,
    fe_add_negation_count=True,
    # No scaling needed for trees
)
```

### Sentence Transformer + Dense Head

```python
opts = OneStepOptions(
    # Preserve original casing for the transformer; no stemming/stopwords
    statement_lower=False,
    statement_stopword_removal=False,
    statement_stemmer="none",
    statement_lemmatizer="none",
    statement_vectorizer_type="embeddings",
    statement_embedding_model="all-MiniLM-L6-v2",
    # Minimal metadata features alongside embeddings
    speaker_add_frequency=True,
    speaker_scale="normalize",
    party_affiliation_add_is_major_party=True,
    fe_add_absolutist_count=True,
    fe_add_sentiment=True,
    fe_scale="normalize",
)
```
