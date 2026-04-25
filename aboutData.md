# Dataset Preprocessing and Feature Engineering Notes

This dataset is for binary classification of political claims into true vs false. The goal is to build features that help a model identify claims that are likely false while avoiding leakage and overfitting.

## Core Principles

- Keep `id` only for traceability, not as a model feature.
- Treat `label` as the binary target and validate that it is clean and consistent.
- Use the claim text in `statement` as the main signal.
- Use metadata columns as supporting context, not as direct memorization keys.
- Build frequency, rarity, and interaction features carefully, ideally with leakage-safe training splits.

## Column-by-Column Preprocessing

### `id` - unique identifier

Recommended preprocessing:

- Normalize to string and trim whitespace.
- Check for duplicates and missing values.
- Exclude from model features.
- Optionally create a stable hash bucket for reproducible train/validation splits.

Reasoning:

- `id` is not predictive of truthfulness.
- Using it in training would add noise and risk memorization.
- It is still useful for tracking predictions back to the source record.

Feature engineering ideas:

- `id_bucket` for deterministic split assignment.
- Duplicate detection flags if the same record appears more than once.

### `label` - truthfulness target

Recommended preprocessing:

- Convert to a strict binary target.
- Validate that only two classes exist.
- Map the false-claim class to `1` and the true-claim class to `0`.
- Compute class weights if the classes are imbalanced.

Reasoning:

- This is the supervision signal for the model.
- A clean target is more important than any feature engineering.
- Imbalance-aware training usually performs better than raw accuracy-focused training.

Feature engineering ideas:

- `class_weight` for training.
- Threshold tuning on validation data.
- Evaluation by precision, recall, F1, and PR-AUC for the false-claim class.

### `statement` - text of the claim

Recommended preprocessing:

- Lowercase the text.
- Remove HTML and URLs.
- Normalize whitespace.
- Preserve meaningful punctuation if using text models.
- Optionally remove stopwords or stem, but keep a clean raw-text version too.

Reasoning:

- This is the strongest predictive column.
- Claim wording often contains the cues that distinguish false from true statements.
- Text models usually outperform metadata-only models on this task.

Feature engineering ideas:

- TF-IDF word n-grams.
- TF-IDF character n-grams.
- Length features: character count, word count, average word length.
- Style features: uppercase ratio, punctuation counts, quote count, number count.
- Linguistic cues: negation words, hedge words, absolutist words, question marks, exclamation marks.
- Readability or complexity metrics.
- Named-entity counts and digit/date density.

### `subject` - topic or topics of the claim

Recommended preprocessing:

- Normalize to lowercase.
- Remove noise and standardize separators.
- Split multi-topic strings when the field contains multiple topics.
- Keep a primary subject and a cleaned subject field.
- Group rare subjects into `other`.

Reasoning:

- Subject provides topical context for the claim.
- Some false-claim patterns are topic-specific.
- Rare categories can create sparsity if handled naively.

Feature engineering ideas:

- `subject_primary` for categorical encoding.
- `subject_topic_count`.
- `subject_has_multiple_topics`.
- One-hot encoding for common subjects.
- Frequency encoding for high-cardinality subject values.
- Subject clusters or topic taxonomy labels if available.

### `speaker` - person making the claim

Recommended preprocessing:

- Normalize speaker names to lowercase.
- Keep the raw name for traceability.
- Avoid naive one-hot encoding when there are many speakers.
- Group rare speakers into `other`.
- Use frequency encoding or similar compact encodings.

Reasoning:

- Speaker identity can be helpful, but it is high-cardinality and can overfit.
- Some speakers may have characteristic phrasing or reliability patterns.
- Direct memorization of speaker names is risky if the split is not careful.

Feature engineering ideas:

- Speaker frequency.
- Rare-speaker flag.
- Speaker name length.
- Number of name tokens.
- Title/prefix indicator such as `dr`, `sen`, `gov`, `rep`, `mr`.
- Speaker-level historical aggregates computed out of fold.

### `speaker_job` - occupation of the speaker

Recommended preprocessing:

- Normalize job titles to lowercase.
- Keep the original text.
- Group rare occupations into `other`.
- Use compact encodings instead of one-hotting every unique job title.

Reasoning:

- Occupation is informative because different professions may speak in different contexts.
- The field is noisy and often inconsistent, so cleaning is important.
- Rare occupations can produce sparse features without adding much value.

Feature engineering ideas:

- Job frequency.
- Rare-job flag.
- Job length and token count.
- Indicators for `/`, `&`, or multi-part titles.
- Broad occupation buckets such as politics, law, media, business, education, healthcare, and public service.

### `state_info` - geographic context

Recommended preprocessing:

- Normalize location text to lowercase.
- Standardize U.S. state names and abbreviations.
- Group rare geographic values into `other`.
- Preserve raw text for reference.

Reasoning:

- Location may capture regional political or reporting differences.
- Raw geographic strings can be messy and inconsistent.
- Strong normalization reduces noise and improves category stability.

Feature engineering ideas:

- `state_info_is_us_state`.
- `state_info_frequency`.
- `state_info_is_rare`.
- `state_info_token_count`.
- `state_info_has_us_words`.
- Map states to broader regions such as Northeast, South, Midwest, and West.

### `party_affiliation` - political party of the speaker

Recommended preprocessing:

- Normalize party labels.
- Map common variants and abbreviations to canonical names.
- Group uncommon party labels into `other`.
- Add a major-party indicator.

Reasoning:

- Party affiliation can be a useful contextual signal for political claims.
- The raw field often contains inconsistent spellings or abbreviations.
- Major parties are usually more useful than one-off labels.

Feature engineering ideas:

- `party_affiliation_normalized`.
- `party_affiliation_grouped`.
- `party_affiliation_frequency`.
- `party_affiliation_is_major_party`.
- `party_affiliation_is_rare`.
- One-hot encoding for common parties.

## Cross-Column Feature Engineering Ideas

These combinations often add more value than any single column alone.

### Interaction features

- `speaker × subject`
- `speaker × party_affiliation`
- `subject × party_affiliation`
- `speaker_job × subject`
- `state_info × party_affiliation`
- `speaker × statement_length`

Reasoning:

- Some speakers behave differently across topics.
- Political context often changes by subject and region.
- Interaction features help the model capture patterns that are not visible in isolation.

### Aggregate features

- Speaker-level truth rate.
- Subject-level truth rate.
- Party-level truth rate.
- Average statement length by speaker or subject.
- Average punctuation or number usage by speaker.

Reasoning:

- Aggregates can capture historical patterns.
- These are often strong signals for classical ML models.
- They must be computed with cross-validation or out-of-fold logic to avoid leakage.

### Text-specific feature ideas

- Negation count.
- Hedge word count.
- Absolutist word count.
- Numeral and date counts.
- Proper noun count.
- Readability scores.
- Sentiment or subjectivity signals.
- Character n-gram representations.

Reasoning:

- Fake or unverifiable claims often show stylistic or linguistic patterns.
- These features complement TF-IDF and metadata signals.

## Recommended Modeling Setup

1. Start with a baseline using `statement` TF-IDF plus basic metadata encodings.
2. Add cleaned categorical features from `subject`, `speaker_job`, `party_affiliation`, and `state_info`.
3. Use `speaker` carefully with frequency encoding and leakage-safe aggregates.
4. Train a linear classifier such as Logistic Regression or Linear SVM first.
5. Evaluate with class-aware metrics, especially recall and F1 for the false-claim class.

## Practical Cautions

- Do not include `id` as a feature.
- Do not compute target-based aggregates on the full dataset before splitting.
- Be careful with high-cardinality fields like `speaker` and `speaker_job`.
- Use grouped or frequency-based encodings to reduce sparsity.
- Validate preprocessing decisions with cross-validation and error analysis.

## Summary

The dataset is best treated as a text-first classification problem with contextual metadata. The strongest immediate gains usually come from the cleaned `statement` text, then from careful handling of `subject`, `speaker`, `speaker_job`, `state_info`, and `party_affiliation`. The most useful engineering direction is to combine text representations with leakage-safe aggregate and interaction features.
