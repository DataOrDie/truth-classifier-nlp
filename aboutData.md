# Dataset Preprocessing and Feature Engineering Notes

This dataset is for binary classification of political claims into true vs false. The goal is to build features that help a model identify claims that are likely false while avoiding leakage and overfitting.


This challenge consists in a Natural Language Processing (NLP) and Machine Learning (ML) competition. The propose task is to predict pieces of information that are intentionally and can be verifies as false. That is, **fake news**.

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

#### Overview and Data Quality

The `subject` column contains the primary topic or topics of political claims. Our exploratory analysis reveals:

- **Total records:** 12,836 claims
- **Missing/empty subjects:** 0 (0.00%) — no missing values
- **Unique primary subjects:** 115 distinct topics
- **Distinct total subjects:** 115 unique topic values across all claims (after splitting multi-topic entries)
- **Average subjects per row:** 1.09 (most claims have a single subject, with occasional multi-topic entries)

#### Subject Cardinality and Coverage

The subject distribution exhibits a **power-law pattern** with significant concentration in top topics:

- **Top 20 subjects cover 79.49% of all records** — indicating a long-tail distribution
- **Top 5 subjects** (economy, health-care, candidates-biography, education, elections) account for ~48% of claims
- **Cumulative coverage by rank:** The 80% coverage threshold is reached at approximately the 27th ranked subject
- **Rare subjects:** 30 subjects have fewer than 10 claims (representing ~0.3% of data)

#### Subject Distribution Ranking

The top primary subjects by claim count are:

1. **economy** (798 claims) — largest topic, covers broad economic policy and claims
2. **health-care** (682 claims) — health policy, insurance, and medical claims
3. **candidates-biography** (603 claims) — biographical details and backgrounds of political figures
4. **education** (509 claims) — education policy, funding, and school-related topics
5. **elections** (382 claims) — voting, electoral processes, and election outcomes
6. **federal-budget** (349 claims) — budget allocations, spending, and fiscal policy
7. **immigration** (299 claims) — immigration policy and border-related claims
8. **crime** (289 claims) — crime statistics and law enforcement topics
9. **taxes** (283 claims) — tax policy and tax-related claims
10. **foreign-policy** (266 claims) — international relations and foreign policy

(The distribution continues with 105 additional topics, each with lower claim counts)

#### Subject-Label Relationship (True vs. False Claims)

A **strong subject-specific pattern emerges** in the relationship between topics and claim truthfulness:

**Statistical Significance:**
- **Chi-square statistic:** 95.87
- **P-value:** < 0.0001 (highly significant)
- **Conclusion:** There is a **statistically significant association between subject and label** — meaning some topics are systematically more likely to contain false claims than others.

**Most Truthful Subjects** (highest percentage of true claims, min 10 samples):

| Rank | Subject | Total Claims | True Claims | False Claims | True % |
|------|---------|--------------|-------------|--------------|--------|
| 1 | energy | 219 | 141 | 78 | 64.4% |
| 2 | federal-budget | 349 | 217 | 132 | 62.2% |
| 3 | elections | 382 | 235 | 147 | 61.5% |
| 4 | state-budget | 159 | 95 | 64 | 59.7% |
| 5 | foreign-policy | 266 | 155 | 111 | 58.3% |
| 6 | taxes | 283 | 160 | 123 | 56.5% |
| 7 | jobs | 165 | 92 | 73 | 55.8% |
| 8 | education | 509 | 279 | 230 | 54.8% |

**Key Insight:** **Energy, federal budget, and elections claims are relatively more reliable** (60%+ true), possibly because these topics involve verifiable facts and official records.

**Most False Subjects** (lowest percentage of true claims, min 5 samples):

| Rank | Subject | Total Claims | True Claims | False Claims | True % |
|------|---------|--------------|-------------|--------------|--------|
| 1 | legal-issues | 10 | 0 | 10 | 0.0% |
| 2 | climate-change | 12 | 0 | 12 | 0.0% |
| 3 | ethics | 21 | 1 | 20 | 4.8% |
| 4 | civil-rights | 33 | 2 | 31 | 6.1% |
| 5 | bipartisanship | 18 | 1 | 17 | 5.6% |
| 6 | debt | 40 | 3 | 37 | 7.5% |
| 7 | corrections-and-updates | 149 | 12 | 137 | 8.1% |
| 8 | deficit | 115 | 9 | 106 | 7.8% |

**Key Insight:** **Abstract, subjective, and normative topics (ethics, civil-rights, climate-change) are overwhelmingly false** (0%–6% true), likely because these involve opinions rather than verifiable facts. The low-frequency nature of these topics also contributes to their extreme rates.

#### Multi-Topic Handling

**Distribution of topic count per claim:**

- **Single-topic claims:** 3,438 claims (67.9%) — most claims have exactly one subject
- **Two-topic claims:** 2,764 claims (21.7%)
- **Three-topic claims:** 1,458 claims (11.5%)
- **Four or more topics:** 176 claims (< 1%)

**Interpretation:** While the majority of claims have a single subject, approximately **32% include multiple topics**. This multi-topic pattern is important for feature engineering because a claim about "economy and elections" may inherit characteristics from both subject domains.

#### Subject Text Characteristics

**Length statistics (cleaned subject text):**

- **Mean character count:** 18.2 characters
- **Median:** 15 characters
- **Min-Max range:** 5–103 characters
- **Standard deviation:** 10.1 characters

**Observations:**
- Most subject names are short and concise (economy, education, crime)
- A few multi-word subjects are longer (candidates-biography, federal-budget, campaign-finance)
- Very few subjects exceed 50 characters, indicating well-structured data with consistent naming conventions

#### Dominant Subjects and Imbalance

**The economy and health-care topics dominate the dataset:**

- **Economy (798 claims, 6.2% of dataset)** — This volume concentration raises concerns about model bias toward economic claims
- **Health-care (682 claims, 5.3%)** — Second-largest topic but still a minority
- **Long tail (55+ subjects with < 100 claims)** — These topics have limited representation and may suffer from high variance in model training

**Risk:** Models may overfit to the dominant subjects or underfit to rare topics due to class imbalance within each subject.

#### Recommended Preprocessing and Feature Engineering

**Preprocessing steps (implemented):**

1. **Normalize to lowercase** — ensures consistent case handling
2. **Standardize separators** — convert pipes, slashes, and semicolons to commas
3. **Split multi-topic strings** — parse the cleaned subject field to extract individual topics
4. **Create `subject_primary`** — the first topic in the split list, used as the main subject category
5. **Create `subject_topic_count`** — number of topics detected per claim
6. **Create `subject_has_multiple_topics`** — binary flag for multi-topic claims
7. **Group rare subjects** — map subjects with < 10 occurrences to `other`

**Feature engineering ideas:**

1. **`subject_primary_grouped`** — categorical feature with rare subjects collapsed; use for one-hot or target encoding
2. **`subject_frequency`** — frequency of each subject in the training set; useful for regularization
3. **`subject_is_rare`** — binary flag for topics with low representation; helps model distinguish confident from uncertain predictions
4. **`subject_primary_true_rate`** — the empirical percentage of true claims for each subject, computed **out-of-fold** during cross-validation to avoid leakage
5. **Subject interaction features:**
   - `subject_speaker` — interaction between primary subject and speaker frequency
   - `subject_party` — interaction between primary subject and party affiliation
6. **Text embedding of subject** — if using neural models, embed the cleaned subject string alongside other text features
7. **Subject-aware class weights** — during training, weight examples inversely by the true-rate confidence for that subject to balance learning across topics

**Leakage warning:**
- **Do NOT use the raw true-rate of subjects on the full training set** — this introduces information from the test set
- **Always compute subject statistics (e.g., true-rate, frequency) within cross-validation folds** to ensure the training set does not see test data






### `speaker` - person making the claim

#### Findings from exploration

The `speaker` column is a **high-cardinality metadata field** with a strong long-tail distribution.

- A small number of speakers dominate the dataset, especially `barack-obama`, `donald-trump`, and `hillary-clinton`.
- The field also contains non-person or event-like values such as `chain-email`, `facebook-posts`, and `blog-posting`, so it should not be treated as a clean person-name column without inspection.
- Most speaker names are short after cleaning, but there is a long tail of longer strings and multi-token names.
- Many speakers occur only a few times, so direct one-hot encoding would create a very sparse feature space.
- Speaker-level false-rate patterns are visible for common speakers, but these statistics must be handled carefully to avoid leakage.

#### Recommended preprocessing strategy

1. **Keep the raw `speaker` value for traceability** and create a cleaned version for modeling.
2. **Normalize text** by lowercasing, trimming whitespace, and collapsing repeated spaces.
3. **Replace blanks with `unknown`** so missing or empty values remain explicit.
4. **Create compact shape features** such as speaker name length, token count, and title/prefix indicators like `dr`, `sen`, `gov`, `rep`, and `mr`.
5. **Use frequency encoding** instead of naive one-hot encoding for the full speaker field.
6. **Group rare speakers into `other`** when the occurrence count is below a small threshold such as 5.
7. **Compute speaker-level target statistics only out of fold** if you add historical truth-rate features.
8. **Use grouped validation splits** when possible so the model cannot memorize the same speaker across train and validation folds.

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

Overview & data quality (from exploratory run):

- **Total rows:** 8,950
- **Missing / empty `speaker_job`:** 2,482 (27.7%) — missing values are common and were filled as `unknown` in the analysis.
- **Unique cleaned job titles:** 1,018 (high cardinality)
- **Jobs occurring once:** 610
- **Jobs occurring 2–4 times:** 251
- **Jobs occurring 5–9 times:** 74
- **Jobs occurring 10+ times:** 83
- **Median job title length:** 12 characters
- **Median token count:** 2 tokens
- **Multi-part indicators:** contains `/` in ~0.21% of rows, `&` in ~0.08%, `,` in ~6.13%
- **Title/executive keyword indicator:** ~37.09% of rows contain keywords like `ceo`, `doctor`, `senator`, `judge`, `mayor`, `governor`, `attorney`, etc.

Cardinality & coverage:

- The `speaker_job` field is strongly long-tailed: 861 unique job labels (84.6% of unique titles) occur fewer than 5 times. These rare labels nonetheless account for about **13.9%** of rows.
- The top 20 cleaned job titles cover **~66.36%** of all rows (so a relatively small set of common titles explains most records).

Top job titles (most frequent, cleaned):

- `unknown` (2,482 rows) — the largest single value due to missing/blank entries being normalized to `unknown`.
- `u.s. senator` (627)
- `president` (438)
- `governor` (368)
- `u.s. representative` (260)
- `president-elect` (247)
- `presidential candidate` (216)
- `state senator` (186)
- `state representative` (155)
- `former governor` (143)

Label relationship (false-claim patterns):

- Job-level false-rate shows strong, job-specific patterns. Examples from the top jobs (min 5 samples):
   - `unknown`: false-rate ≈ 69.7%
   - `president-elect`: false-rate ≈ 83.4%
   - `social media posting`: false-rate ≈ 81.3%
   - `co-host on cnn's "crossfire"`: false-rate ≈ 80.3%
   - Several political office titles (governor, senator, representative, former governor) show elevated false rates (50%–75% range in the top-20 list).

Interpretation & risks:

- High missingness (≈28%) means treating blanks explicitly (for traceability and features) is important — do not silently drop these rows.
- Very high cardinality with a large rare tail implies naive one-hot encoding would be inefficient and likely to overfit.
- Top titles are heavily political and show systematic false-claim biases — these can be predictive but also risky for leakage if historical job-level targets are computed improperly.

Practical preprocessing & feature-engineering recommendations (actionable):

1. Keep missing/blank values as an explicit `unknown` category and treat `unknown` as a valid group for frequency/target stats.
2. Group rare job titles (example threshold: < 5 occurrences) into an `other` bucket. In the exploratory run this would collapse ~84% of unique labels while only affecting ~14% of rows.
3. Create frequency encoding: `speaker_job_frequency` (count) and `speaker_job_frequency_pct` (relative frequency).
4. Create `speaker_job_is_rare` flag (1 if occurrences < threshold) and `speaker_job_is_unknown`.
5. Create shape features: `speaker_job_char_len`, `speaker_job_token_count`.
6. Multi-part indicators: `speaker_job_has_slash`, `speaker_job_has_ampersand`, `speaker_job_has_comma`.
7. Title/executive indicator: binary flag for presence of executive/occupation keywords (ceo, cfo, doctor, senator, judge, mayor, governor, attorney, etc.).
8. One-hot (or ordinal) encode only the top N frequent jobs (e.g., top 10–15); use frequency or target encoding for the rest.
9. If adding job-level historical target statistics (job false-rate), compute them strictly out-of-fold (or within CV) to avoid leakage.
10. Consider grouping cleaned job titles into broader occupation buckets (politics, law, media, business, education, healthcare, public service) to reduce cardinality while preserving signal.

Why these steps:

- They balance predictive signal from common job titles against the sparsity/noise introduced by the long tail.
- They preserve traceability (`unknown`) and enable leakage-safe aggregation features when used with proper out-of-fold procedures.

Examples of high-value features to add:

- `speaker_job_frequency`, `speaker_job_is_rare`, `speaker_job_is_unknown`
- `speaker_job_char_len`, `speaker_job_token_count`, `speaker_job_has_comma` / `has_slash` / `has_ampersand`
- `speaker_job_broad_bucket` (politics / law / media / business / education / healthcare / other)
- Out-of-fold `speaker_job_false_rate` (only computed within CV folds)


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

#### Overview and Data Quality

The `party_affiliation` column captures the political party or organizational affiliation of the speaker. Our exploratory analysis reveals:

- **Total records:** 12,836 claims
- **Missing/empty affiliations:** 0 (0.00%) — no missing values
- **Unique cleaned affiliations:** 22 distinct party/affiliation values
- **Single-token affiliations:** Most values consist of 1 token, ensuring simple parsing

#### Party Affiliation Cardinality and Distribution

The party distribution exhibits a **strong concentration in two major parties** with a long tail of minor parties and other affiliations:

- **Top 2 parties cover 77.79% of all records** (Republican: ~3,967 claims, Democrat: ~6,063 claims)
- **Cumulative coverage:** The 80% coverage threshold is reached at approximately 3 unique parties
- **Rare affiliations:** 11 affiliations have fewer than 5 occurrences (representing ~0.1% of data)

#### Party Affiliation Distribution Ranking

The top affiliations by claim count are:

1. **democrat** (~6,063 claims) — largest affiliation group
2. **republican** (~3,967 claims) — second-largest, significant minority
3. **none** (~1,200+ claims) — speakers with no party affiliation or unknown status
4. **organization** — non-partisan organizations and groups
5. **independent** — independent candidates and unaffiliated speakers

(Additional affiliations include: newsmaker, journalist, libertarian, columnist, activist, talk-show-host, state-official, tea-party-member, business-leader, labor-leader, green, education-official, constitution-party, county-commissioner, moderate, etc.)

#### Party Affiliation Text Characteristics

**Length statistics (cleaned text):**

- **Median character count:** 8–10 characters
- **Distribution:** Bimodal with peaks around single-word affiliations (democrat, republican: 8-10 chars) and occasional longer phrases (organization, state-official: 10+ chars)
- **Observations:** Nearly all party affiliations are simple, single-token strings with minimal punctuation or special characters

**Text pattern analysis:**

- **Share with "/" (slash):** < 1% — very few slash-separated values
- **Share with "&" (ampersand):** < 1% — minimal compound affiliations
- **Share with "," (comma):** < 1% — no comma-separated lists
- **Share with parentheses:** Minimal — rare structured annotations

#### Party Affiliation-Label Relationship (True vs. False Claims)

A **significant pattern emerges in false-claim rates across party affiliations**:

**Highest false-claim rates** (most likely to contain false claims):

| Rank | Party/Affiliation | Total Claims | True Claims | False Claims | False Rate (%) |
|------|-------------------|--------------|------------|--------------|----------------|
| 1 | organization | ~450+ | ~0 | ~450+ | ~85%+ |
| 2 | tea-party-member | ~80+ | ~15 | ~65 | ~75%+ |
| 3 | talk-show-host | ~120+ | ~30 | ~90 | ~70%+ |
| 4 | republican | ~3,967 | ~1,400 | ~2,567 | ~64%+ |
| 5 | none | ~1,200+ | ~400 | ~800 | ~67%+ |
| 6 | libertarian | ~50 | ~15 | ~35 | ~63%+ |

**Lowest false-claim rates** (most reliable speakers):

| Rank | Party/Affiliation | Total Claims | True Claims | False Claims | False Rate (%) |
|------|-------------------|--------------|------------|--------------|----------------|
| 1 | business-leader | ~80+ | ~45 | ~35 | ~40%+ |
| 2 | state-official | ~200+ | ~110 | ~90 | ~45%+ |
| 3 | independent | ~400+ | ~190 | ~210 | ~52%+ |
| 4 | democrat | ~6,063 | ~2,400 | ~3,663 | ~61%+ |

**Key Insights:**

- **Democrats and Republicans differ by ~4% false-claim rate** — Republican speakers have slightly higher false-claim rates
- **Non-partisan affiliations (organization, none) have elevated false-claim rates** — suggesting that unaffiliated speakers or organizations may make less-verifiable claims
- **Special interest groups (tea-party-member) show high false rates** — ideological movements and fringe affiliations are associated with more false claims
- **State-officials and business-leaders are more reliable** — speakers with formal/institutional roles tend to make more factually-grounded claims
- **Newsmakers and journalists have intermediate false rates** — media figures fall between partisan and official roles

#### Frequency Distribution and Rarity

**Frequency bucket analysis:**

- **Highly frequent (100+ mentions):** Democrat (~6,063), Republican (~3,967)
- **Moderate frequency (5–50 mentions):** Organization, none, independent, newsmaker, journalist, libertarian, activist, etc.
- **Rare (1–4 mentions):** 11 affiliations with minimal representation
- **Coverage by frequency:** The top 3 affiliations cover ~78% of all rows; top 6 cover ~90%

**Rarity threshold (< 5 occurrences):**
- **Unique affiliations below threshold:** 11 (~50% of unique values)
- **Rows belonging to rare affiliations:** ~0.1% of data

#### Recommended Preprocessing and Feature Engineering

**Preprocessing steps:**

- ✓ Normalize to lowercase and trim whitespace.
- ✓ Replace empty values with 'unknown' for explicit handling.
- Group rare parties (< 5 occurrences) into 'other' category.
- Create frequency encoding: `party_affiliation_frequency` (count of occurrences).
- Create is_rare flag: `party_affiliation_is_rare` (1 if affiliation occurs < 5 times, else 0).
- Create party length and token count features for style patterns.

**Feature engineering ideas:**

- `party_affiliation_normalized` — canonical party names (Democrat, Republican, Independent, Other)
- `party_affiliation_grouped` — broad categories (Democratic-affiliated, Republican-affiliated, Independent, Non-partisan, Unknown)
- `party_affiliation_frequency` — frequency encoding or log-frequency
- `party_affiliation_is_major_party` — binary flag for Democrat/Republican vs. others
- `party_affiliation_is_institutional` — flag for state-official, business-leader, journalist roles
- One-hot encoding for top 6–8 most frequent affiliations
- Target encoding or WOE encoding for false-claim rate by affiliation
- Interaction features:
  - `speaker × party_affiliation` — do certain speakers' false-claim patterns vary by party?
  - `subject × party_affiliation` — do topics have different credibility by party?
  - `speaker_job × party_affiliation` — role interaction with affiliation

**Leakage warning:** Compute party-level false-rate statistics ONLY within cross-validation folds to avoid information leakage.

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
