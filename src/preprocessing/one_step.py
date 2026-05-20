"""One-step preprocessing pipeline.

Runs all available preprocessing modules in sequence. Each module's options
are exposed as flat fields on OneStepOptions so callers can tune any parameter
without touching module internals.

Default configuration is intentionally minimal: every module produces only its
core cleaned column. Enable individual add_* flags to grow the feature set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd

from preprocessing import label as label_module
from preprocessing import id as id_module
from preprocessing import subject as subject_module
from preprocessing import statement_ds as statement_module
from preprocessing import speaker as speaker_module
from preprocessing import speaker_job as speaker_job_module
from preprocessing import party_affiliation as party_affiliation_module
from preprocessing import state as state_module
from preprocessing import feature_engineering as fe_module


@dataclass
class OneStepOptions:
	"""Options for the one-step preprocessing pipeline.

	Each section forwards to its corresponding preprocessing module. Field
	names follow the pattern <module>_<option>.

	Defaults are minimal: only core cleaned columns are produced. Set any
	add_* flag to True to enable that feature column.
	"""

	# -----------------------------------------------------------------------
	# Label
	# -----------------------------------------------------------------------
	label_option: str = label_module.DEFAULT_LABEL_OPTION   # 'skip' or 'drop'
	label_source_col: str = label_module.DEFAULT_LABEL_SOURCE_COL

	# -----------------------------------------------------------------------
	# ID
	# -----------------------------------------------------------------------
	id_option: str = id_module.DEFAULT_ID_OPTION            # 'drop' or 'hash_bucket'

	# -----------------------------------------------------------------------
	# Subject  →  produces: subject_clean
	# -----------------------------------------------------------------------
	subject_source_col: str = subject_module.DEFAULT_SUBJECT_SOURCE_COL
	subject_keep_original: bool = False
	subject_clean_text: bool = subject_module.DEFAULT_CLEAN_TEXT
	subject_normalize_separators: bool = subject_module.DEFAULT_NORMALIZE_SEPARATORS
	subject_split_topics: bool = subject_module.DEFAULT_SPLIT_TOPICS
	subject_primary_strategy: str = subject_module.DEFAULT_PRIMARY_STRATEGY
	subject_rare_threshold: int = subject_module.DEFAULT_RARE_THRESHOLD
	subject_rare_label: str = subject_module.DEFAULT_RARE_LABEL
	subject_group_rare: bool = False                        # requires add_primary + add_grouped_primary
	subject_max_topics_for_primary: int | None = subject_module.DEFAULT_MAX_TOPICS_FOR_PRIMARY
	subject_multi_topic_label: str = subject_module.DEFAULT_MULTI_TOPIC_LABEL
	subject_add_length_features: bool = False
	subject_add_topic_list: bool = False
	subject_add_topic_count: bool = False
	subject_add_multiple_topics_flag: bool = False
	subject_add_primary: bool = False
	subject_add_grouped_primary: bool = False               # requires add_primary
	subject_add_subject_frequency: bool = False             # requires add_primary
	subject_add_subject_is_rare: bool = False               # requires add_primary
	subject_add_subject_primary_true_rate: bool = False     # leakage risk — CV folds only
	subject_label_col: str | None = subject_module.DEFAULT_SUBJECT_LABEL_COL
	# 'none' | 'standardize' (z-score) | 'normalize' (min-max)
	# Scales: topic_count, length, token_count, frequency. NOT binary flags.
	subject_scale: str = subject_module.DEFAULT_SCALE
	subject_verbose: bool = False

	# -----------------------------------------------------------------------
	# Statement  →  produces: statement_original, statement_clean
	# -----------------------------------------------------------------------
	statement_source_col: str = statement_module.DEFAULT_SOURCE_COL
	statement_original_output_col: str = statement_module.DEFAULT_ORIGINAL_OUTPUT_COL
	statement_output_col: str = statement_module.DEFAULT_CLEAN_OUTPUT_COL
	statement_keep_original: bool = False
	statement_lower: bool = statement_module.DEFAULT_LOWER
	statement_remove_html: bool = statement_module.DEFAULT_REMOVE_HTML
	statement_remove_urls: bool = statement_module.DEFAULT_REMOVE_URLS
	statement_replace_numbers: bool = statement_module.DEFAULT_REPLACE_NUMBERS
	statement_number_token: str = statement_module.DEFAULT_NUMBER_TOKEN
	statement_stopword_removal: bool = statement_module.DEFAULT_STOPWORD_REMOVAL
	# 'none' | 'porter' | 'snowball' — porter/snowball require NLTK
	statement_stemmer: str = statement_module.DEFAULT_STEMMER
	statement_lemmatizer: str = statement_module.DEFAULT_LEMMATIZER
	statement_keep_negations: bool = True
	statement_remove_punctuation: bool = statement_module.DEFAULT_REMOVE_PUNCTUATION
	statement_repair_polluted_statements: bool = statement_module.DEFAULT_REPAIR_POLLUTED_STATEMENTS
	statement_add_rare_token_features: bool = False
	statement_rare_token_threshold: int = statement_module.DEFAULT_RARE_TOKEN_THRESHOLD
	statement_token_freqs: dict | None = None
	statement_add_spelling_errors: bool = False
	statement_add_lexical_features: bool = False
	statement_add_pollution_features: bool = False
	statement_add_ner_features: bool = False
	statement_ner_model: str = statement_module.DEFAULT_NER_MODEL
	statement_vectorizer_type: str = statement_module.DEFAULT_VECTORIZER   # 'none','tfidf','bigram','binary','embeddings'
	statement_vectorizer_max_features: int | None = statement_module.DEFAULT_MAX_FEATURES
	statement_vectorizer_min_df: int = statement_module.DEFAULT_MIN_DF
	statement_vectorizer_max_df: float = statement_module.DEFAULT_MAX_DF
	statement_embedding_model: str = statement_module.DEFAULT_EMBEDDING_MODEL
	statement_fitted_vectorizer: object | None = None
	# 'none' | 'standardize' | 'normalize'
	# Scales: char_len, word_count, upper_ratio, exclamation_count, question_count,
	#   digit_ratio, rare_token_count, avg_token_freq, spelling_err_count,
	#   tab_count, newline_count, NER counts. NOT binary flags or vec_* columns.
	statement_scale: str = statement_module.DEFAULT_SCALE
	statement_verbose: bool = False

	# -----------------------------------------------------------------------
	# Speaker  →  produces: speaker_clean
	# -----------------------------------------------------------------------
	speaker_source_col: str = speaker_module.DEFAULT_SPEAKER_SOURCE_COL
	speaker_keep_original: bool = False
	speaker_clean_text: bool = speaker_module.DEFAULT_CLEAN_TEXT
	speaker_normalize_separators: bool = speaker_module.DEFAULT_NORMALIZE_SEPARATORS
	speaker_group_rare: bool = False                        # requires add_grouped_speaker
	speaker_rare_threshold: int = speaker_module.DEFAULT_RARE_THRESHOLD
	speaker_rare_label: str = speaker_module.DEFAULT_RARE_LABEL
	speaker_add_length_features: bool = False
	speaker_add_frequency: bool = False
	speaker_add_is_rare: bool = False
	speaker_add_grouped_speaker: bool = False               # requires group_rare
	speaker_add_title_flag: bool = False
	speaker_add_comma_flag: bool = False
	speaker_add_period_flag: bool = False
	speaker_add_token_count: bool = False
	speaker_add_speaker_primary_true_rate: bool = False     # leakage risk — CV folds only
	speaker_label_col: str | None = speaker_module.DEFAULT_SPEAKER_LABEL_COL
	# 'none' | 'standardize' | 'normalize'
	# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
	speaker_scale: str = speaker_module.DEFAULT_SCALE
	speaker_verbose: bool = False

	# -----------------------------------------------------------------------
	# Speaker job  →  produces: speaker_job_clean
	# -----------------------------------------------------------------------
	speaker_job_source_col: str = speaker_job_module.DEFAULT_SOURCE_COL
	speaker_job_keep_original: bool = False
	speaker_job_clean_text: bool = speaker_job_module.DEFAULT_CLEAN_TEXT
	speaker_job_normalize_separators: bool = speaker_job_module.DEFAULT_NORMALIZE_SEPARATORS
	speaker_job_group_rare: bool = False                    # requires add_grouped_job
	speaker_job_rare_threshold: int = speaker_job_module.DEFAULT_RARE_THRESHOLD
	speaker_job_rare_label: str = speaker_job_module.DEFAULT_RARE_LABEL
	speaker_job_add_length_features: bool = False
	speaker_job_add_frequency: bool = False
	speaker_job_add_is_rare: bool = False
	speaker_job_add_grouped_job: bool = False               # requires group_rare
	speaker_job_add_title_flag: bool = False
	speaker_job_add_comma_flag: bool = False
	speaker_job_add_slash_flag: bool = False
	speaker_job_add_ampersand_flag: bool = False
	speaker_job_add_token_count: bool = False
	speaker_job_add_job_primary_true_rate: bool = False     # leakage risk — CV folds only
	speaker_job_job_label_col: Optional[str] = speaker_job_module.DEFAULT_JOB_LABEL_COL
	# 'none' | 'standardize' | 'normalize'
	# Scales: char_len, token_count, frequency, frequency_pct. NOT binary flags.
	speaker_job_scale: str = speaker_job_module.DEFAULT_SCALE
	speaker_job_verbose: bool = False

	# -----------------------------------------------------------------------
	# Party affiliation  →  produces: party_affiliation_clean
	# -----------------------------------------------------------------------
	party_affiliation_source_col: str = party_affiliation_module.DEFAULT_SOURCE_COL
	party_affiliation_keep_original: bool = False
	party_affiliation_clean_text: bool = party_affiliation_module.DEFAULT_CLEAN_TEXT
	party_affiliation_group_rare: bool = False              # requires add_grouped_party
	party_affiliation_rare_threshold: int = party_affiliation_module.DEFAULT_RARE_THRESHOLD
	party_affiliation_rare_label: str = party_affiliation_module.DEFAULT_RARE_LABEL
	party_affiliation_add_length_features: bool = False
	party_affiliation_add_frequency: bool = False
	party_affiliation_add_is_rare: bool = False
	party_affiliation_add_grouped_party: bool = False       # requires group_rare
	party_affiliation_add_slash_flag: bool = False
	party_affiliation_add_ampersand_flag: bool = False
	party_affiliation_add_comma_flag: bool = False
	party_affiliation_add_parentheses_flag: bool = False
	party_affiliation_add_token_count: bool = False
	party_affiliation_add_is_major_party: bool = False
	party_affiliation_add_is_institutional: bool = False
	party_affiliation_add_party_primary_true_rate: bool = False  # leakage risk — CV folds only
	party_affiliation_party_label_col: Optional[str] = party_affiliation_module.DEFAULT_PARTY_LABEL_COL
	# 'none' | 'standardize' | 'normalize'
	# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
	party_affiliation_scale: str = party_affiliation_module.DEFAULT_SCALE
	party_affiliation_verbose: bool = False

	# -----------------------------------------------------------------------
	# State info  →  produces: state_info_clean
	# -----------------------------------------------------------------------
	state_source_col: str = state_module.DEFAULT_SOURCE_COL
	state_drop: bool = state_module.DEFAULT_DROP
	state_keep_original: bool = False
	state_clean_text: bool = state_module.DEFAULT_CLEAN_TEXT
	state_normalize_state: bool = state_module.DEFAULT_NORMALIZE_STATE
	state_group_rare: bool = False                          # requires add_grouped_state
	state_rare_threshold: int = state_module.DEFAULT_RARE_THRESHOLD
	state_rare_label: str = state_module.DEFAULT_RARE_LABEL
	state_add_is_us_state: bool = False
	state_add_frequency: bool = False
	state_add_is_rare: bool = False
	state_add_grouped_state: bool = False                   # requires group_rare
	state_add_length_features: bool = False
	state_add_token_count: bool = False
	state_add_has_us_words: bool = False
	state_add_us_region: bool = False
	# 'none' | 'standardize' | 'normalize'
	# Scales: frequency, frequency_pct, char_len, token_count. NOT binary flags.
	state_scale: str = state_module.DEFAULT_SCALE
	state_verbose: bool = False

	# -----------------------------------------------------------------------
	# Feature engineering  →  produces: fe_* columns
	# -----------------------------------------------------------------------
	# Source column overrides — change only if you renamed the upstream outputs.
	fe_statement_col: str = fe_module.DEFAULT_STATEMENT_COL
	fe_statement_original_col: str = fe_module.DEFAULT_STATEMENT_ORIGINAL_COL
	fe_speaker_col: str = fe_module.DEFAULT_SPEAKER_COL
	fe_subject_col: str = fe_module.DEFAULT_SUBJECT_COL
	fe_party_col: str = fe_module.DEFAULT_PARTY_COL
	fe_speaker_job_col: str = fe_module.DEFAULT_SPEAKER_JOB_COL
	fe_state_col: str = fe_module.DEFAULT_STATE_COL
	fe_label_col: Optional[str] = fe_module.DEFAULT_LABEL_COL  # required for leakage-risk aggregates

	# Interaction features (all off by default)
	fe_add_speaker_subject: bool = False
	fe_add_speaker_party: bool = False
	fe_add_subject_party: bool = False
	fe_add_speaker_job_subject: bool = False
	fe_add_state_party: bool = False
	fe_add_speaker_statement_len_bucket: bool = False
	fe_statement_len_bins: tuple = fe_module.DEFAULT_STATEMENT_LEN_BINS  # (short_max, medium_max) word counts

	# Aggregate features (all off by default — leakage risk for *_true_rate)
	fe_add_speaker_true_rate: bool = False        # WARNING: CV folds only
	fe_add_subject_true_rate: bool = False        # WARNING: CV folds only
	fe_add_party_true_rate: bool = False          # WARNING: CV folds only
	fe_add_speaker_avg_statement_len: bool = False
	fe_add_subject_avg_statement_len: bool = False
	fe_add_speaker_avg_punctuation: bool = False
	fe_add_speaker_avg_number_ratio: bool = False

	# Text-style features (all off by default)
	fe_add_negation_count: bool = False
	fe_add_hedge_count: bool = False
	fe_add_absolutist_count: bool = False
	fe_add_numeral_count: bool = False
	fe_add_proper_noun_count: bool = False        # uses statement_original column
	fe_add_readability: bool = False
	fe_add_sentiment: bool = False                # requires: pip install textblob

	# 'none' | 'standardize' | 'normalize'
	# Scales: avg_statement_len, avg_punctuation, avg_number_ratio, true_rate features,
	#   negation/hedge/absolutist/numeral/proper_noun counts, readability, sentiment.
	# NOT: interaction string columns (fe_speaker_subject, etc.) or fe_speaker_len_bucket.
	fe_scale: str = fe_module.DEFAULT_SCALE

	fe_verbose: bool = False


def get_default_options() -> dict[str, Any]:
	"""Return defaults for all active preprocessing steps."""
	return asdict(OneStepOptions())


def add_tree_features(
	df_frame: pd.DataFrame,
	raw_dates: pd.DataFrame | None = None,
	options: OneStepOptions | None = None,
) -> pd.DataFrame:
	"""Compatibility no-op while tree feature modules are not available."""
	_ = raw_dates
	_ = options
	return df_frame.copy()


def preprocess_one_step(
	df: pd.DataFrame,
	options: OneStepOptions | None = None,
	is_tree_model: bool = False,
) -> pd.DataFrame:
	"""Run all available preprocessing modules in sequence.

	Parameters
	----------
	df : pd.DataFrame
		Input raw dataframe.
	options : OneStepOptions | None
		Pipeline options. If None, minimal defaults are used.
	is_tree_model : bool
		When True, calls add_tree_features (currently a no-op).
	"""
	opts = options or OneStepOptions()

	df_out = label_module.preprocess_label(
		df=df,
		option=opts.label_option,
		source_col=opts.label_source_col,
	)

	df_out = id_module.preprocess_id(
		df=df_out,
		option=opts.id_option,
	)

	df_out = subject_module.preprocess_subject(
		df=df_out,
		source_col=opts.subject_source_col,
		keep_original=opts.subject_keep_original,
		clean_text=opts.subject_clean_text,
		normalize_separators=opts.subject_normalize_separators,
		split_topics=opts.subject_split_topics,
		primary_strategy=opts.subject_primary_strategy,
		rare_threshold=opts.subject_rare_threshold,
		rare_label=opts.subject_rare_label,
		group_rare=opts.subject_group_rare,
		max_topics_for_primary=opts.subject_max_topics_for_primary,
		multi_topic_label=opts.subject_multi_topic_label,
		add_length_features=opts.subject_add_length_features,
		add_topic_list=opts.subject_add_topic_list,
		add_topic_count=opts.subject_add_topic_count,
		add_multiple_topics_flag=opts.subject_add_multiple_topics_flag,
		add_primary=opts.subject_add_primary,
		add_grouped_primary=opts.subject_add_grouped_primary,
		add_subject_frequency=opts.subject_add_subject_frequency,
		add_subject_is_rare=opts.subject_add_subject_is_rare,
		add_subject_primary_true_rate=opts.subject_add_subject_primary_true_rate,
		subject_label_col=opts.subject_label_col,
		scale=opts.subject_scale,
		verbose=opts.subject_verbose,
	)

	df_out = statement_module.preprocess_statement_ds(
		df=df_out,
		source_col=opts.statement_source_col,
		original_output_col=opts.statement_original_output_col,
		clean_output_col=opts.statement_output_col,
		keep_original=opts.statement_keep_original,
		lower=opts.statement_lower,
		remove_html=opts.statement_remove_html,
		remove_urls=opts.statement_remove_urls,
		replace_numbers=opts.statement_replace_numbers,
		remove_punctuation=opts.statement_remove_punctuation,
		number_token=opts.statement_number_token,
		stopword_removal=opts.statement_stopword_removal,
		stemmer=opts.statement_stemmer,
		lemmatizer=opts.statement_lemmatizer,
		keep_negations=opts.statement_keep_negations,
		add_rare_token_features=opts.statement_add_rare_token_features,
		rare_token_threshold=opts.statement_rare_token_threshold,
		token_freqs=opts.statement_token_freqs,
		add_spelling_errors=opts.statement_add_spelling_errors,
		add_lexical_features=opts.statement_add_lexical_features,
		add_pollution_features=opts.statement_add_pollution_features,
		add_ner_features=opts.statement_add_ner_features,
		ner_model=opts.statement_ner_model,
		vectorizer_type=opts.statement_vectorizer_type,
		vectorizer_max_features=opts.statement_vectorizer_max_features,
		vectorizer_min_df=opts.statement_vectorizer_min_df,
		vectorizer_max_df=opts.statement_vectorizer_max_df,
		embedding_model=opts.statement_embedding_model,
		fitted_vectorizer=opts.statement_fitted_vectorizer,
		repair_polluted_statements=opts.statement_repair_polluted_statements,
		scale=opts.statement_scale,
		verbose=opts.statement_verbose,
	)

	df_out = speaker_module.preprocess_speaker(
		df=df_out,
		source_col=opts.speaker_source_col,
		keep_original=opts.speaker_keep_original,
		clean_text=opts.speaker_clean_text,
		normalize_separators=opts.speaker_normalize_separators,
		group_rare=opts.speaker_group_rare,
		rare_threshold=opts.speaker_rare_threshold,
		rare_label=opts.speaker_rare_label,
		add_length_features=opts.speaker_add_length_features,
		add_frequency=opts.speaker_add_frequency,
		add_is_rare=opts.speaker_add_is_rare,
		add_grouped_speaker=opts.speaker_add_grouped_speaker,
		add_title_flag=opts.speaker_add_title_flag,
		add_comma_flag=opts.speaker_add_comma_flag,
		add_period_flag=opts.speaker_add_period_flag,
		add_token_count=opts.speaker_add_token_count,
		add_speaker_primary_true_rate=opts.speaker_add_speaker_primary_true_rate,
		speaker_label_col=opts.speaker_label_col,
		scale=opts.speaker_scale,
		verbose=opts.speaker_verbose,
	)

	df_out = speaker_job_module.preprocess_speaker_job(
		df=df_out,
		source_col=opts.speaker_job_source_col,
		keep_original=opts.speaker_job_keep_original,
		clean_text=opts.speaker_job_clean_text,
		normalize_separators=opts.speaker_job_normalize_separators,
		group_rare=opts.speaker_job_group_rare,
		rare_threshold=opts.speaker_job_rare_threshold,
		rare_label=opts.speaker_job_rare_label,
		add_length_features=opts.speaker_job_add_length_features,
		add_frequency=opts.speaker_job_add_frequency,
		add_is_rare=opts.speaker_job_add_is_rare,
		add_grouped_job=opts.speaker_job_add_grouped_job,
		add_title_flag=opts.speaker_job_add_title_flag,
		add_comma_flag=opts.speaker_job_add_comma_flag,
		add_slash_flag=opts.speaker_job_add_slash_flag,
		add_ampersand_flag=opts.speaker_job_add_ampersand_flag,
		add_token_count=opts.speaker_job_add_token_count,
		add_job_primary_true_rate=opts.speaker_job_add_job_primary_true_rate,
		job_label_col=opts.speaker_job_job_label_col,
		scale=opts.speaker_job_scale,
		verbose=opts.speaker_job_verbose,
	)

	df_out = party_affiliation_module.preprocess_party_affiliation(
		df=df_out,
		source_col=opts.party_affiliation_source_col,
		keep_original=opts.party_affiliation_keep_original,
		clean_text=opts.party_affiliation_clean_text,
		group_rare=opts.party_affiliation_group_rare,
		rare_threshold=opts.party_affiliation_rare_threshold,
		rare_label=opts.party_affiliation_rare_label,
		add_length_features=opts.party_affiliation_add_length_features,
		add_frequency=opts.party_affiliation_add_frequency,
		add_is_rare=opts.party_affiliation_add_is_rare,
		add_grouped_party=opts.party_affiliation_add_grouped_party,
		add_slash_flag=opts.party_affiliation_add_slash_flag,
		add_ampersand_flag=opts.party_affiliation_add_ampersand_flag,
		add_comma_flag=opts.party_affiliation_add_comma_flag,
		add_parentheses_flag=opts.party_affiliation_add_parentheses_flag,
		add_token_count=opts.party_affiliation_add_token_count,
		add_is_major_party=opts.party_affiliation_add_is_major_party,
		add_is_institutional=opts.party_affiliation_add_is_institutional,
		add_party_primary_true_rate=opts.party_affiliation_add_party_primary_true_rate,
		party_label_col=opts.party_affiliation_party_label_col,
		scale=opts.party_affiliation_scale,
		verbose=opts.party_affiliation_verbose,
	)

	df_out = state_module.preprocess_state_info(
		df=df_out,
		source_col=opts.state_source_col,
		drop=opts.state_drop,
		keep_original=opts.state_keep_original,
		clean_text=opts.state_clean_text,
		normalize_state=opts.state_normalize_state,
		group_rare=opts.state_group_rare,
		rare_threshold=opts.state_rare_threshold,
		rare_label=opts.state_rare_label,
		add_is_us_state=opts.state_add_is_us_state,
		add_frequency=opts.state_add_frequency,
		add_is_rare=opts.state_add_is_rare,
		add_grouped_state=opts.state_add_grouped_state,
		add_length_features=opts.state_add_length_features,
		add_token_count=opts.state_add_token_count,
		add_has_us_words=opts.state_add_has_us_words,
		add_us_region=opts.state_add_us_region,
		scale=opts.state_scale,
		verbose=opts.state_verbose,
	)

	df_out = fe_module.preprocess_feature_engineering(
		df=df_out,
		statement_col=opts.fe_statement_col,
		statement_original_col=opts.fe_statement_original_col,
		speaker_col=opts.fe_speaker_col,
		subject_col=opts.fe_subject_col,
		party_col=opts.fe_party_col,
		speaker_job_col=opts.fe_speaker_job_col,
		state_col=opts.fe_state_col,
		label_col=opts.fe_label_col,
		add_speaker_subject=opts.fe_add_speaker_subject,
		add_speaker_party=opts.fe_add_speaker_party,
		add_subject_party=opts.fe_add_subject_party,
		add_speaker_job_subject=opts.fe_add_speaker_job_subject,
		add_state_party=opts.fe_add_state_party,
		add_speaker_statement_len_bucket=opts.fe_add_speaker_statement_len_bucket,
		statement_len_bins=opts.fe_statement_len_bins,
		add_speaker_true_rate=opts.fe_add_speaker_true_rate,
		add_subject_true_rate=opts.fe_add_subject_true_rate,
		add_party_true_rate=opts.fe_add_party_true_rate,
		add_speaker_avg_statement_len=opts.fe_add_speaker_avg_statement_len,
		add_subject_avg_statement_len=opts.fe_add_subject_avg_statement_len,
		add_speaker_avg_punctuation=opts.fe_add_speaker_avg_punctuation,
		add_speaker_avg_number_ratio=opts.fe_add_speaker_avg_number_ratio,
		add_negation_count=opts.fe_add_negation_count,
		add_hedge_count=opts.fe_add_hedge_count,
		add_absolutist_count=opts.fe_add_absolutist_count,
		add_numeral_count=opts.fe_add_numeral_count,
		add_proper_noun_count=opts.fe_add_proper_noun_count,
		add_readability=opts.fe_add_readability,
		add_sentiment=opts.fe_add_sentiment,
		scale=opts.fe_scale,
		verbose=opts.fe_verbose,
	)

	if is_tree_model:
		df_out = add_tree_features(df_out, raw_dates=None, options=opts)

	return df_out


__all__ = ["OneStepOptions", "preprocess_one_step", "add_tree_features", "get_default_options"]
