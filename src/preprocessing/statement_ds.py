"""Dual-stream statement preprocessing utilities.

This module keeps two representations of the same statement text:
- an original-cased stream for NER and transformer-style features
- a cleaned lowercase stream for token-based features such as TF-IDF and n-grams

The module is intended for workflows where preprocessing happens before the
train/validation split. It returns a single modified DataFrame and does not
perform any splitting itself.
This module now owns the statement cleaning, feature extraction, and
vectorization helpers that used to live in ``statement.py``. The dual-stream
API keeps the cleaned statement text plus an optional original-cased copy for
NER and embedding-style features.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import scipy.sparse

DEFAULT_SOURCE_COL = "statement"
DEFAULT_CLEAN_OUTPUT_COL = "statement_clean"
DEFAULT_OUTPUT_COL = DEFAULT_CLEAN_OUTPUT_COL
DEFAULT_ORIGINAL_OUTPUT_COL = "statement_original"
DEFAULT_KEEP_ORIGINAL = True
DEFAULT_LOWER = True
DEFAULT_REMOVE_HTML = True
DEFAULT_REMOVE_URLS = True
DEFAULT_REPLACE_NUMBERS = False
DEFAULT_NUMBER_TOKEN = "<NUM>"
DEFAULT_STOPWORD_REMOVAL = False
DEFAULT_STEMMER = "none"
DEFAULT_LEMMATIZER = "none"
DEFAULT_VERBOSE = False
DEFAULT_REMOVE_PUNCTUATION = False
DEFAULT_ADD_RARE_TOKEN_FEATURES = False
DEFAULT_RARE_TOKEN_THRESHOLD = 1
DEFAULT_ADD_SPELLING_ERRORS = False
DEFAULT_ADD_LEXICAL_FEATURES = False
DEFAULT_ADD_POLLUTION_FEATURES = False
DEFAULT_VECTORIZER = "none"
DEFAULT_MAX_FEATURES = None
DEFAULT_MIN_DF = 1
DEFAULT_MAX_DF = 1.0
DEFAULT_EMBEDDING_MODEL = "all-mpnet-base-v2"
DEFAULT_ADD_NER_FEATURES = False
DEFAULT_NER_MODEL = "en_core_web_sm"
DEFAULT_REPAIR_POLLUTED_STATEMENTS = True
DEFAULT_SCALE = 'none'  # 'none' | 'standardize' | 'normalize'

_nltk_available = False
_stop_words: set[str] = set()
_porter = None
_snowball = None
_wordnet_lemmatizer = None

_sklearn_available = False
_TfidfVectorizer = None
_CountVectorizer = None

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

    _TfidfVectorizer = TfidfVectorizer
    _CountVectorizer = CountVectorizer
    _sklearn_available = True
except Exception:
    _sklearn_available = False
    _TfidfVectorizer = None
    _CountVectorizer = None

_sentence_transformers_available = False
_SentenceTransformer = None

try:
    from sentence_transformers import SentenceTransformer

    _SentenceTransformer = SentenceTransformer
    _sentence_transformers_available = True
except Exception:
    _sentence_transformers_available = False
    _SentenceTransformer = None

_spacy_available = False

try:
    import spacy

    _spacy_available = True
except Exception:
    _spacy_available = False

try:
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer, WordNetLemmatizer

    try:
        from nltk.stem.snowball import EnglishStemmer
    except Exception:
        EnglishStemmer = None

    try:
        _stop_words = set(stopwords.words("english"))
    except Exception:
        _stop_words = set()

    _porter = PorterStemmer()
    _wordnet_lemmatizer = WordNetLemmatizer()
    if "EnglishStemmer" in globals() and EnglishStemmer is not None:
        try:
            _snowball = EnglishStemmer()
        except Exception:
            _snowball = None
    _nltk_available = True
except Exception:
    _nltk_available = False
    _stop_words = set()
    _porter = None
    _snowball = None
    _wordnet_lemmatizer = None

_NEGATIONS: set[str] = {"no", "not", "nor", "never", "n't"}
_PUNCT_TRANSLATION_TABLE = str.maketrans("", "", string.punctuation)
_english_vocab: set[str] | None = None

try:
    from nltk.corpus import words as nltk_words

    try:
        _english_vocab = set(word.lower() for word in nltk_words.words())
    except Exception:
        _english_vocab = None
except Exception:
    _english_vocab = None


def _scale_col(series: pd.Series, method: str) -> pd.Series:
    """Apply z-score standardization or min-max normalization to a numeric series."""
    s = series.astype(float)
    if method == 'standardize':
        mu, sigma = s.mean(), s.std()
        return (s - mu) / sigma if sigma > 0 else s - mu
    if method == 'normalize':
        mn, mx = s.min(), s.max()
        rng = mx - mn
        return (s - mn) / rng if rng > 0 else s - mn
    return s


def _ensure_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame")
    return df[col].fillna("").astype(str)


def _clean_text(
    text: str,
    lower: bool = True,
    remove_html: bool = True,
    remove_urls: bool = True,
    replace_numbers: bool = False,
    number_token: str = DEFAULT_NUMBER_TOKEN,
) -> str:
    value = text
    # Normalize unicode quotes and dashes to ASCII equivalents
    value = value.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
    value = value.replace('\u2018', "'").replace('\u2019', "'")  # Smart single quotes
    value = value.replace('\u2013', '-').replace('\u2014', '-')  # En dash and em dash
    if lower:
        value = value.lower()
    if remove_html:
        value = re.sub(r"<[^>]+>", " ", value)
    if remove_urls:
        value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    if replace_numbers:
        value = re.sub(r"\b\d+[\d,.:/-]*\b", f" {number_token} ", value)
    value = re.sub(r"[^a-z0-9\s'?!.,%\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _tokenize(text: str, use_nltk: bool = False) -> list[str]:
    if use_nltk:
        try:
            from nltk.tokenize import word_tokenize

            return word_tokenize(text)
        except Exception:
            pass
    return text.split()


def _apply_stemming(tokens: Iterable[str], stemmer_name: str) -> list[str]:
    if stemmer_name == "porter" and _porter is not None:
        return [(_porter.stem(token)) for token in tokens]
    if stemmer_name == "snowball" and _snowball is not None:
        return [(_snowball.stem(token)) for token in tokens]
    return list(tokens)


def _apply_lemmatization(tokens: Iterable[str]) -> list[str]:
    if _wordnet_lemmatizer is None:
        return list(tokens)
    return [(_wordnet_lemmatizer.lemmatize(token)) for token in tokens]


def _build_stoplist(keep_negations: bool = True) -> set[str]:
    if not _stop_words:
        return set()
    if keep_negations:
        return set(word for word in _stop_words if word not in _NEGATIONS)
    return set(_stop_words)


def _remove_punctuation_from_tokens(tokens: Iterable[str]) -> list[str]:
    return [token for token in (value.translate(_PUNCT_TRANSLATION_TABLE) for value in tokens) if token]


def _create_vectorizer(
    vectorizer_type: str,
    max_features: int | None = None,
    min_df: int = 1,
    max_df: float = 1.0,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
):
    if vectorizer_type == "none":
        return None

    if vectorizer_type == "embeddings":
        if not _sentence_transformers_available:
            raise ImportError(
                "sentence-transformers is required for embeddings. Install it with: pip install sentence-transformers"
            )
        try:
            return _SentenceTransformer(embedding_model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load embedding model '{embedding_model}'. Make sure it's a valid sentence-transformers model. Error: {exc}"
            )

    if not _sklearn_available:
        raise ImportError(
            "scikit-learn is required for tfidf/bigram/binary vectorization. Install it with: pip install scikit-learn"
        )

    if vectorizer_type == "tfidf":
        return _TfidfVectorizer(max_features=max_features, min_df=min_df, max_df=max_df)
    if vectorizer_type == "bigram":
        return _CountVectorizer(ngram_range=(1, 2), max_features=max_features, min_df=min_df, max_df=max_df)
    if vectorizer_type == "binary":
        return _CountVectorizer(binary=True, max_features=max_features, min_df=min_df, max_df=max_df)

    raise ValueError(f"Unknown vectorizer_type: {vectorizer_type}")


def _apply_vectorizer(
    texts: pd.Series,
    vectorizer,
    vectorizer_type: str,
    output_col_prefix: str,
    fit: bool = True,
):
    if vectorizer is None:
        return pd.DataFrame(), None

    if vectorizer_type == "embeddings":
        embeddings = vectorizer.encode(texts.tolist(), convert_to_numpy=True)
        col_names = [f"{output_col_prefix}_{index}" for index in range(embeddings.shape[1])]
        vec_df = pd.DataFrame(embeddings, columns=col_names, index=texts.index)
        return vec_df, vectorizer

    if fit:
        vectorized = vectorizer.fit_transform(texts)
    else:
        vectorized = vectorizer.transform(texts)

    feature_names = vectorizer.get_feature_names_out()
    col_names = [f"{output_col_prefix}_{name}" for name in feature_names]

    if scipy.sparse.issparse(vectorized):
        vectorized_dense = vectorized.toarray()
    else:
        vectorized_dense = vectorized

    vec_df = pd.DataFrame(vectorized_dense, columns=col_names, index=texts.index)
    return vec_df, vectorizer


def _load_spacy_model(model_name: str):
    if not _spacy_available:
        return None

    try:
        return spacy.load(model_name)
    except OSError:
        raise ImportError(
            f"spacy model '{model_name}' not found. Install it with:\n  python -m spacy download {model_name}"
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load spacy model '{model_name}': {exc}")


def _extract_ner_features(text: str, nlp) -> dict:
    entity_counts = {
        "total_entities": 0,
        "PERSON": 0,
        "ORG": 0,
        "GPE": 0,
        "DATE": 0,
        "NUM": 0,
        "OTHER": 0,
    }

    if not text or nlp is None:
        return entity_counts

    try:
        doc = nlp(text)
        entity_counts["total_entities"] = len(doc.ents)

        for ent in doc.ents:
            label = ent.label_
            if label in ["CARDINAL", "MONEY", "PERCENT", "QUANTITY"]:
                entity_counts["NUM"] += 1
            elif label in entity_counts:
                entity_counts[label] += 1
            else:
                entity_counts["OTHER"] += 1
    except Exception:
        pass

    return entity_counts


def _statement_pollution_profile(text: str) -> dict[str, int | float | bool]:
    """Measure common spillover artifacts in a statement string."""
    raw = "" if text is None else str(text)
    tab_count = raw.count("\t")
    newline_count = raw.count("\n") + raw.count("\r")
    url_count = len(re.findall(r"https?://|www\.", raw))
    quote_count = raw.count('"') + raw.count("'")
    special_char_count = len(re.findall(r"[^A-Za-z0-9\s]", raw))
    word_count = max(len(raw.split()), 1)
    statement_length = len(raw)
    tab_to_word_ratio = tab_count / word_count
    special_char_ratio = special_char_count / max(statement_length, 1)
    row_spillover_flag = tab_count > 0 or newline_count > 0
    pollution_score = (
        tab_count * 25
        + newline_count * 25
        + url_count * 4
        + quote_count * 2
        + special_char_ratio * 10
        + min(tab_to_word_ratio, 10) * 10
        + min(statement_length / word_count, 30)
    )

    return {
        "statement_length": statement_length,
        "word_count": word_count,
        "tab_count": tab_count,
        "newline_count": newline_count,
        "url_count": url_count,
        "quote_count": quote_count,
        "special_char_count": special_char_count,
        "special_char_ratio": special_char_ratio,
        "tab_to_word_ratio": tab_to_word_ratio,
        "row_spillover_flag": row_spillover_flag,
        "pollution_score": pollution_score,
    }


def _repair_statement_text(text: str, lower: bool = False, repair_pollution: bool = True) -> str:
    """Normalize structural statement pollution while preserving case when requested."""
    raw = "" if text is None else str(text)

    if repair_pollution:
        segments = [segment.strip() for segment in re.split(r"[\t\r\n]+", raw) if segment.strip()]
        if segments:
            raw = segments[0]

    raw = raw.strip().strip('"').strip("'")
    raw = re.sub(r"\s+", " ", raw).strip()

    if lower:
        raw = raw.lower()

    return raw


@dataclass
class StatementDSOptions:
    """Options for the dual-stream statement preprocessing pipeline."""

    source_col: str = DEFAULT_SOURCE_COL
    original_output_col: str = DEFAULT_ORIGINAL_OUTPUT_COL
    keep_original: bool = DEFAULT_KEEP_ORIGINAL

    # Original-cased stream options.
    add_ner_features: bool = DEFAULT_ADD_NER_FEATURES
    ner_model: str = DEFAULT_NER_MODEL

    # Cleaned stream options.
    clean_output_col: str = DEFAULT_CLEAN_OUTPUT_COL
    lower: bool = DEFAULT_LOWER
    remove_html: bool = DEFAULT_REMOVE_HTML
    remove_urls: bool = DEFAULT_REMOVE_URLS
    replace_numbers: bool = DEFAULT_REPLACE_NUMBERS
    number_token: str = DEFAULT_NUMBER_TOKEN
    stopword_removal: bool = DEFAULT_STOPWORD_REMOVAL
    stemmer: str = DEFAULT_STEMMER
    lemmatizer: str = DEFAULT_LEMMATIZER
    keep_negations: bool = True
    verbose: bool = DEFAULT_VERBOSE
    remove_punctuation: bool = DEFAULT_REMOVE_PUNCTUATION
    add_rare_token_features: bool = DEFAULT_ADD_RARE_TOKEN_FEATURES
    rare_token_threshold: int = DEFAULT_RARE_TOKEN_THRESHOLD
    token_freqs: dict | None = None
    add_spelling_errors: bool = DEFAULT_ADD_SPELLING_ERRORS
    add_lexical_features: bool = DEFAULT_ADD_LEXICAL_FEATURES
    add_pollution_features: bool = DEFAULT_ADD_POLLUTION_FEATURES
    vectorizer_type: str = DEFAULT_VECTORIZER
    vectorizer_max_features: int | None = DEFAULT_MAX_FEATURES
    vectorizer_min_df: int = DEFAULT_MIN_DF
    vectorizer_max_df: float = DEFAULT_MAX_DF
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    fitted_vectorizer: object | None = None
    repair_polluted_statements: bool = DEFAULT_REPAIR_POLLUTED_STATEMENTS
    scale: str = DEFAULT_SCALE


def preprocess_statement_ds(
    df: pd.DataFrame,
    source_col: str = DEFAULT_SOURCE_COL,
    original_output_col: str = DEFAULT_ORIGINAL_OUTPUT_COL,
    keep_original: bool = DEFAULT_KEEP_ORIGINAL,
    # Original-cased stream options.
    add_ner_features: bool = DEFAULT_ADD_NER_FEATURES,
    ner_model: str = DEFAULT_NER_MODEL,
    # Cleaned stream options.
    clean_output_col: str = DEFAULT_CLEAN_OUTPUT_COL,
    lower: bool = DEFAULT_LOWER,
    remove_html: bool = DEFAULT_REMOVE_HTML,
    remove_urls: bool = DEFAULT_REMOVE_URLS,
    replace_numbers: bool = DEFAULT_REPLACE_NUMBERS,
    number_token: str = DEFAULT_NUMBER_TOKEN,
    stopword_removal: bool = DEFAULT_STOPWORD_REMOVAL,
    stemmer: str = DEFAULT_STEMMER,
    lemmatizer: str = DEFAULT_LEMMATIZER,
    keep_negations: bool = True,
    verbose: bool = DEFAULT_VERBOSE,
    remove_punctuation: bool = DEFAULT_REMOVE_PUNCTUATION,
    add_rare_token_features: bool = DEFAULT_ADD_RARE_TOKEN_FEATURES,
    rare_token_threshold: int = DEFAULT_RARE_TOKEN_THRESHOLD,
    token_freqs: dict | None = None,
    add_spelling_errors: bool = DEFAULT_ADD_SPELLING_ERRORS,
    add_lexical_features: bool = DEFAULT_ADD_LEXICAL_FEATURES,
    add_pollution_features: bool = DEFAULT_ADD_POLLUTION_FEATURES,
    vectorizer_type: str = DEFAULT_VECTORIZER,
    vectorizer_max_features: int | None = DEFAULT_MAX_FEATURES,
    vectorizer_min_df: int = DEFAULT_MIN_DF,
    vectorizer_max_df: float = DEFAULT_MAX_DF,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    fitted_vectorizer: object | None = None,
    repair_polluted_statements: bool = DEFAULT_REPAIR_POLLUTED_STATEMENTS,
    scale: str = DEFAULT_SCALE,
) -> pd.DataFrame:
    """Add cleaned and optionally original statement streams to a DataFrame."""

    source = _ensure_series(df, source_col)
    out = df.copy()
    _scalable: list[str] = []

    pollution_profile = source.apply(_statement_pollution_profile) if add_pollution_features else None
    if add_pollution_features and pollution_profile is not None:
        out["statement_row_spillover_flag"] = pollution_profile.apply(lambda row: row["row_spillover_flag"])
        out["statement_tab_count"] = pollution_profile.apply(lambda row: row["tab_count"])
        out["statement_newline_count"] = pollution_profile.apply(lambda row: row["newline_count"])
        _scalable += ["statement_tab_count", "statement_newline_count"]

    original_text = source.apply(
        lambda text: _repair_statement_text(
            text,
            lower=False,
            repair_pollution=repair_polluted_statements,
        )
    )
    if keep_original:
        out[original_output_col] = original_text

    cleaned = source.apply(
        lambda text: _clean_text(
            _repair_statement_text(
                text,
                lower=lower,
                repair_pollution=repair_polluted_statements,
            ),
            lower=lower,
            remove_html=remove_html,
            remove_urls=remove_urls,
            replace_numbers=replace_numbers,
            number_token=number_token,
        )
    )

    use_nltk_tokenize = _nltk_available
    stoplist = _build_stoplist(keep_negations=keep_negations)
    token_freqs_local = token_freqs

    def _compute_doc_token_freqs_from_series(series: pd.Series) -> dict:
        freq: dict[str, int] = {}
        for txt in series.fillna("").astype(str):
            txt = _repair_statement_text(txt, lower=lower, repair_pollution=repair_polluted_statements)
            cleaned_txt = _clean_text(
                txt,
                lower=lower,
                remove_html=remove_html,
                remove_urls=remove_urls,
                replace_numbers=replace_numbers,
                number_token=number_token,
            )
            tokens = _tokenize(cleaned_txt, use_nltk=use_nltk_tokenize)
            if remove_punctuation:
                tokens = _remove_punctuation_from_tokens(tokens)
            for token in tokens:
                freq[token] = freq.get(token, 0) + 1
        return freq

    def _process_one(text: str) -> str:
        tokens = _tokenize(text, use_nltk=use_nltk_tokenize)
        if remove_punctuation:
            tokens = _remove_punctuation_from_tokens(tokens)
        if stopword_removal and stoplist:
            tokens = [token for token in tokens if token not in stoplist]
        if stemmer != "none":
            tokens = _apply_stemming(tokens, stemmer)
        if lemmatizer == "wordnet":
            tokens = _apply_lemmatization(tokens)
        return " ".join(tokens)

    if add_rare_token_features and token_freqs_local is None:
        try:
            token_freqs_local = _compute_doc_token_freqs_from_series(source)
        except Exception:
            token_freqs_local = None

    processed_clean = cleaned.apply(_process_one)

    def _tokens_for_doc(text: str) -> list[str]:
        cleaned_text = _clean_text(
            _repair_statement_text(
                text,
                lower=lower,
                repair_pollution=repair_polluted_statements,
            ),
            lower=lower,
            remove_html=remove_html,
            remove_urls=remove_urls,
            replace_numbers=replace_numbers,
            number_token=number_token,
        )
        tokens = _tokenize(cleaned_text, use_nltk=use_nltk_tokenize)
        if remove_punctuation:
            tokens = _remove_punctuation_from_tokens(tokens)
        if stopword_removal and stoplist:
            tokens = [token for token in tokens if token not in stoplist]
        if stemmer != "none":
            tokens = _apply_stemming(tokens, stemmer)
        if lemmatizer == "wordnet":
            tokens = _apply_lemmatization(tokens)
        return tokens

    rare_counts: list[int] = []
    avg_freqs: list[float] = []
    spelling_err_counts: list[int] = []
    for txt in source:
        tokens = _tokens_for_doc(txt)
        if add_rare_token_features and token_freqs_local:
            rare_count = sum(1 for token in tokens if token_freqs_local.get(token, 0) <= rare_token_threshold)
            avg_freq = float(sum(token_freqs_local.get(token, 0) for token in tokens) / max(len(tokens), 1))
        else:
            rare_count = 0
            avg_freq = 0.0
        if add_spelling_errors:
            if _english_vocab is not None:
                spelling_err_count = sum(1 for token in tokens if token.lower() not in _english_vocab)
            else:
                spelling_err_count = sum(1 for token in tokens if not any(ch.isalpha() for ch in token))
        else:
            spelling_err_count = 0
        rare_counts.append(rare_count)
        avg_freqs.append(avg_freq)
        spelling_err_counts.append(spelling_err_count)

    out[clean_output_col] = processed_clean

    if add_lexical_features:
        out[f"{original_output_col}_char_len"] = original_text.str.len()
        out[f"{original_output_col}_word_count"] = original_text.str.split().str.len()
        out[f"{source_col}_upper_ratio"] = original_text.apply(lambda text: sum(ch.isupper() for ch in text) / max(len(text), 1))
        out[f"{source_col}_exclamation_count"] = original_text.str.count("!")
        out[f"{source_col}_question_count"] = original_text.str.count(r"\?")
        out[f"{clean_output_col}_digit_ratio"] = out[clean_output_col].apply(
            lambda text: sum(ch.isdigit() for ch in text) / max(len(text), 1)
        )
        _scalable += [
            f"{original_output_col}_char_len", f"{original_output_col}_word_count",
            f"{source_col}_upper_ratio", f"{source_col}_exclamation_count",
            f"{source_col}_question_count", f"{clean_output_col}_digit_ratio",
        ]

    if add_rare_token_features:
        out[f"{clean_output_col}_rare_token_count"] = rare_counts
        out[f"{clean_output_col}_avg_token_freq"] = avg_freqs
        _scalable += [f"{clean_output_col}_rare_token_count", f"{clean_output_col}_avg_token_freq"]
    if add_spelling_errors:
        out[f"{clean_output_col}_spelling_err_count"] = spelling_err_counts
        _scalable.append(f"{clean_output_col}_spelling_err_count")

    if add_ner_features:
        try:
            nlp = _load_spacy_model(ner_model)
            ner_features = {
                "total_entities": [],
                "PERSON": [],
                "ORG": [],
                "GPE": [],
                "DATE": [],
                "NUM": [],
                "OTHER": [],
            }

            for text in original_text:
                entity_counts = _extract_ner_features(text, nlp)
                for key, value in entity_counts.items():
                    ner_features[key].append(value)

            for entity_type, counts in ner_features.items():
                out[f"{original_output_col}_{entity_type}"] = counts
                _scalable.append(f"{original_output_col}_{entity_type}")

            if verbose:
                print(f"NER features extracted using model: {ner_model}")
        except Exception as exc:
            if verbose:
                print(f"Warning: NER feature extraction failed: {exc}")

    if vectorizer_type != "none":
        vectorizer_texts = original_text if vectorizer_type == "embeddings" else out[clean_output_col]

        if fitted_vectorizer is None:
            vectorizer = _create_vectorizer(
                vectorizer_type=vectorizer_type,
                max_features=vectorizer_max_features,
                min_df=vectorizer_min_df,
                max_df=vectorizer_max_df,
                embedding_model=embedding_model,
            )
            vec_df, _ = _apply_vectorizer(
                vectorizer_texts,
                vectorizer,
                vectorizer_type,
                f"{original_output_col if vectorizer_type == 'embeddings' else clean_output_col}_vec",
                fit=True,
            )
            out = pd.concat([out, vec_df], axis=1)
        else:
            vec_df, _ = _apply_vectorizer(
                vectorizer_texts,
                fitted_vectorizer,
                vectorizer_type,
                f"{original_output_col if vectorizer_type == 'embeddings' else clean_output_col}_vec",
                fit=False,
            )
            out = pd.concat([out, vec_df], axis=1)

    if scale != 'none':
        for col in _scalable:
            if col in out.columns:
                out[col] = _scale_col(out[col], scale)

    if verbose:
        empty_after = (out[clean_output_col].str.len() == 0).sum()
        print(f"Total rows: {len(out):,}")
        print(f"Empty statements after cleaning: {empty_after:,}")
        print(f"pollution_repair={repair_polluted_statements}")
        print(f"stopword_removal={stopword_removal}, stemmer={stemmer}, lemmatizer={lemmatizer}")
        if vectorizer_type != "none":
            if vectorizer_type == "embeddings":
                print(f"Vectorization: {vectorizer_type} (model: {embedding_model})")
            else:
                print(f"Vectorization: {vectorizer_type}")

    return out


preprocess_statement = preprocess_statement_ds

__all__ = ["DEFAULT_SCALE", "StatementDSOptions", "preprocess_statement_ds", "preprocess_statement"]