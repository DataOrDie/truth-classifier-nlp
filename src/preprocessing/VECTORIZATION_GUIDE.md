# Statement Vectorization Guide

## Overview

The `statement.py` preprocessing module now supports configurable vectorization of the cleaned statement text. You can easily switch between three types of vectors: **TF-IDF**, **bigram**, and **binary** vectors.

## Quick Start

```python
from src.preprocessing.statement_ds import preprocess_statement

# With TF-IDF vectorization
result = preprocess_statement(
    df,
    vectorizer_type='tfidf',        # Enable TF-IDF
    vectorizer_max_features=100,    # Keep top 100 terms
)
```

## Vectorization Types

### 1. TF-IDF (`'tfidf'`)
- **What it does**: Computes TF-IDF (Term Frequency-Inverse Document Frequency) weighted scores for each term
- **Use case**: Best for machine learning models that benefit from weighted term importance
- **Output**: Decimal values (0.0 to ~1.0) representing importance of each term in each document
- **Features count**: One feature per unique term (up to `max_features`)

```python
result = preprocess_statement(df, vectorizer_type='tfidf')
```

### 2. Bigram (`'bigram'`)
- **What it does**: Counts both unigrams (single words) AND bigrams (word pairs)
- **Use case**: Captures word relationships and phrases, good for semantic meaning
- **Output**: Integer counts (0, 1, 2, ...) of how many times each term/phrase appears
- **Features count**: ~2x more features since it includes both unigrams and bigrams

```python
result = preprocess_statement(df, vectorizer_type='bigram')
```

### 3. Binary (`'binary'`)
- **What it does**: Creates binary (0/1) indicators of term presence
- **Use case**: Simple presence/absence features, efficient for sparse data
- **Output**: Binary values (0 or 1) indicating whether each term is present
- **Features count**: One feature per unique term (up to `max_features`)

```python
result = preprocess_statement(df, vectorizer_type='binary')
```

### 4. None (`'none'`)
- **What it does**: No vectorization applied
- **Use case**: Default behavior, just clean text and basic features
- **Output**: Only text and simple statistical features (word count, char length, etc.)

```python
result = preprocess_statement(df, vectorizer_type='none')
```

## Parameters

### Core Vectorization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `vectorizer_type` | str | `'none'` | Type: `'none'`, `'tfidf'`, `'bigram'`, or `'binary'` |
| `vectorizer_max_features` | int or None | None | Maximum features to extract (None = unlimited) |
| `vectorizer_min_df` | int | 1 | Minimum document frequency (ignore terms in <N docs) |
| `vectorizer_max_df` | float | 1.0 | Maximum document frequency ratio (ignore common terms) |
| `fitted_vectorizer` | object | None | Pre-fitted vectorizer for test data (internal use) |

### Example with Multiple Parameters

```python
result = preprocess_statement(
    df,
    vectorizer_type='tfidf',
    vectorizer_max_features=100,      # Keep top 100 terms
    vectorizer_min_df=2,               # Ignore terms appearing in <2 docs
    vectorizer_max_df=0.95,            # Ignore terms in >95% of docs
)
```

## Train-Test Split Handling

**Important**: When working with train/test splits, always use `preprocess_statement_train_test()` to prevent data leakage. The vectorizer will be fitted **only on training data** and then applied to test data.

```python
from src.preprocessing.statement_ds import preprocess_statement_train_test

train_proc, test_proc = preprocess_statement_train_test(
    train_df,
    test_df,
    vectorizer_type='tfidf',
    vectorizer_max_features=100,
)

# Both train_proc and test_proc will have identical vectorized features
# because the vectorizer was fitted on train_df only
```

## Output Columns

When vectorization is enabled, new columns are added with names following the pattern:

```
{output_col}_vec_{term}
```

For example, if `output_col='statement_clean'` and you have TF-IDF vectorization:
- `statement_clean_vec_president`
- `statement_clean_vec_said`
- `statement_clean_vec_percent`
- etc.

## Combining with Other Preprocessing Options

Vectorization works seamlessly with all other preprocessing options:

```python
result = preprocess_statement(
    df,
    # Text cleaning
    lower=True,
    remove_html=True,
    remove_urls=True,
    
    # Token-level processing
    stopword_removal=True,
    stemmer='porter',
    
    # Features
    add_rare_token_features=True,
    add_spelling_errors=True,
    
    # Vectorization
    vectorizer_type='tfidf',
    vectorizer_max_features=50,
)
```

## Feature Engineering Tips

### Reducing Sparsity
If you have too many sparse features, use these parameters:

```python
result = preprocess_statement(
    df,
    vectorizer_type='tfidf',
    vectorizer_max_features=50,       # Limit to 50 most important terms
    vectorizer_min_df=5,              # Only terms in 5+ documents
    vectorizer_max_df=0.8,            # Exclude very common terms (>80% docs)
)
```

### Capturing Phrases
For capturing multi-word expressions:

```python
result = preprocess_statement(
    df,
    vectorizer_type='bigram',        # Includes bigrams
    vectorizer_max_features=200,     # Bigrams create more features
)
```

### Efficient Binary Features
For memory-efficient sparse features:

```python
result = preprocess_statement(
    df,
    vectorizer_type='binary',         # Most memory efficient
    vectorizer_max_features=100,
)
```

## Example Workflows

### Minimal Setup
```python
train_proc, test_proc = preprocess_statement_train_test(
    train_df, test_df,
    vectorizer_type='tfidf'
)
```

### Production Setup with Tuning
```python
train_proc, test_proc = preprocess_statement_train_test(
    train_df, test_df,
    # Cleaning
    lower=True,
    remove_urls=True,
    replace_numbers=True,
    
    # Processing
    stopword_removal=True,
    stemmer='porter',
    
    # Features
    add_rare_token_features=True,
    
    # Vectorization
    vectorizer_type='tfidf',
    vectorizer_max_features=100,
    vectorizer_min_df=3,
    vectorizer_max_df=0.85,
)
```

## Technical Notes

- Vectorizers are fitted on **cleaned, processed text** (after all stemming, lemmatization, etc.)
- Scikit-learn's `TfidfVectorizer` and `CountVectorizer` are used under the hood
- Sparse matrices are automatically converted to dense for easier downstream use
- For very large datasets with many features, consider keeping sparse matrices for memory efficiency
- Vectorization requires scikit-learn (`sklearn`) to be installed

## Troubleshooting

### Too Many Features
If output has too many columns, reduce `vectorizer_max_features`:
```python
vectorizer_type='tfidf',
vectorizer_max_features=50,  # Instead of default (unlimited)
```

### Too Few Features
If important terms are being dropped, check `min_df` and `max_df`:
```python
vectorizer_type='tfidf',
vectorizer_min_df=1,          # Include rare terms
vectorizer_max_df=1.0,        # Include all terms
```

### No Features Generated
Make sure:
1. Your DataFrame has a `statement` column
2. Scikit-learn is installed: `pip install scikit-learn`
3. `vectorizer_type` is not `'none'`

## See Also

- [vectorization_example.ipynb](notebooks/vectorization_example.ipynb) - Detailed examples
- `src/preprocessing/statement.py` - Source code with full docstrings
