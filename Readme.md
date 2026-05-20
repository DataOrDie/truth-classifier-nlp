# False Political Claim Detection — NLP Kaggle Competition

Binary classification of political statements as **true (0)** or **false (1)**.
UPM university course project. Target metric: **Macro F1 score**.

[Competition link](https://www.kaggle.com/competitions/2025-26-false-political-claim-detection)

---

## Dataset

| File                         | Rows  | Description                |
| ---------------------------- | ----- | -------------------------- |
| `data/train.csv`             | 8,950 | Labeled samples            |
| `data/test_nolabel.csv`      | 3,860 | Unlabeled test samples     |
| `data/sample_submission.csv` | —     | Required submission format |

**Class distribution:** 35.25% true (0) · 64.75% false (1) — imbalanced.
Class weights used in training: `{0: 1.42, 1: 0.77}`.

**Columns:** `id`, `label`, `statement`, `subject`, `speaker`, `speaker_job`, `state_info`, `party_affiliation`

---

## Project Structure

```
truth-classifier-nlp/
├── src/
│   ├── preprocessing/      # Feature engineering pipeline
│   │   ├── one_step.py     # Orchestrator — start here
│   │   ├── statement_ds.py # TF-IDF, embeddings, NER, lexical features
│   │   ├── subject.py      # Multi-topic splitting & label encoding
│   │   ├── speaker.py      # Speaker encoding with rare-grouping
│   │   ├── speaker_job.py  # Occupation encoding
│   │   ├── party_affiliation.py
│   │   ├── state.py
│   │   ├── label.py
│   │   └── id.py           # MD5-based hash-bucket IDs
│   ├── training/           # Model training scripts (run directly with Python)
│   │   ├── stacking.py
│   │   ├── LGBMClassifier.py
│   │   ├── catboost.py
│   │   ├── randomForestClassifier.py
│   │   ├── logisticRegression.py
│   │   ├── transformer_kfold.py
│   │   ├── transformer_lora_kfold.py
│   │   └── transformer_hybrid.py  # (+ other transformer variants)
│   └── submit/
│       ├── kaggle.py              # Standard submission
│       └── kaggle-modulo.py       # With probability calibration
├── notebooks/
│   ├── AdvancedModels/    # DeBERTa, Mistral LoRA, Stacking ensembles
│   ├── LinearModels/      # Logistic Regression, Naive Bayes, SVM
│   ├── TreeModels/        # LightGBM, CatBoost, Random Forest, XGBoost
│   └── team_notebooks/    # Individual member exploration notebooks
├── models/                # Saved model artifacts (.joblib) — git-ignored
├── data/
└── wandb/                 # W&B experiment logs — git-ignored
```

---

## Preprocessing Pipeline

All feature engineering runs through `src/preprocessing/one_step.py` via `OneStepOptions`:

```python
from src.preprocessing.one_step import OneStepOptions

opts = OneStepOptions(
    statement_vectorization="tfidf",  # "tfidf" | "bigram" | "binary" | "none"
    subject_strategy="multi",         # "first" | "most_frequent" | "multi"
    speaker_min_count=5,
    # ... other per-module options
)
```

Each module supports named **presets** (e.g., `"minimal"`, `"expanded"`, `"rare_safe"`) that bundle common option combinations. See [PREPROCESSING_OPTIONS.md](PREPROCESSING_OPTIONS.md) for the full reference.

**Text processing options in `statement_ds.py`:**

- Vectorization: TF-IDF, bigram, binary bag-of-words
- Stemming: Porter, Snowball
- Lemmatization: WordNet
- Sentence embeddings: `all-MiniLM-L6-v2`
- NER, spelling error detection, lexical features

---

## Running Training

No build system. Run scripts directly:

```bash
python src/training/stacking.py
python src/training/LGBMClassifier.py
python src/training/transformer_kfold.py
```

Training uses **stratified k-fold cross-validation** and logs metrics and plots to W&B.
Run `wandb login` once before your first training run.

**Logged metrics:** Accuracy, F1, AUC, MCC, Balanced Accuracy
**Logged artifacts:** Confusion matrix, ROC/PR curves, model `.joblib` files → `models/`

---

## Generating a Submission

```bash
python src/submit/kaggle.py
```

Loads the saved model + `OneStepOptions` from `models/`, runs the same preprocessing on `data/test_nolabel.csv`, and writes a submission CSV.

Use `kaggle-modulo.py` for probability calibration/threshold adjustment.

---

## Models Explored

| Category       | Models                                                  |
| -------------- | ------------------------------------------------------- |
| Baselines      | DummyClassifier, Logistic Regression, Naive Bayes       |
| Tree-based     | Random Forest, Extra Trees, LightGBM, CatBoost, XGBoost |
| Text + tabular | TF-IDF + LR/SVM/NB, sentence embeddings + tree models   |
| Deep learning  | MLP, CNN, LSTM, BiLSTM, GRU                             |
| Transformers   | BERT, RoBERTa, DistilBERT, DeBERTa, Mistral (LoRA)      |
| Ensembles      | Voting, Stacking, threshold tuning                      |

**Current best baseline:** Stacking ensemble — Macro F1 = **0.6168**

---

## Key Dependencies

```
pandas · scikit-learn · scipy · lightgbm · catboost · xgboost
sentence-transformers · nltk · transformers · wandb · joblib · jupyter
```

> No `requirements.txt` exists yet. Install packages manually as needed.

---

## Team

Marilyn · Belen · Ivan · Rosali · Marla · Silvia
