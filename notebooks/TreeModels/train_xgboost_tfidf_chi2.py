import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from xgboost import XGBClassifier

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_PREPROCESSING = ROOT_DIR / "src" / "preprocessing"

sys.path.append(str(SRC_PREPROCESSING))

from preprocessing import (
    download_nltk_resources,
    clean_category,
    preprocess_statement,
    keep_top_categories,
    TOP_PARTIES,
    NEGATIONS,
)

from nltk.corpus import stopwords


def parse_subjects(series):
    subject_list = (
        series
        .fillna("missing")
        .astype(str)
        .str.lower()
        .str.split(",")
    )
    return subject_list.apply(lambda xs: [x.strip() for x in xs])


def preprocess_train_test(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    test_ids = test_df["id"].copy()

    if "id" in train_df.columns:
        train_df = train_df.drop(columns=["id"])

    if "id" in test_df.columns:
        test_df = test_df.drop(columns=["id"])

    train_df["speaker_job_missing"] = train_df["speaker_job"].isnull().astype(int)
    train_df["state_info_missing"] = train_df["state_info"].isnull().astype(int)

    test_df["speaker_job_missing"] = test_df["speaker_job"].isnull().astype(int)
    test_df["state_info_missing"] = test_df["state_info"].isnull().astype(int)

    cat_cols_raw = [
        "subject",
        "speaker",
        "speaker_job",
        "state_info",
        "party_affiliation",
    ]

    for col in cat_cols_raw:
        train_df[col] = clean_category(train_df[col])
        test_df[col] = clean_category(test_df[col])

    stop_words = set(stopwords.words("english")) - NEGATIONS

    train_df["statement_clean"] = train_df["statement"].apply(
        lambda text: preprocess_statement(text, stop_words)
    )
    test_df["statement_clean"] = test_df["statement"].apply(
        lambda text: preprocess_statement(text, stop_words)
    )

    train_df["subject_list"] = parse_subjects(train_df["subject"])
    test_df["subject_list"] = parse_subjects(test_df["subject"])

    mlb = MultiLabelBinarizer()

    train_subject_encoded = mlb.fit_transform(train_df["subject_list"])
    test_subject_encoded = mlb.transform(test_df["subject_list"])

    subject_cols = [f"subject_{subject}" for subject in mlb.classes_]

    train_subject_df = pd.DataFrame(
        train_subject_encoded,
        columns=subject_cols,
        index=train_df.index,
    )

    test_subject_df = pd.DataFrame(
        test_subject_encoded,
        columns=subject_cols,
        index=test_df.index,
    )

    train_df = pd.concat([train_df, train_subject_df], axis=1)
    test_df = pd.concat([test_df, test_subject_df], axis=1)

    train_df["speaker_top"] = keep_top_categories(train_df["speaker"], top_n=50)
    train_top_speakers = set(train_df["speaker_top"].unique()) - {"other"}

    test_df["speaker_top"] = test_df["speaker"].apply(
        lambda x: x if x in train_top_speakers else "other"
    )

    train_df["speaker_job_top"] = keep_top_categories(train_df["speaker_job"], top_n=50)
    train_top_jobs = set(train_df["speaker_job_top"].unique()) - {"other"}

    test_df["speaker_job_top"] = test_df["speaker_job"].apply(
        lambda x: x if x in train_top_jobs else "other"
    )

    train_df["state_info_top"] = keep_top_categories(train_df["state_info"], top_n=30)
    train_top_states = set(train_df["state_info_top"].unique()) - {"other"}

    test_df["state_info_top"] = test_df["state_info"].apply(
        lambda x: x if x in train_top_states else "other"
    )

    train_df["party_top"] = train_df["party_affiliation"].apply(
        lambda x: x if x in TOP_PARTIES else "other"
    )

    test_df["party_top"] = test_df["party_affiliation"].apply(
        lambda x: x if x in TOP_PARTIES else "other"
    )

    return train_df, test_df, test_ids, subject_cols


download_nltk_resources()

train_raw = pd.read_csv(ROOT_DIR / "data" / "train.csv")
test_raw = pd.read_csv(ROOT_DIR / "data" / "test_nolabel.csv")
sample_submission = pd.read_csv(ROOT_DIR / "data" / "sample_submission.csv")

train_df, test_df, test_ids, subject_cols = preprocess_train_test(train_raw, test_raw)

feature_cols = [
    "statement_clean",
    "speaker_top",
    "speaker_job_top",
    "state_info_top",
    "party_top",
    "speaker_job_missing",
    "state_info_missing",
] + subject_cols

X = train_df[feature_cols].copy()
y = train_df["label"].copy()
X_kaggle = test_df[feature_cols].copy()

text_col = "statement_clean"
cat_cols = ["speaker_top", "speaker_job_top", "state_info_top", "party_top"]
num_cols = ["speaker_job_missing", "state_info_missing"] + subject_cols

X[text_col] = X[text_col].fillna("")
X_kaggle[text_col] = X_kaggle[text_col].fillna("")

for col in cat_cols:
    X[col] = X[col].fillna("missing").astype(str)
    X_kaggle[col] = X_kaggle[col].fillna("missing").astype(str)

X[num_cols] = (
    X[num_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype("int8")
)

X_kaggle[num_cols] = (
    X_kaggle[num_cols]
    .apply(pd.to_numeric, errors="coerce")
    .fillna(0)
    .astype("int8")
)

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.25,
    random_state=42,
    stratify=y_train_full,
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.90,
                max_features=80000,
                sublinear_tf=True,
            ),
            text_col,
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            cat_cols,
        ),
        (
            "num",
            "passthrough",
            num_cols,
        ),
    ]
)

print("Transformando features...")

X_train_vec = preprocessor.fit_transform(X_train)
X_valid_vec = preprocessor.transform(X_valid)
X_test_vec = preprocessor.transform(X_test)
X_kaggle_vec = preprocessor.transform(X_kaggle)

k = min(50000, X_train_vec.shape[1])

selector = SelectKBest(score_func=chi2, k=k)

X_train_sel = selector.fit_transform(X_train_vec, y_train)
X_valid_sel = selector.transform(X_valid_vec)
X_test_sel = selector.transform(X_test_vec)
X_kaggle_sel = selector.transform(X_kaggle_vec)

print("Features antes de chi2:", X_train_vec.shape[1])
print("Features despues de chi2:", X_train_sel.shape[1])

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb = XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_estimators=1200,
    early_stopping_rounds=40,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=scale_pos_weight,
)

param_distributions = {
    "max_depth": [2, 3, 4, 5, 6],
    "min_child_weight": [1, 3, 5, 7, 10],
    "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8, 0.9],
    "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
    "gamma": [0, 0.5, 1, 2, 5],
    "reg_alpha": [0, 0.01, 0.1, 0.5, 1],
    "reg_lambda": [0.5, 1, 2, 5, 10],
}

search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_distributions,
    n_iter=35,
    scoring="f1_macro",
    cv=3,
    random_state=42,
    n_jobs=1,
    verbose=2,
)

search.fit(
    X_train_sel,
    y_train,
    eval_set=[(X_valid_sel, y_valid)],
    verbose=False,
)

print("\nMejores parametros:")
print(search.best_params_)
print("Mejor F1 macro CV:", search.best_score_)

best_model = search.best_estimator_

valid_proba = best_model.predict_proba(X_valid_sel)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
scores = []

for threshold in thresholds:
    y_valid_pred = (valid_proba >= threshold).astype(int)
    scores.append(f1_score(y_valid, y_valid_pred, average="macro"))

best_idx = np.argmax(scores)
best_threshold = thresholds[best_idx]

print("\nMejor threshold:", best_threshold)
print("Mejor F1 macro valid:", scores[best_idx])

test_proba = best_model.predict_proba(X_test_sel)[:, 1]
y_pred = (test_proba >= best_threshold).astype(int)

print("\nRESULTADO TEST")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("F1 macro:", f1_score(y_test, y_pred, average="macro"))
print("F1 weighted:", f1_score(y_test, y_pred, average="weighted"))
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[0, 1],
)

disp.plot(cmap="Blues")
plt.title(f"Matriz de confusion XGBoost - threshold={best_threshold:.2f}")
plt.show()

kaggle_proba = best_model.predict_proba(X_kaggle_sel)[:, 1]
kaggle_preds = (kaggle_proba >= best_threshold).astype(int)

submission = sample_submission.copy()
submission["label"] = submission["id"].map(dict(zip(test_ids, kaggle_preds)))

output_path = ROOT_DIR / "submission_xgboost_tfidf_chi2.csv"
submission.to_csv(output_path, index=False)

print("\nArchivo generado:")
print(output_path)

print("\nPrimeras filas submission:")
print(submission.head())

print("\nDistribucion de predicciones:")
print(submission["label"].value_counts())
