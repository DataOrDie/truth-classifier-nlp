import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingGridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

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

text_cols = ["statement_clean"]
cat_cols = ["speaker_top", "speaker_job_top", "state_info_top", "party_top"]
num_cols = ["speaker_job_missing", "state_info_missing"] + subject_cols

X[text_cols] = X[text_cols].fillna("")
X_kaggle[text_cols] = X_kaggle[text_cols].fillna("")

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

print("Distribucion train:")
print(y_train.value_counts())

print("\nDistribucion valid:")
print(y_valid.value_counts())

print("\nDistribucion test:")
print(y_test.value_counts())

preprocessor = ColumnTransformer(
    transformers=[
        (
            "word_tfidf",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.90,
                max_features=30000,
                sublinear_tf=True,
            ),
            "statement_clean",
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

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LinearSVC(max_iter=7000)),
    ]
)

param_grid = {
    "classifier__C": [0.01, 0.1, 1, 3, 10],
    "classifier__loss": ["hinge", "squared_hinge"],
    "classifier__class_weight": [None, "balanced"],
}

search = HalvingGridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring="f1_macro",
    cv=3,
    factor=2,
    n_jobs=-1,
    verbose=1,
)

search.fit(X_train, y_train)

print("\nMejores parametros:")
print(search.best_params_)
print("Mejor F1 macro CV:", search.best_score_)

best_model = search.best_estimator_

feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
coefs = best_model.named_steps["classifier"].coef_[0]

coef_df = pd.DataFrame({
    "feature": feature_names,
    "coef": coefs,
})

print("\nEmpujan hacia clase 1:")
print(coef_df.sort_values("coef", ascending=False).head(30).to_string(index=False))

print("\nEmpujan hacia clase 0:")
print(coef_df.sort_values("coef", ascending=True).head(30).to_string(index=False))

calibrated_model = CalibratedClassifierCV(
    estimator=best_model,
    method="sigmoid",
    cv=3,
)

calibrated_model.fit(X_train, y_train)

valid_proba = calibrated_model.predict_proba(X_valid)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
scores = []

for threshold in thresholds:
    y_valid_pred = (valid_proba >= threshold).astype(int)
    scores.append(f1_score(y_valid, y_valid_pred, average="macro"))

best_idx = np.argmax(scores)
best_threshold = thresholds[best_idx]

print("\nMejor threshold:", best_threshold)
print("Mejor F1 macro valid:", scores[best_idx])

test_proba = calibrated_model.predict_proba(X_test)[:, 1]
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
plt.title(f"Matriz de confusion - threshold={best_threshold:.2f}")
plt.show()

final_model = search.best_estimator_

calibrated_final_model = CalibratedClassifierCV(
    estimator=final_model,
    method="sigmoid",
    cv=3,
)

calibrated_final_model.fit(X, y)

kaggle_proba = calibrated_final_model.predict_proba(X_kaggle)[:, 1]
kaggle_preds = (kaggle_proba >= best_threshold).astype(int)

submission = sample_submission.copy()

map_id_to_pred = dict(zip(test_ids, kaggle_preds))
submission["label"] = submission["id"].map(map_id_to_pred)

output_path = ROOT_DIR / "submission_linear_svm.csv"
submission.to_csv(output_path, index=False)

print("\nArchivo generado:")
print(output_path)

print("\nPrimeras filas submission:")
print(submission.head())

print("\nDistribucion de predicciones:")
print(submission["label"].value_counts())
